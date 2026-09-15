"""Calcola il voto dei soggetti, confronta i modelli e scrive i risultati."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def majority_vote(task_predictions, tie_class=0):
    """Riduce le decisioni dei task a una decisione per soggetto.

    Input: tabella con subject_id, label e prediction (0 oppure 1).
    Output: una riga per soggetto con conteggio dei voti e decisione finale.
    """
    rows = []
    for subject_id, group in task_predictions.groupby("subject_id", sort=True):
        if group["label"].nunique() != 1:
            raise ValueError("Etichette discordanti nei task di un soggetto.")
        # Le predizioni sono 0/1: la loro somma coincide con i voti per 1.
        positive_votes = int(group["prediction"].sum())
        negative_votes = len(group) - positive_votes
        tie = positive_votes == negative_votes
        # La regola di parita' e' fissata in config.py e non usa la label reale.
        prediction = tie_class if tie else int(positive_votes > negative_votes)
        rows.append({"subject_id": subject_id, "label": int(group["label"].iloc[0]),
                     "task_count": len(group), "class_0_votes": negative_votes,
                     "class_1_votes": positive_votes, "tie": tie, "prediction": prediction})
    return pd.DataFrame(rows)


def compute_metrics(labels, predictions):
    """Calcola le metriche considerando 1 come classe positiva."""
    # zero_division=0 assegna zero alle metriche con denominatore nullo.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0)
    # Righe: classi di riferimento. Colonne: classi predette. Ordine: [0, 1].
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"count": len(labels), "accuracy": accuracy_score(labels, predictions),
            "precision": precision, "recall": recall, "f1": f1,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate_models(metadata, svm_predictions, lstm_predictions, tie_class=0):
    """Confronta entrambi i modelli sugli stessi task e sugli stessi soggetti.

    Restituisce tre tabelle: decisioni dei task, voto dei soggetti e metriche.
    Non esegue training e non sceglie parametri in base ai risultati del test.
    """
    task_results = metadata[["subject_id", "task_id", "label"]].copy()
    task_results["svm_prediction"] = svm_predictions
    task_results["lstm_prediction"] = lstm_predictions
    metric_rows, voting_tables = [], []

    for name in ["svm", "lstm"]:
        # Per ogni modello si usa la stessa funzione di voting e valutazione.
        current_tasks = task_results[["subject_id", "label"]].copy()
        current_tasks["prediction"] = task_results[f"{name}_prediction"]
        votes = majority_vote(current_tasks, tie_class)
        for level, frame in [("task", current_tasks), ("subject", votes)]:
            metric_rows.append({"model": name.upper(), "level": level,
                                **compute_metrics(frame["label"], frame["prediction"])})
        # I prefissi identificano il modello; ID, label e numero di task sono comuni.
        voting_tables.append(votes.rename(columns={
            column: f"{name}_{column}" for column in ["class_0_votes", "class_1_votes", "tie", "prediction"]
        }))

    subject_results = voting_tables[0].merge(
        voting_tables[1], on=["subject_id", "label", "task_count"], validate="one_to_one")
    return task_results, subject_results, pd.DataFrame(metric_rows)


def save_results(output_dir, task_results, subject_results, metrics, subject_split, history, summary):
    """Scrive sei file di testo/CSV; plots.py aggiunge il settimo file PNG.

    La cartella e i nomi sono fissi. Ogni esecuzione completata aggiorna questi
    file; non genera un'altra cartella run_... e non cancella altri file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Si salvano solo i dati necessari per leggere e ricostruire il confronto.
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    task_results.to_csv(output_dir / "task_predictions.csv", index=False)
    subject_results.to_csv(output_dir / "subject_predictions.csv", index=False)
    subject_split.to_csv(output_dir / "subject_split.csv", index=False)
    history.to_csv(output_dir / "lstm_history.csv", index=False)

    best_row = history.loc[history["epoch"].eq(summary["best_epoch"])].iloc[0]
    lines = ["CONFRONTO SVM / LSTM", f"Esecuzione: {summary['created_utc']}", "",
             f"Dataset: {summary['subject_count']} soggetti, {summary['task_count']} task utilizzati.",
             f"CSV disponibili: {summary['csv_count']}; vuoti: {summary['empty_csv']}; non validi: {summary['invalid_csv']}.",
             "Le classi sono indicate con 0 e 1. Il significato e' descritto nel README.",
             f"Diagnosi mancanti o vuote assegnate alla classe 0: {summary['default_class_0_subjects']} soggetti.",
             "", "SUDDIVISIONE DEI DATI"]
    for name in ["train", "validation", "test"]:
        part = subject_split[subject_split["split"].eq(name)]
        lines.append(f"{name}: {len(part)} soggetti, {int(part['task_count'].sum())} task; "
                     f"classe 0: {int(part.label.eq(0).sum())}, classe 1: {int(part.label.eq(1).sum())}.")
    lines += ["", "MODELLI SELEZIONATI",
              f"SVM: kernel RBF, C={summary['svm_C']}, gamma={summary['svm_gamma']}, {summary['feature_count']} feature.",
              f"LSTM: epoca {summary['best_epoch']}; F1 di validation={best_row.validation_f1:.4f}; "
              f"loss di validation={best_row.validation_loss:.4f}.",
              "Il criterio di selezione e' F1 dei task di validation. Per la LSTM, a parita' di F1, si sceglie la loss minore.",
              "I task di uno stesso soggetto appartengono a un solo insieme. I due modelli usano la stessa suddivisione.",
              "", "METRICHE SUL TEST", metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
              "", "COME LEGGERE LE METRICHE",
              "count = numero di task o soggetti valutati. La classe positiva e' 1.",
              "Accuracy = frazione di classificazioni corrette.",
              "Precision = frazione dei predetti 1 che hanno etichetta 1.",
              "Recall = frazione degli esempi con etichetta 1 riconosciuti come 1.",
              "F1 = media armonica di precision e recall. Un valore 0.80 corrisponde all'80%.",
              "TN: 0 correttamente classificati; FP: 0 classificati come 1; FN: 1 classificati come 0; TP: 1 correttamente classificati.",
              "I risultati per soggetto derivano dal voto dei task. In parita' si usa la classe fissata nei parametri.",
              "Il numero ridotto di soggetti di test limita la stabilita' delle percentuali.",
              "", "CONTENUTO DELLA CARTELLA RESULTS",
              "summary.txt: questo riepilogo, con parametri e significato dei risultati.",
              "metrics.csv: le quattro righe di confronto (due modelli, due livelli).",
              "task_predictions.csv: una riga per CSV del test, con label e decisioni dei due modelli.",
              "subject_predictions.csv: una riga per soggetto del test, con voti e decisioni finali.",
              "subject_split.csv: ID e gruppo di appartenenza di ogni soggetto.",
              "lstm_history.csv: loss di training, loss e F1 di validation per tutte le epoche.",
              "confusion_matrices.png: quattro matrici con riferimento sulle righe e predizione sulle colonne.",
              "", "PARAMETRI DELL'ESECUZIONE"]
    lines.extend(f"{name} = {value}" for name, value in summary["settings"].items())
    lines += ["", "CONFIGURAZIONI SVM PROVATE SU VALIDATION"]
    lines.extend(f"C={row['C']}, gamma={row['gamma']}, F1={row['validation_f1']:.4f}"
                 for row in summary["svm_trials"])
    lines += ["", "AMBIENTE", f"Python {summary['python']} - {summary['system']}"]
    lines.extend(f"{name}: {value}" for name, value in summary["packages"].items())
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
