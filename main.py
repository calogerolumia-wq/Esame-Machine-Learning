"""Esegue il confronto e aggiorna i sette file della cartella results.

Avvio: python main.py
I parametri modificabili sono raccolti in config.py.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch

from config import Settings
from data_loader import load_dataset
from evaluation import evaluate_models, save_results
from feature_engineering import extract_features
from models import predict_lstm, set_seed, train_lstm, train_svm
from plots import plot_confusion_matrices
from preprocessing import resample_sequence, split_subjects, standardize_sequences


def main():
    """Carica i task, addestra i modelli e valuta lo stesso insieme di test."""
    # I percorsi dipendono dalla posizione di main.py, non dalla cartella
    # da cui viene aperto il terminale. Non vengono create sottocartelle run_*.
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    output_dir = root / "results"
    settings = Settings()
    if settings.epochs < 1:
        raise ValueError("Il numero di epoche deve essere almeno 1.")
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA non disponibile: impostare device='cpu' in config.py.")
    set_seed(settings.seed, settings.cpu_threads)
    device = torch.device(settings.device)

    # 1. Un elemento di tasks contiene una sola registrazione CSV completa.
    # subjects descrive le persone; audit permette di contare i CSV esclusi.
    print("Caricamento dei dati...", flush=True)
    tasks, subjects, audit = load_dataset(data_dir, settings)
    empty_count = int(audit["status"].eq("empty").sum())
    invalid_count = int(audit["status"].eq("invalid").sum())
    print(f"Dataset: {len(subjects)} soggetti, {len(tasks)} task utilizzati.")
    print(f"Classe 0: {int(subjects.label.eq(0).sum())} soggetti; "
          f"Classe 1: {int(subjects.label.eq(1).sum())} soggetti.")
    print(f"CSV esclusi: {empty_count} vuoti, {invalid_count} non validi.")

    # 2. metadata contiene una riga per task. La divisione viene invece
    # decisa sugli ID dei soggetti, poi estesa a tutti i loro task.
    metadata = pd.DataFrame([
        {key: task[key] for key in ["subject_id", "task_id", "label"]}
        for task in tasks
    ])
    if metadata.empty:
        raise ValueError("Il dataset non contiene task utilizzabili.")
    subject_split = split_subjects(metadata, settings)
    metadata["split"] = metadata["subject_id"].map(
        subject_split.set_index("subject_id")["split"])

    # 3. Si costruiscono due ingressi diversi sugli stessi task e nello
    # stesso ordine: matrice (task, 22) e tensore (task, 256, 5).
    feature_frame = pd.DataFrame([extract_features(task["series"]) for task in tasks])
    features = feature_frame.to_numpy()
    sequences = np.stack([
        resample_sequence(task["series"], settings.sequence_length) for task in tasks
    ])
    labels = metadata["label"].to_numpy()
    train_mask = metadata["split"].eq("train").to_numpy()
    validation_mask = metadata["split"].eq("validation").to_numpy()
    test_mask = metadata["split"].eq("test").to_numpy()
    for name, mask in [("Training", train_mask), ("Validation", validation_mask), ("Test", test_mask)]:
        print(f"{name}: {metadata.loc[mask, 'subject_id'].nunique()} soggetti, {int(mask.sum())} task")

    # 4. La SVM confronta C e gamma sulla validation. Lo scaler delle feature
    # fa parte della sua pipeline e viene stimato soltanto sul training.
    print("\nSVM: addestramento e selezione dei parametri...", flush=True)
    svm_model, svm_trials = train_svm(
        features[train_mask], labels[train_mask],
        features[validation_mask], labels[validation_mask], settings)
    selected_svm = svm_model.named_steps["svc"]
    print(f"SVM selezionata: C={selected_svm.C}, gamma={selected_svm.gamma}.")

    # 5. Anche i canali delle sequenze sono standardizzati sul solo training.
    # Lo scaler serve al calcolo; non occorre salvarlo per questo confronto.
    train_sequences, validation_sequences, test_sequences, _ = standardize_sequences(
        sequences[train_mask], sequences[validation_mask], sequences[test_mask])
    print("\nLSTM: addestramento...", flush=True)
    lstm_model, history, best_epoch = train_lstm(
        train_sequences, labels[train_mask], validation_sequences,
        labels[validation_mask], settings, device)

    # 6. Il test viene utilizzato dopo la selezione di entrambi i modelli.
    # Per la LSTM una probabilita' >= 0.5 produce la classe 1.
    test_metadata = metadata.loc[test_mask].reset_index(drop=True)
    svm_predictions = svm_model.predict(features[test_mask])
    lstm_probabilities = predict_lstm(lstm_model, test_sequences, settings.batch_size, device)
    lstm_predictions = (lstm_probabilities >= 0.5).astype(int)
    task_results, subject_results, metrics = evaluate_models(
        test_metadata, svm_predictions, lstm_predictions, settings.tie_class)

    # 7. Nel file dei soggetti si conservano anche l'ID originale e il numero
    # dei task, cosi' S01, S02, ecc. restano riconducibili alle cartelle dei dati.
    task_counts = metadata.groupby("subject_id").size()
    subject_split = subject_split.merge(
        subjects[["subject_id", "source_id"]], on="subject_id", validate="one_to_one")
    subject_split["task_count"] = subject_split["subject_id"].map(task_counts)

    # Parametri e controlli aggregati confluiscono nel riepilogo leggibile,
    # evitando molti piccoli file JSON e inventari separati.
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "subject_count": len(subjects), "task_count": len(tasks),
        "csv_count": len(audit), "empty_csv": empty_count, "invalid_csv": invalid_count,
        "default_class_0_subjects": int(subjects["label_source"].eq("default_class_0").sum()),
        "feature_count": features.shape[1], "best_epoch": best_epoch,
        "svm_C": selected_svm.C, "svm_gamma": selected_svm.gamma,
        "svm_trials": svm_trials.to_dict("records"), "settings": asdict(settings),
        "python": platform.python_version(), "system": platform.system(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "torch", "matplotlib"]},
    }
    save_results(output_dir, task_results, subject_results, metrics, subject_split, history, summary)
    plot_confusion_matrices(metrics, output_dir)

    # Il terminale mostra una tabella breve; i CSV conservano tutta la precisione.
    console_table = metrics[["model", "level", "count", "accuracy", "precision", "recall", "f1"]].copy()
    console_table["level"] = console_table["level"].map({"task": "Task", "subject": "Soggetti"})
    console_table = console_table.rename(columns={"model": "Modello", "level": "Livello", "count": "N",
                                                "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"})
    print("\nRisultati sul test:")
    print(console_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nRisultati aggiornati in: {output_dir}")
    print("Apri summary.txt per il riepilogo e il significato dei file.")


if __name__ == "__main__":
    main()
