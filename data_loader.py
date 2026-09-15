"""Lettura delle anagrafiche e dei singoli task di handwriting."""

import csv
from pathlib import Path

import pandas as pd


def read_subject_info(path):
    """Restituisce ID originale e campo diagnostico da un'anagrafica.

    ID serve a riconoscere il soggetto; il campo diagnostico determina la label.
    Nessuno dei due viene inserito nelle feature numeriche dei modelli.
    """
    fields = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if ":" in line:
            # Si separa soltanto al primo ':', preservando eventuali ':' nel valore.
            key, value = line.split(":", 1)
            fields[key.strip().casefold()] = value.strip()
    return fields.get("id", ""), fields.get("malattie diagnosticate")


def diagnosis_to_label(diagnosis):
    """Applica la codifica binaria; un campo assente o vuoto riceve classe 0.

    Restituisce la label e la sua origine per il riepilogo interno dei dati.
    La corrispondenza fra codici e classi e' riportata nel README.
    """
    # Normalizza maiuscole e spazi. None, stringa vuota e soli spazi
    # vengono trattati nello stesso modo, secondo la regola del progetto.
    value = " ".join((diagnosis or "").casefold().split())
    if value == "":
        return 0, "default_class_0"
    if value in {"autismo", "autistico", "autistica", "asd", "disturbo dello spettro autistico"}:
        return 1, "explicit_class_1"
    if value in {"nessuno", "nessuna", "non autismo", "non autistico", "non autistica"}:
        return 0, "explicit_class_0"
    raise ValueError(f"Diagnosi non riconosciuta: {diagnosis!r}")


def read_task_csv(path):
    """Legge i cinque campi iniziali, stabili anche nei CSV con virgole decimali.

    PointDisplayX/Y possono contenere virgole decimali non racchiuse fra
    virgolette: le righe hanno allora 19 o 20 campi, contro i 18 dell'header.
    Timestamp, PointX, PointY, Phase e Pressure precedono quei campi.
    csv.reader permette di leggerli senza spostare implicitamente le colonne.
    """
    records = []
    expanded_rows = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if not header:
            return pd.DataFrame(), {"raw_rows": 0, "expanded_rows": 0}
        expected = ["Timestamp", "PointX", "PointY", "Phase", "Pressure"]
        if [value.strip() for value in header[:5]] != expected:
            raise ValueError(f"Intestazione non riconosciuta: {Path(path).name}")
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) not in {18, 19, 20}:
                raise ValueError(f"{Path(path).name}, riga {line_number}: formato inatteso.")
            expanded_rows += len(row) > len(header)
            records.append([value.strip() for value in row[:5]])
    frame = pd.DataFrame(records, columns=["timestamp", "x", "y", "phase", "pressure"])
    # I valori non convertibili diventano NaN: preprocessing.py li gestira'
    # all'interno della singola serie, senza mescolare task diversi.
    for column in ["x", "y", "pressure"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, {"raw_rows": len(records), "expanded_rows": int(expanded_rows)}


def load_dataset(data_dir, settings):
    """Carica le registrazioni e restituisce tasks, subjects e audit.

    tasks: lista di serie complete con ID del soggetto, nome CSV e label.
    subjects: una riga per persona, con ID originale e numero di task validi.
    audit: una riga per CSV, utile a contare quelli vuoti o non utilizzabili.
    """
    from preprocessing import clean_task

    root = Path(data_dir)
    # L'ordine fisso delle cartelle rende stabile l'associazione S01, S02, ...
    # quando il contenuto del dataset non cambia.
    info_paths = sorted(root.rglob("Anagrafica*.txt"))
    if not info_paths:
        raise ValueError("Nessuna anagrafica trovata nella cartella data.")
    tasks, subjects, audit = [], [], []
    seen_ids = set()

    for subject_number, info_path in enumerate(info_paths, start=1):
        subject_id = f"S{subject_number:02d}"
        source_id, diagnosis = read_subject_info(info_path)
        if not source_id or source_id in seen_ids:
            raise ValueError("ID assente o ripetuto: verificare le cartelle dei soggetti.")
        seen_ids.add(source_id)
        label, label_source = diagnosis_to_label(diagnosis)

        csv_dirs = [p for p in info_path.parent.iterdir() if p.is_dir() and p.name.casefold() == "csv"]
        if len(csv_dirs) != 1:
            raise ValueError(f"{subject_id}: serve una cartella Csv accanto all'anagrafica.")
        csv_paths = sorted(csv_dirs[0].glob("*.csv"))
        valid_count = 0
        for csv_path in csv_paths:
            frame, read_info = read_task_csv(csv_path)
            entry = {"subject_id": subject_id, "task_id": csv_path.name, **read_info}
            if frame.empty:
                # Un CSV vuoto non e' un esempio di training.
                audit.append({**entry, "status": "empty", "valid_rows": 0})
                continue
            series, clean_info = clean_task(frame, settings.sample_interval)
            entry.update(clean_info)
            if series is None:
                audit.append({**entry, "status": "invalid"})
                continue
            valid_count += 1
            audit.append({**entry, "status": "included"})
            tasks.append({"subject_id": subject_id, "task_id": csv_path.name,
                          "label": label, "series": series})
        subjects.append({"subject_id": subject_id, "source_id": source_id,
                         "diagnosis": diagnosis, "label": label, "label_source": label_source,
                         "csv_count": len(csv_paths), "valid_tasks": valid_count})
    return tasks, pd.DataFrame(subjects), pd.DataFrame(audit)
