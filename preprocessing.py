"""Pulizia di ciascuna serie, ricampionamento e divisione per soggetto."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEQUENCE_CHANNELS = ["time_seconds", "x", "y", "pressure", "pen_down"]


def clean_task(frame, sample_interval):
    """Ordina un task e ricostruisce l'asse temporale nominale a 5 ms.

    I timestamp del computer servono all'ordinamento. Non sono utilizzati
    come intervalli di campionamento: possono registrare eventi a raffiche.
    La durata e le derivate usano l'intervallo nominale dichiarato.
    """
    if sample_interval <= 0:
        raise ValueError("sample_interval deve essere positivo.")
    frame = frame.copy()
    # Timestamp serve per l'ordine dei campioni. Il tempo numerico della serie
    # viene costruito dopo, usando l'intervallo nominale configurato.
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="ISO8601", utc=True, errors="coerce")
    invalid_timestamps = int(frame["timestamp"].isna().sum())
    frame = frame.dropna(subset=["timestamp"])
    reordered = not frame["timestamp"].is_monotonic_increasing
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    info = {"invalid_timestamps": invalid_timestamps, "reordered": reordered,
            "valid_rows": len(frame), "interpolated_values": 0}
    if len(frame) < 3:
        return None, {**info, "reason": "less_than_three_samples"}

    columns = ["x", "y", "pressure"]
    frame[columns] = frame[columns].replace([np.inf, -np.inf], np.nan)
    frame.loc[frame["pressure"] < 0, "pressure"] = np.nan
    info["interpolated_values"] = int(frame[columns].isna().sum().sum())
    if frame[columns].isna().all().any():
        return None, {**info, "reason": "missing_numeric_channel"}
    # La ricostruzione utilizza solo i campioni dello stesso task.
    frame[columns] = frame[columns].interpolate(limit_direction="both")
    phase = frame["phase"].str.casefold()
    if not phase.isin(["hover", "beginstroke", "movestroke", "endstroke"]).all():
        raise ValueError("Phase contiene valori non riconosciuti.")

    # pen_down indica il contatto; stroke_id distingue i singoli tratti.
    # BeginStroke e il campione dopo EndStroke iniziano un nuovo tratto
    # anche quando manca un campione Hover fra due tratti consecutivi.
    pen_down = phase.ne("hover").to_numpy()
    previous_down = np.r_[False, pen_down[:-1]]
    previous_end = np.r_[False, phase.eq("endstroke").to_numpy()[:-1]]
    starts = pen_down & (~previous_down | phase.eq("beginstroke").to_numpy() | previous_end)
    stroke_id = np.where(pen_down, np.cumsum(starts), 0)
    series = frame[columns].copy()
    series.insert(0, "time_seconds", np.arange(len(frame)) * sample_interval)
    series["pen_down"] = pen_down.astype(float)
    series["stroke_id"] = stroke_id
    info["duration_seconds"] = float(series["time_seconds"].iloc[-1])
    info["reason"] = ""
    return series, info


def resample_sequence(series, sequence_length):
    """Restituisce un array (sequence_length, 5) che rappresenta l'intero task.

    I canali sono tempo relativo, X, Y, pressione e contatto. Inizio e fine
    rimangono inclusi; le serie lunghe vengono compresse, non tagliate all'inizio.
    """
    if sequence_length < 3:
        raise ValueError("sequence_length deve essere almeno 3.")
    source_time = series["time_seconds"].to_numpy()
    # La griglia dipende solo dalla durata di questo task, non dagli altri soggetti.
    target_time = np.linspace(source_time[0], source_time[-1], sequence_length)
    values = [target_time]
    for column in ["x", "y", "pressure"]:
        values.append(np.interp(target_time, source_time, series[column]))
    # Il contatto resta binario: si prende il campione temporalmente piu' vicino.
    nearest = np.rint(target_time / (source_time[1] - source_time[0])).astype(int)
    nearest = np.clip(nearest, 0, len(series) - 1)
    values.append(series["pen_down"].to_numpy()[nearest])
    return np.column_stack(values).astype(np.float32)


def standardize_sequences(train_sequences, validation_sequences, test_sequences):
    """Standardizza ogni canale con media e deviazione standard del training.

    Riceve tre tensori (task, istanti, canali) e restituisce gli stessi tre
    tensori trasformati, insieme allo scaler utilizzato.
    """
    channel_count = train_sequences.shape[-1]
    scaler = StandardScaler()
    scaler.fit(train_sequences.reshape(-1, channel_count))

    def transform(sequences):
        """Applica le statistiche gia' stimate e ripristina la forma originale."""
        # Il reshape serve solo allo scaler; ogni task conserva il proprio asse.
        return scaler.transform(sequences.reshape(-1, channel_count)).reshape(sequences.shape).astype(np.float32)

    return transform(train_sequences), transform(validation_sequences), transform(test_sequences), scaler


def split_subjects(metadata, settings):
    """Divide gli ID unici e restituisce subject_id, label e split.

    stratify mantiene, per quanto possibile, la proporzione tra le due classi.
    Dividere le singole righe dei task potrebbe far comparire la stessa persona
    sia nel training sia nel test: per questo si estraggono prima gli ID unici.
    """
    subjects = metadata[["subject_id", "label"]].drop_duplicates().sort_values("subject_id")
    if subjects["subject_id"].duplicated().any():
        raise ValueError("Uno stesso soggetto ha etichette discordanti.")
    counts = subjects["label"].value_counts()
    if len(counts) != 2 or counts.min() < 3:
        raise ValueError(
            "Servono almeno 3 soggetti per classe per train/validation/test."
        )
    if not (0 < settings.test_fraction < 1 and 0 < settings.validation_fraction < 1
            and settings.test_fraction + settings.validation_fraction < 1):
        raise ValueError("Frazioni di suddivisione non valide.")
    development, test = train_test_split(
        subjects, test_size=settings.test_fraction, random_state=settings.seed,
        stratify=subjects["label"])
    # Dopo aver separato il test, il 20% del totale equivale al 25% del restante 80%.
    relative_validation = settings.validation_fraction / (1 - settings.test_fraction)
    train, validation = train_test_split(
        development, test_size=relative_validation, random_state=settings.seed,
        stratify=development["label"])
    parts = []
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        if part["label"].nunique() != 2:
            raise ValueError(f"Entrambe le classi devono essere presenti in {name}.")
        parts.append(part.assign(split=name))
    return pd.concat(parts).sort_values("subject_id").reset_index(drop=True)
