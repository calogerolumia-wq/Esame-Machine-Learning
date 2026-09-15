"""Trasforma ciascun CSV in un singolo vettore numerico per la SVM."""

import numpy as np


def extract_features(series):
    """Restituisce le 22 feature di un task sotto forma di dizionario ordinato.

    Le feature sono calcolate sulla serie pulita prima del ricampionamento.
    Coordinate e pressione rimangono nelle unita' del dispositivo.
    """
    features = {}
    # Quattro statistiche per ciascuno dei tre canali: 12 feature in totale.
    # Si considera tutto il task, inclusi gli istanti senza contatto.
    for column in ["x", "y", "pressure"]:
        values = series[column].to_numpy()
        features[f"{column}_mean"] = float(values.mean())
        features[f"{column}_std"] = float(values.std(ddof=0))  # Divisore N.
        features[f"{column}_min"] = float(values.min())
        features[f"{column}_max"] = float(values.max())

    time = series["time_seconds"].to_numpy()
    position = series[["x", "y"]].to_numpy()
    stroke_id = series["stroke_id"].to_numpy()
    time_step = np.diff(time)
    # Differenze tra posizioni consecutive divise per il tempo: vettori velocita'.
    velocity = np.diff(position, axis=0) / time_step[:, None]
    # Non si misura come scrittura il salto da uno stroke al successivo.
    valid_edges = (stroke_id[:-1] > 0) & (stroke_id[:-1] == stroke_id[1:])
    speed = np.linalg.norm(velocity[valid_edges], axis=1)
    # Per l'accelerazione servono tre campioni consecutivi dello stesso stroke.
    valid_acceleration = valid_edges[:-1] & valid_edges[1:]
    acceleration = np.linalg.norm(
        (np.diff(velocity, axis=0) / ((time_step[:-1] + time_step[1:]) / 2)[:, None])[valid_acceleration],
        axis=1)
    durations = []
    for current_id in np.unique(stroke_id[stroke_id > 0]):
        stroke_time = time[stroke_id == current_id]
        durations.append(stroke_time[-1] - stroke_time[0])

    # Se non esistono tratti abbastanza lunghi, velocita'/accelerazione
    # non calcolabili sono rappresentate da zero. Il numero di stroke resta esplicito.
    features["duration_seconds"] = float(time[-1] - time[0])
    features["speed_mean"] = float(speed.mean()) if speed.size else 0.0
    features["speed_max"] = float(speed.max()) if speed.size else 0.0
    features["acceleration_mean"] = float(acceleration.mean()) if acceleration.size else 0.0
    features["acceleration_max"] = float(acceleration.max()) if acceleration.size else 0.0
    features["stroke_count"] = len(durations)
    features["stroke_duration_mean"] = float(np.mean(durations)) if durations else 0.0
    features["stroke_duration_std"] = float(np.std(durations)) if durations else 0.0
    features["stroke_duration_max"] = float(np.max(durations)) if durations else 0.0
    features["pen_down_fraction"] = float(series["pen_down"].mean())
    return features
