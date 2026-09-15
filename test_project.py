"""Controlli facoltativi sul codice, separati dal normale avvio di main.py.

Coprono gli errori che altererebbero il confronto: lettura delle colonne,
etichette, ordine temporale, separazione dei soggetti, scaling e voting.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import Settings
from data_loader import diagnosis_to_label, read_task_csv
from evaluation import evaluate_models, majority_vote
from feature_engineering import extract_features
from models import LSTMClassifier
from preprocessing import clean_task, resample_sequence, split_subjects, standardize_sequences


class ProjectTests(unittest.TestCase):
    """Ogni controllo costruisce un piccolo caso con risultato noto."""
    def test_missing_and_blank_diagnoses_are_non_autism(self):
        """Controlla che tutti i modi di rappresentare un campo vuoto diano 0."""
        self.assertEqual(diagnosis_to_label("autismo")[0], 1)
        self.assertEqual(diagnosis_to_label("nessuno")[0], 0)
        for value in [None, "", " ", "\t\n"]:
            self.assertEqual(diagnosis_to_label(value), (0, "default_class_0"))
        with self.assertRaises(ValueError):
            diagnosis_to_label("sospetto autismo")
        self.assertEqual(diagnosis_to_label("non autistico")[0], 0)

    def test_csv_decimal_commas_do_not_shift_coordinates(self):
        """Una riga con piu' campi dell'header deve mantenere X, Y e pressione."""
        header = "Timestamp,PointX,PointY,Phase,Pressure,PointDisplayX,PointDisplayY,PointRawX,PointRawY,PressureRaw,TimestampRaw,Sequence,Rotation,Azimuth,Altitude,TiltX,TiltY,PenId\n"
        row = "2025-03-22T18:10:58.8567634Z,10797,8809,MoveStroke,12000,2070,2957,574,6388,10797,8809,1500,49995,15162,0,143,51,24,32,0x00000001\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.csv"
            path.write_text(header + row, encoding="utf-8")
            frame, info = read_task_csv(path)
        self.assertEqual(frame.loc[0, "x"], 10797)
        self.assertEqual(frame.loc[0, "y"], 8809)
        self.assertEqual(frame.loc[0, "pressure"], 12000)
        self.assertEqual(info["expanded_rows"], 1)

    def test_temporal_order_and_interpolation_stay_inside_task(self):
        """Usa tre campioni fuori ordine con il valore centrale da ricostruire."""
        frame = pd.DataFrame({"timestamp": ["2025-01-01T00:00:00.010Z", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00.005Z"],
                              "x": [10.0, 0.0, np.nan], "y": [0.0, 0.0, 0.0],
                              "pressure": [10.0, 10.0, 10.0], "phase": ["EndStroke", "BeginStroke", "MoveStroke"]})
        series, info = clean_task(frame, 0.005)
        np.testing.assert_allclose(series["x"], [0, 5, 10])
        np.testing.assert_allclose(series["time_seconds"], [0, 0.005, 0.010])
        self.assertTrue(info["reordered"])
        sequence = resample_sequence(series, 8)
        self.assertEqual(sequence.shape, (8, 5))
        np.testing.assert_allclose(sequence[[0, -1], 1], [0, 10])

    def test_motion_does_not_include_jump_between_strokes(self):
        """Un grande salto tra due tratti non deve diventare velocita' di scrittura."""
        series = pd.DataFrame({"time_seconds": np.arange(6) * 0.005,
                               "x": [0, 1, 2, 10000, 10001, 10002], "y": np.zeros(6),
                               "pressure": np.ones(6), "pen_down": np.ones(6), "stroke_id": [1, 1, 1, 2, 2, 2]})
        features = extract_features(series)
        self.assertEqual(features["stroke_count"], 2)
        self.assertAlmostEqual(features["speed_mean"], 200)
        self.assertAlmostEqual(features["acceleration_max"], 0, places=6)
        self.assertEqual(len(features), 22)

    def test_no_subject_can_cross_partitions(self):
        """Tutti i task di una persona devono ricevere la stessa partizione."""
        metadata = pd.DataFrame([{"subject_id": f"S{i:02}", "label": i % 2, "task_id": task}
                                 for i in range(30) for task in [1, 2, 3]])
        split = split_subjects(metadata, Settings())
        self.assertEqual(len(split), 30)
        self.assertFalse(split["subject_id"].duplicated().any())
        expanded = metadata.merge(split[["subject_id", "split"]], on="subject_id")
        self.assertTrue(expanded.groupby("subject_id")["split"].nunique().eq(1).all())
        for _, part in split.groupby("split"):
            self.assertEqual(set(part.label), {0, 1})

    def test_test_values_do_not_change_scaler(self):
        """Valori estremi nella validation e nel test non devono alterare la media stimata."""
        train = np.array([[[0.0], [2.0]], [[4.0], [6.0]]], dtype=np.float32)
        validation = np.full((1, 2, 1), 100.0, dtype=np.float32)
        test = np.full((1, 2, 1), -100.0, dtype=np.float32)
        scaled_train, _, _, scaler = standardize_sequences(train, validation, test)
        self.assertAlmostEqual(scaler.mean_[0], 3.0)
        self.assertAlmostEqual(float(scaled_train.mean()), 0.0)
        self.assertEqual(scaled_train.shape, train.shape)

    def test_majority_vote_and_fixed_tie_rule(self):
        """Distingue una maggioranza effettiva da un pareggio risolto con classe 0."""
        predictions = pd.DataFrame({"subject_id": ["S1"] * 3 + ["S2"] * 2,
                                     "label": [1, 1, 1, 0, 0], "prediction": [1, 0, 1, 0, 1]})
        subjects = majority_vote(predictions)
        self.assertEqual(subjects.prediction.tolist(), [1, 0])
        self.assertEqual(subjects.tie.tolist(), [False, True])

    def test_lstm_outputs_one_probability_per_csv(self):
        """Tre sequenze devono produrre tre uscite e gradienti per il training."""
        torch.manual_seed(42)
        model = LSTMClassifier(5, 8)
        outputs = model(torch.randn(3, 12, 5))
        self.assertEqual(tuple(outputs.shape), (3,))
        self.assertTrue(bool(((outputs >= 0) & (outputs <= 1)).all()))
        outputs.sum().backward()
        self.assertIsNotNone(model.lstm.weight_ih_l0.grad)

    def test_combined_results_preserve_both_models_and_task_identity(self):
        """Controlla l'unione dei risultati senza perdere task o scambiare i modelli."""
        metadata = pd.DataFrame({"subject_id": ["S1"] * 3 + ["S2"] * 2,
                                 "task_id": ["1.csv", "2.csv", "3.csv", "1.csv", "2.csv"],
                                 "label": [1, 1, 1, 0, 0]})
        tasks, subjects, metrics = evaluate_models(metadata, [1, 0, 0, 0, 1], [1, 1, 0, 1, 1])
        self.assertEqual(len(tasks), 5)
        self.assertEqual(tasks.task_id.tolist(), metadata.task_id.tolist())
        self.assertEqual(subjects.svm_prediction.tolist(), [0, 0])
        self.assertEqual(subjects.lstm_prediction.tolist(), [1, 1])
        self.assertEqual(subjects.svm_tie.tolist(), [False, True])
        self.assertEqual(metrics["count"].tolist(), [5, 2, 5, 2])


if __name__ == "__main__":
    unittest.main()
