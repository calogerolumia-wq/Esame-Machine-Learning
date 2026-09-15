"""Parametri dell'esperimento, raccolti in un unico file."""

from dataclasses import dataclass


@dataclass
class Settings:
    """Modificare qui le impostazioni, poi eseguire nuovamente main.py."""

    seed: int = 42                    # Seme per divisione dei soggetti e training.
    test_fraction: float = 0.20       # Quota dei soggetti riservata al test finale.
    validation_fraction: float = 0.20 # Quota del totale usata per scegliere i modelli.
    sample_interval: float = 0.005    # Secondi tra campioni nell'asse temporale nominale.
    sequence_length: int = 256        # Numero di istanti che rappresentano ogni CSV nella LSTM.
    hidden_size: int = 32             # Dimensione dello stato nascosto della LSTM.
    batch_size: int = 16              # Task elaborati insieme a ogni aggiornamento dei pesi.
    epochs: int = 40                  # Passaggi completi sul training; viene scelta l'epoca migliore.
    learning_rate: float = 0.001      # Passo di aggiornamento dell'ottimizzatore Adam.
    gradient_clip: float = 1.0        # Limite alla norma del gradiente per contenerne la crescita.
    svm_c_values: tuple = (0.1, 1.0, 10.0)        # Penalizzazioni degli errori da confrontare.
    svm_gamma_values: tuple = ("scale", 0.01, 0.1) # Parametri del kernel RBF da confrontare.
    tie_class: int = 0                # Classe scelta in caso di parita' nel voto dei task.
    cpu_threads: int = 2              # Numero di thread usati da PyTorch sulla CPU.
    device: str = "cpu"               # Dispositivo di calcolo; "cuda" richiede una GPU compatibile.
    progress_interval: int = 10       # Stampa una riga ogni 10 epoche, oltre alla prima e all'ultima.
