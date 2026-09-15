"""SVM con kernel RBF e classificatore LSTM many-to-one."""

import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed, cpu_threads=2):
    """Fissa le principali sorgenti casuali per rendere ripetibile il confronto.

    Versioni delle librerie e piattaforme diverse possono comunque introdurre
    differenze numeriche, soprattutto durante il training della rete.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_svm(train_features, train_labels, validation_features, validation_labels, settings):
    """Seleziona C e gamma sul validation set; il test non entra nella scelta.

    Ogni riga delle matrici di input rappresenta un task con 22 feature.
    Restituisce la pipeline selezionata e la tabella delle configurazioni provate.
    """
    best_model, best_score, trials = None, -1.0, []
    for c_value in settings.svm_c_values:
        for gamma_value in settings.svm_gamma_values:
            # fit stima prima lo scaler e poi la SVM, sempre sugli stessi task
            # di training. predict riusa quello scaler sulla validation.
            model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c_value, gamma=gamma_value))
            model.fit(train_features, train_labels)
            predictions = model.predict(validation_features)
            score = f1_score(validation_labels, predictions, zero_division=0)
            trials.append({"C": c_value, "gamma": gamma_value, "validation_f1": score})
            # In caso di parita' si mantiene la prima configurazione della griglia.
            if score > best_score:
                best_model, best_score = model, score
    return best_model, pd.DataFrame(trials)


class LSTMClassifier(nn.Module):
    """Classificatore many-to-one: una sequenza produce una probabilita'.

    Ingresso: (batch, istanti, canali). Uscita: (batch,).
    L'unico strato LSTM mantiene 32 componenti nello stato nascosto predefinito.
    """

    def __init__(self, input_size, hidden_size):
        """Crea LSTM, trasformazione lineare e sigmoide finale."""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequences):
        """Calcola la probabilita' della classe 1 per ogni task del batch."""
        # Non viene fornito uno stato precedente: ogni sequenza parte da zero.
        # Lo stato nascosto non viene trasferito da un CSV al successivo.
        _, (hidden_state, _) = self.lstm(sequences)
        # hidden_state[-1] riassume l'intera sequenza dell'unico strato.
        return self.sigmoid(self.fc(hidden_state[-1])).squeeze(1)


def predict_lstm(model, sequences, batch_size, device):
    """Restituisce le probabilita' nello stesso ordine delle sequenze ricevute."""
    loader = DataLoader(TensorDataset(torch.from_numpy(sequences)), batch_size=batch_size, shuffle=False)
    model.eval()  # Modalita' di valutazione: non vengono aggiornati i pesi.
    probabilities = []
    with torch.no_grad():  # Non serve costruire il grafo dei gradienti per predire.
        for (inputs,) in loader:
            probabilities.extend(model(inputs.to(device)).cpu().numpy())
    return np.asarray(probabilities)


def train_lstm(train_sequences, train_labels, validation_sequences, validation_labels, settings, device):
    """Addestra la rete e conserva l'epoca migliore in validazione.

    Restituisce il modello gia' riportato ai pesi scelti, tutte le misure per
    epoca e il numero dell'epoca selezionata. Il test non viene passato qui.
    """
    model = LSTMClassifier(train_sequences.shape[-1], settings.hidden_size).to(device)
    # Ogni elemento del dataset e' la coppia (intero task, label del soggetto).
    # shuffle cambia l'ordine dei task tra i batch, non quello degli istanti.
    dataset = TensorDataset(torch.from_numpy(train_sequences), torch.tensor(train_labels, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(settings.seed))
    criterion = nn.BCELoss()  # Errore fra probabilita' sigmoide e label binaria.
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    best_key, best_state, best_epoch = (-1.0, -float("inf")), None, 0
    history = []

    for epoch in range(1, settings.epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Azzera i gradienti del batch precedente.
            probabilities = model(inputs)
            loss = criterion(probabilities, labels)
            loss.backward()  # Propaga l'errore attraverso la rete e gli istanti.
            nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimizer.step()  # Aggiorna i pesi usando i gradienti calcolati.
            # Pesa la loss con la dimensione del batch: l'ultimo puo' essere piu' piccolo.
            total_loss += loss.item() * len(inputs)

        # La validation misura il modello aggiornato, senza modificarne i pesi.
        validation_probabilities = predict_lstm(model, validation_sequences, settings.batch_size, device)
        validation_loss = criterion(torch.tensor(validation_probabilities),
                                    torch.tensor(validation_labels, dtype=torch.float32)).item()
        validation_f1 = f1_score(validation_labels, validation_probabilities >= 0.5, zero_division=0)
        train_loss = total_loss / len(dataset)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss, "validation_f1": validation_f1})
        # Prima F1 dei task, poi loss per risolvere eventuali parita'.
        current_key = (validation_f1, -validation_loss)
        if current_key > best_key:
            best_key, best_epoch = current_key, epoch
            # Una copia indipendente evita che le epoche successive sovrascrivano
            # anche i pesi dell'epoca che vogliamo conservare.
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch == settings.epochs or epoch % max(1, settings.progress_interval) == 0:
            print(f"Epoca {epoch:02d}/{settings.epochs} | loss training: {train_loss:.4f} | "
                  f"loss validation: {validation_loss:.4f} | F1 validation: {validation_f1:.4f}", flush=True)

    # La rete restituita corrisponde all'epoca scelta, che puo' essere diversa dall'ultima.
    model.load_state_dict(best_state)
    print(f"LSTM selezionata: epoca {best_epoch}/{settings.epochs}, "
          f"F1 validation={best_key[0]:.4f}.", flush=True)
    return model, pd.DataFrame(history), best_epoch
