# ✍️ Handwriting Classification — SVM e LSTM

Classificazione di serie temporali di scrittura mediante due approcci: una **Support Vector Machine (SVM)** su feature estratte dai task e una rete **Long Short-Term Memory (LSTM)** sulle sequenze temporali.

Entrambi i modelli producono una predizione per registrazione. Le predizioni vengono poi aggregate con un **voto di maggioranza** per ottenere la classificazione del soggetto.

| | |
|---|---|
| **Progetto** | Classificazione di serie temporali di handwriting: confronto tra SVM e LSTM |
| **Autore** | Calogero Lumia |
| **Corso** | Machine Learning |
| **Tecnologie** | Python, scikit-learn, PyTorch |

---

## 📌 Indice

- [Introduzione](#introduzione)
- [Dataset e preprocessing](#dataset-e-preprocessing)
- [Requisiti](#requisiti)
- [Struttura del codice](#struttura-del-codice)
- [Utilizzo](#utilizzo)
- [Confronto tra i modelli](#confronto-tra-i-modelli)
- [Valutazione e risultati](#valutazione-e-risultati)
- [File generati](#file-generati)

---

<a id="introduzione"></a>
## 🧠 Introduzione

L'obiettivo è confrontare due rappresentazioni degli stessi dati di handwriting nella classificazione binaria **Autismo / Non autismo**, secondo la codifica delle etichette descritta nella sezione seguente.

La SVM utilizza caratteristiche numeriche che riassumono posizione, pressione e movimento della penna. La LSTM elabora invece la successione temporale dei campioni e apprende una rappresentazione della registrazione durante l'addestramento.

L'unità di classificazione iniziale è il **task**, corrispondente a un intero file CSV. Le righe del CSV sono campioni della stessa sequenza, non esempi indipendenti. Ogni task eredita l'etichetta del soggetto; i task non vengono suddivisi in finestre né concatenati tra loro.

---

<a id="dataset-e-preprocessing"></a>
## 📂 Dataset e preprocessing

### Organizzazione dei dati

La cartella `data/` contiene, per ciascun soggetto, un file `Anagrafica*.txt` e una sottocartella `Csv/` con le registrazioni. Il caricamento è gestito da [data_loader.py](data_loader.py).

Il dataset incluso comprende **29 soggetti** e **594 file CSV**: **505 task utilizzabili** e **89 CSV vuoti**, esclusi dall'elaborazione. Nell'esecuzione salvata non risultano ulteriori task scartati come non validi.

| Classe | Codifica utilizzata | Soggetti |
|---|---|---:|
| **0 — Non autismo** | Indicazione esplicita di assenza oppure campo diagnostico mancante o vuoto | 15 |
| **1 — Autismo** | Indicazione esplicita di autismo | 14 |

> **Assunzione sulle etichette:** dei 15 soggetti della classe 0, uno riporta `nessuno` e 14 hanno il campo diagnostico vuoto. L'assegnazione dei campi vuoti alla classe 0 è una convenzione del progetto, non una conferma dell'assenza di autismo.

L'ID serve ad associare i task alla persona e il campo diagnostico determina l'etichetta. **Nessuno dei due viene utilizzato come feature di ingresso.**

### Preparazione delle registrazioni

Il lettore utilizza le prime cinque colonne dei CSV: `Timestamp`, `PointX`, `PointY`, `Phase` e `Pressure`. In [preprocessing.py](preprocessing.py) i campioni vengono ordinati cronologicamente; i timestamp non interpretabili sono eliminati e i valori numerici mancanti vengono ricostruiti mediante interpolazione all'interno dello stesso task. Le pressioni negative sono trattate come valori mancanti. Un task con meno di tre campioni validi o un intero canale numerico mancante viene escluso.

Il tempo relativo è ricostruito con un **intervallo nominale di 0,005 secondi** tra campioni, configurabile tramite `sample_interval`. I timestamp originali servono all'ordinamento, non al calcolo degli intervalli temporali: durata, velocità e accelerazione dipendono quindi da questa assunzione.

Il campo `Phase` distingue `Hover`, `BeginStroke`, `MoveStroke` ed `EndStroke`. Da questi eventi si ricavano il contatto della penna (`pen_down`) e gli **stroke**, cioè i singoli tratti di scrittura. La stessa serie pulita viene poi rappresentata come vettore di feature per la SVM e come sequenza ricampionata per la LSTM.

---

<a id="requisiti"></a>
## 📦 Requisiti

Il progetto utilizza Python e le dipendenze indicate in [requirements.txt](requirements.txt):

| Libreria | Utilizzo |
|---|---|
| `numpy` | Calcolo numerico, feature e sequenze |
| `pandas` | Gestione delle tabelle e dei risultati |
| `scikit-learn` | SVM, standardizzazione, suddivisione e metriche |
| `torch` | Rete LSTM e addestramento |
| `matplotlib` | Matrici di confusione |

L'ambiente registrato nel [riepilogo incluso](results/summary.txt) è **Python 3.12.14 su Linux, con esecuzione su CPU**. Anche la configurazione predefinita del codice utilizza la CPU e non richiede una GPU.

---

<a id="struttura-del-codice"></a>
## 📁 Struttura del codice

Principali file e cartelle:

```text
Progetto_SVM_LSTM_Handwriting/
│
├── data/                       # Anagrafiche e registrazioni CSV
├── results/                    # Tabelle, riepilogo e matrici di confusione
│
├── main.py                     # Esecuzione dell'intera pipeline
├── config.py                   # Parametri dell'esperimento
├── data_loader.py              # Lettura dei dati e assegnazione delle etichette
├── preprocessing.py            # Pulizia, split per soggetto e sequenze LSTM
├── feature_engineering.py      # Estrazione delle 22 feature per task
├── models.py                   # SVM, LSTM e selezione dei modelli
├── evaluation.py               # Voting, metriche e salvataggio delle tabelle
├── plots.py                    # Generazione delle matrici di confusione
├── test_project.py             # Controlli automatici sul codice
│
├── requirements.txt            # Dipendenze Python
└── README.md                   # Descrizione e istruzioni del progetto
```

---

<a id="utilizzo"></a>
## ⚙️ Utilizzo

### Installazione e avvio — Windows PowerShell

Aprire il terminale nella cartella che contiene `main.py`. Alla prima installazione, creare l'ambiente virtuale e installare le dipendenze:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Per eseguire il progetto:

```powershell
.\.venv\Scripts\python.exe main.py
```

Con l'ambiente virtuale già attivo è sufficiente:

```bash
python main.py
```

L'avvio esegue caricamento, preprocessing, suddivisione dei soggetti, addestramento dei due modelli, valutazione sul test e salvataggio dei risultati. Non occorrono script di preparazione separati.

### Configurazione

I parametri si modificano nella classe `Settings` di [config.py](config.py), prima dell'avvio. Comprendono il seme casuale (`seed=42`), le quote di validation e test, la griglia SVM, la lunghezza delle sequenze e i parametri di addestramento della LSTM.

Per esempio, `epochs=40` indica le epoche da eseguire e `progress_interval=10` la frequenza di stampa durante il training, oltre alla prima e all'ultima epoca. Lo storico completo viene comunque salvato in `results/lstm_history.csv`.

### Controlli automatici facoltativi

```powershell
.\.venv\Scripts\python.exe -m unittest test_project.py
```

I nove test verificano lettura dei CSV, codifica delle etichette, preprocessing, feature, separazione dei soggetti, standardizzazione, voting, uscita della LSTM e coerenza delle tabelle dei risultati. Sono separati dall'esecuzione ordinaria di `main.py`.

---

<a id="confronto-tra-i-modelli"></a>
## 🔄 Confronto tra i modelli

### Approccio 1 — SVM su feature estratte

[feature_engineering.py](feature_engineering.py) rappresenta ogni task con **22 feature**, calcolate sulla serie pulita prima del ricampionamento:

| Quantità | Caratteristiche | Numero |
|---|---|---:|
| X, Y e pressione | Media, deviazione standard, minimo e massimo per canale | 12 |
| Tempo | Durata nominale del task | 1 |
| Velocità | Media e massimo del modulo | 2 |
| Accelerazione | Media e massimo del modulo | 2 |
| Stroke | Numero dei tratti | 1 |
| Durata degli stroke | Media, deviazione standard e massimo | 3 |
| Contatto | Frazione di campioni con penna a contatto | 1 |
| **Totale** | | **22** |

Le statistiche di posizione e pressione considerano l'intero task, inclusi i campioni `Hover`. Velocità e accelerazione utilizzano invece solo campioni consecutivi dello stesso stroke, escludendo i salti tra tratti diversi. Coordinate e pressione restano nelle unità del dispositivo.

La pipeline in [models.py](models.py) applica `StandardScaler` e una **SVM con kernel RBF**, che permette una frontiera di decisione non lineare. La standardizzazione viene stimata soltanto sul training e riutilizzata su validation e test.

La selezione confronta **nove combinazioni** di iperparametri:

```text
C     = [0.1, 1.0, 10.0]
gamma = ["scale", 0.01, 0.1]
```

`C` controlla la penalizzazione degli errori; `gamma` regola la similarità del kernel. Viene mantenuta la configurazione con il maggiore **F1 dei task di validation**; in caso di parità, la prima incontrata nella griglia. Il confronto avviene sul validation set fisso, non tramite cross-validation.

### Approccio 2 — LSTM sulle sequenze temporali

Ogni registrazione viene ricampionata in **256 istanti** distribuiti dall'inizio alla fine del task. Non vengono mantenuti soltanto i primi 256 campioni: viene rappresentata l'intera estensione temporale del CSV.

I cinque canali sono `time_seconds`, `x`, `y`, `pressure` e `pen_down`. X, Y e pressione sono interpolati linearmente; il contatto utilizza il campione più vicino e rimane binario prima della standardizzazione. Lo scaler dei canali viene stimato esclusivamente sui task di training.

L'architettura definita in [models.py](models.py) è:

```text
Sequenza (256, 5) -> LSTM (32 unità) -> Strato lineare (32 -> 1) -> Sigmoide
```

La rete utilizza un unico strato LSTM unidirezionale. L'ultimo stato nascosto produce una probabilità per task: valori **maggiori o uguali a 0,5** vengono assegnati alla classe 1. Lo stato ricorrente riparte da zero per ogni sequenza e non viene trasferito tra CSV.

| Parametro di training | Valore predefinito |
|---|---|
| Funzione di perdita | Binary cross-entropy (`BCELoss`) |
| Ottimizzatore | Adam |
| Learning rate | `0.001` |
| Batch size | `16` task |
| Epoche | `40` |
| Limite alla norma del gradiente | `1.0` |

Vengono eseguite tutte le epoche configurate, conservando i pesi dell'epoca con il maggiore **F1 dei task di validation**. A parità di F1 viene preferita la loss di validation più bassa. Prima del test vengono ripristinati questi pesi: non è previsto un arresto anticipato del training.

### Riepilogo

| Caratteristica | SVM | LSTM |
|---|---|---|
| Ingresso per task | Vettore di 22 feature | Sequenza di 256 istanti e 5 canali |
| Rappresentazione | Statistiche e caratteristiche del movimento definite esplicitamente | Rappresentazione appresa dalla successione temporale |
| Modello | SVM con kernel RBF | LSTM, strato lineare e sigmoide |
| Selezione su validation | Combinazione di `C` e `gamma` | Pesi dell'epoca migliore |
| Libreria | scikit-learn | PyTorch |

---

<a id="valutazione-e-risultati"></a>
## 📊 Valutazione e risultati

### Suddivisione per soggetto

La divisione è stratificata rispetto alla classe ed effettuata **sui soggetti, non sui singoli task**. Tutte le registrazioni di una persona rimangono nello stesso insieme, evitando che lo stesso soggetto compaia sia nel training sia nel test.

Con `seed=42` e quote del 20% per validation e test, la suddivisione inclusa è:

| Insieme | Soggetti | Classe 0 | Classe 1 | Task |
|---|---:|---:|---:|---:|
| Training | 17 | 9 | 8 | 291 |
| Validation | 6 | 3 | 3 | 107 |
| Test | 6 | 3 | 3 | 107 |
| **Totale** | **29** | **15** | **14** | **505** |

I due modelli utilizzano gli stessi task e gli stessi gruppi di soggetti. Il training serve all'addestramento, la validation alla selezione dei modelli e il test al confronto finale. Non viene effettuato un successivo riaddestramento su training e validation uniti: il protocollo è un **singolo hold-out**.

### Dal task al soggetto

Le predizioni dei task di ciascuna persona vengono aggregate con **majority voting**: vince la classe con più voti. Ogni task ha lo stesso peso; non si effettua una media delle probabilità. In caso di parità viene assegnata la **classe 0**, secondo `tie_class=0`, senza utilizzare l'etichetta di riferimento.

Per esempio, le decisioni `[1, 1, 0]` producono una classificazione del soggetto pari a `1`. La regola viene applicata separatamente alle predizioni SVM e LSTM in [evaluation.py](evaluation.py).

### Metriche

La valutazione viene effettuata sia sui task sia sui soggetti. **La classe positiva è sempre la classe 1.**

| Metrica | Significato |
|---|---|
| **Accuracy** | Frazione delle classificazioni corrette sul totale |
| **Precision** | Tra gli esempi predetti come classe 1, frazione con etichetta 1 |
| **Recall** | Tra gli esempi con etichetta 1, frazione riconosciuta come classe 1 |
| **F1** | Media armonica di precision e recall |

Le matrici di confusione riportano le classi di riferimento sulle righe e quelle predette sulle colonne, nell'ordine `[0, 1]`. I conteggi TN, FP, FN e TP sono conservati anche in `metrics.csv`. Le metriche con denominatore nullo vengono riportate come zero.

### Risultati dell'esecuzione

La SVM selezionata utilizza **`C=10.0` e `gamma="scale"`**, con F1 di validation pari a **0,7723**. Per la LSTM è stata selezionata l'**epoca 40**, con F1 di validation **0,6286** e loss di validation **0,7133**.

| Modello | Livello | Corretti / totale | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| SVM | Task | 78 / 107 | 72,90% | 81,58% | 58,49% | 68,13% |
| LSTM | Task | 74 / 107 | 69,16% | 66,67% | 75,47% | 70,80% |
| SVM | Soggetti | 4 / 6 | 66,67% | 100,00% | 33,33% | 50,00% |
| LSTM | Soggetti | 6 / 6 | 100,00% | 100,00% | 100,00% | 100,00% |

In questa esecuzione la SVM ottiene accuracy e precision maggiori sui task, mentre la LSTM raggiunge recall e F1 maggiori. Dopo il voting, la LSTM classifica correttamente tutti e sei i soggetti del test, mentre la SVM ne classifica correttamente quattro. L'aggregazione spiega perché una maggiore accuracy sui task non comporti necessariamente una maggiore accuracy sui soggetti.

![Matrici di confusione di SVM e LSTM sui task e sui soggetti del test](results/confusion_matrices.png)

> La tabella nel README è un riferimento statico all'esecuzione indicata. Una nuova esecuzione aggiorna i file in `results/`, incluso il grafico, ma non questa tabella. Per i nuovi valori consultare `summary.txt` e `metrics.csv`. Versioni delle librerie o piattaforme diverse possono produrre differenze numeriche anche con lo stesso seme.

### Limiti del confronto

Il test contiene soltanto **sei soggetti**: una diversa classificazione modifica l'accuracy per soggetto di circa **16,7 punti percentuali**. Inoltre, i task della stessa persona non sono osservazioni indipendenti e chi ha più registrazioni pesa maggiormente nelle metriche per task. Il risultato del 100% della LSTM riguarda quindi questo specifico hold-out, non dimostra una superiorità generale del modello.

Restano rilevanti l'assegnazione dei 14 campi diagnostici vuoti alla classe 0, l'uso del tempo nominale e la possibile perdita di dettagli nel ricampionamento a 256 istanti. I risultati descrivono un esperimento sul dataset disponibile e **non costituiscono una validazione diagnostica**.

---

<a id="file-generati"></a>
## 💾 File generati

Al termine dell'esecuzione vengono creati o aggiornati **sette file** direttamente nella cartella `results/`:

| File | Contenuto |
|---|---|
| [summary.txt](results/summary.txt) | Riepilogo del dataset, suddivisione, modelli selezionati, metriche, parametri e ambiente |
| [metrics.csv](results/metrics.csv) | Confronto dei due modelli per task e per soggetto, con metriche e conteggi delle matrici |
| [task_predictions.csv](results/task_predictions.csv) | Etichetta e predizioni SVM/LSTM per ogni task del test |
| [subject_predictions.csv](results/subject_predictions.csv) | Voti, eventuali pareggi e classificazioni finali dei soggetti del test |
| [subject_split.csv](results/subject_split.csv) | Assegnazione dei soggetti a training, validation e test, con ID e numero di task |
| [lstm_history.csv](results/lstm_history.csv) | Loss di training, loss di validation e F1 di validation per ogni epoca |
| [confusion_matrices.png](results/confusion_matrices.png) | Le quattro matrici di confusione in un'unica immagine |

**Ogni esecuzione completata sovrascrive gli stessi sette file.** Per conservare un esperimento precedente, copiare o rinominare `results/` prima del nuovo avvio. Il programma non crea sottocartelle `run_*` e non salva su disco i modelli addestrati o gli scaler, che rimangono in memoria durante il confronto.
