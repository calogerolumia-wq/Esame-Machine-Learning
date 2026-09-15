# Handwriting: confronto SVM e LSTM

Il progetto confronta una SVM sulle feature dei task e una LSTM sulle sequenze
temporali. Ogni CSV produce una predizione; il voto tra i task produce la
classificazione del soggetto.

## Avvio

Dal terminale nella cartella che contiene `main.py`:

```powershell
.venv\Scripts\python.exe main.py
```

Con l'ambiente virtuale già attivo basta `python main.py`.
Solo alla prima installazione servono `python -m venv .venv` e
`.venv\Scripts\python.exe -m pip install -r requirements.txt`.

I parametri si modificano in `config.py`. Per esempio, `epochs` indica il numero
massimo di epoche e `progress_interval` ogni quante epoche stampare una riga.
Il comando di avvio esegue direttamente caricamento, addestramento e confronto.

## Codifica delle classi

Nel terminale, nei CSV dei risultati e nel grafico sono usate le diciture
**Classe 0** e **Classe 1**. La corrispondenza è:

| Codice | Significato |
|---|---|
| 0 | Non autismo; include anche il campo diagnostico mancante o vuoto |
| 1 | Autismo |

I dati inclusi contengono 29 soggetti: 15 della classe 0 e 14 della classe 1.
Sono utilizzati 505 task; gli altri 89 CSV sono vuoti. La codifica dei campi
mancanti o vuoti viene applicata automaticamente.

## Che cosa mostra il terminale

| Voce | Significato |
|---|---|
| Dataset | Quanti soggetti e CSV utilizzabili sono stati caricati |
| Training | Dati con cui vengono aggiornati i parametri dei modelli |
| Validation | Dati separati usati per scegliere C e gamma della SVM e l'epoca LSTM |
| Test | Dati usati dopo le scelte precedenti per il confronto finale |
| Epoca | Un passaggio completo sui task di training |
| loss training | Errore medio della rete durante l'epoca, misurato con binary cross-entropy |
| loss validation | Stessa funzione di errore misurata sui task di validation |
| F1 validation | F1 dei task di validation, calcolato con soglia 0.5 |
| LSTM selezionata | L'epoca di cui vengono effettivamente usati i pesi per il test |

I 17 soggetti di training forniscono 291 task. Gli altri due gruppi contengono
6 soggetti e 107 task ciascuno: hanno la stessa numerosità, ma persone diverse.
Tutti i task di un soggetto rimangono nello stesso insieme.

Un'epoca non è un nuovo dataset: i 291 task di training vengono utilizzati più
volte per aggiornare i pesi. Con batch da 16, ogni epoca comprende 19 batch,
l'ultimo dei quali contiene 3 task. L'ordine dei task può cambiare, quello degli
istanti dentro ogni CSV rimane invariato.

La loss non è una percentuale di classificazioni errate. Un valore più basso
indica un errore minore rispetto alla funzione usata. Precision, recall e F1
sono invece calcolate dopo aver trasformato le probabilità in decisioni 0/1.
Per questo loss e F1 possono avere andamenti diversi.

F1 pari a zero non è un errore di esecuzione: indica che in quella valutazione
non vengono riconosciuti correttamente esempi della classe positiva. Non basta,
da solo, a stabilire che tutte le predizioni siano 0.

La LSTM viene selezionata in base al maggiore F1 di validation, usando la loss
per risolvere le parità. Vengono eseguite tutte le epoche configurate, ma la rete
restituita ha i pesi dell'epoca selezionata, che può essere precedente all'ultima.
Il terminale stampa solo la prima epoca, poi una ogni 10 e l'ultima; tutte le
misure restano disponibili in `lstm_history.csv`.

## Come leggere la tabella finale

| Colonna | Significato |
|---|---|
| model / Modello | SVM oppure LSTM |
| level / Livello | Valutazione sui singoli task oppure sui soggetti dopo il voting |
| count / N | Numero di elementi valutati: 107 task oppure 6 soggetti |
| accuracy | Frazione di decisioni corrette sul totale |
| precision | Fra le decisioni 1, quante hanno etichetta di riferimento 1 |
| recall | Fra gli esempi con etichetta 1, quanti sono riconosciuti come 1 |
| f1 | Media armonica di precision e recall |

Le metriche usano 1 come classe positiva. Il valore 0.80 corrisponde all'80%.
I conteggi TN, FP, FN e TP sono conservati in `metrics.csv` e nelle matrici:
TN e TP sono decisioni corrette; FP è un riferimento 0 predetto 1; FN è un
riferimento 1 predetto 0.

### Esempio: esecuzione del 15/09/2026 alle 08:41 UTC

Il run originariamente salvato in `run_20260915_084132_955243` riportava:

| Modello | Livello | Corretti | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| SVM | Task | 78/107 | 72.90% | 81.58% | 58.49% | 68.13% |
| LSTM | Task | 71/107 | 66.36% | 63.93% | 73.58% | 68.42% |
| SVM | Soggetti | 4/6 | 66.67% | 100% | 33.33% | 50% |
| LSTM | Soggetti | 5/6 | 83.33% | 100% | 66.67% | 80% |

In quell'esecuzione la LSTM selezionata era l'epoca **25**, con F1 di validation
**0.6226**. All'epoca 40 F1 era 0.5882: sono quindi stati ripristinati i pesi
precedentemente conservati all'epoca 25.

La SVM aveva più task corretti, mentre la LSTM riconosceva più soggetti dopo il
voting. È possibile perché un soggetto può essere classificato correttamente
anche se alcuni suoi task sono errati: conta la maggioranza dei voti. Per
esempio, tre decisioni [1, 1, 0] danno una decisione finale 1.

Il calo della loss di training insieme a una loss di validation più alta nelle
ultime epoche suggerisce overfitting: la rete migliora sui dati usati per
addestrarla senza un miglioramento analogo sui dati di validation. La scelta
si basa comunque sul criterio F1 definito in anticipo.

I valori di questo esempio appartengono a quel run. Ogni nuova esecuzione
riporta i propri risultati in `summary.txt`; versioni delle librerie e piattaforme
diverse possono produrre differenze numeriche. Con soli 6 soggetti di test,
una decisione cambia l'accuracy di circa 16.7 punti percentuali.

## Come viene creata la cartella results

`main.py` ricava la propria posizione e imposta `output_dir = root / "results"`.
Al termine del confronto chiama `save_results`, in `evaluation.py`, che crea la
cartella con `mkdir(parents=True, exist_ok=True)` e salva le tabelle con `to_csv`
e il riepilogo con `write_text`. `plots.py` salva il PNG con `savefig`.

La cartella ora contiene **sette file**, direttamente al suo interno:

| File | Contenuto e uso |
|---|---|
| `summary.txt` | Punto di partenza: dati usati, modelli selezionati, metriche, parametri, ambiente e legenda dei file |
| `metrics.csv` | Quattro righe di confronto: due modelli per due livelli, con metriche e conteggi della matrice |
| `task_predictions.csv` | Una riga per task del test: `subject_id`, `task_id`, `label`, `svm_prediction`, `lstm_prediction` |
| `subject_predictions.csv` | Una riga per soggetto del test: label, numero di task, voti per le due classi, pareggi e decisioni SVM/LSTM |
| `subject_split.csv` | Una riga per soggetto: ID di lavoro, ID originale, label, insieme di appartenenza e numero di task |
| `lstm_history.csv` | Tutte le epoche con `train_loss`, `validation_loss`, `validation_f1` |
| `confusion_matrices.png` | Un'immagine con quattro matrici: SVM e LSTM, per task e soggetto; righe = riferimento, colonne = predizione |

Nei file di predizione, `label` è il riferimento associato all'anagrafica e
`prediction` è la decisione del modello. Per i soggetti, `svm_class_1_votes`
conta quanti task la SVM ha assegnato a 1; il prefisso `lstm_` indica gli stessi
conteggi per la rete. `tie=True` segnala la parità, risolta con la classe 0.

**Ogni esecuzione completata aggiorna gli stessi sette file.** Per conservare
un risultato precedente, copiare o rinominare la cartella prima di un nuovo avvio.
Le vecchie cartelle `run_...` già presenti sul computer non vengono cancellate
automaticamente dal programma.

La precedente versione creava una sottocartella per ogni esecuzione usando
la data e l'ora UTC nel nome. Per esempio, `20260915_084132_955243` significava
15 settembre 2026, ore 08:41:32, con la parte finale riservata ai microsecondi.
Quei nomi servivano a conservare separati i vari avvii.

### A cosa servivano gli altri file della versione precedente

| File precedenti | Funzione e gestione attuale |
|---|---|
| `RESULTS.md` | Riepilogo; sostituito da `summary.txt`, leggibile anche con Blocco note |
| `svm_task_predictions.csv`, `lstm_task_predictions.csv` | Predizioni separate; ora affiancate in `task_predictions.csv` |
| `svm_subject_predictions.csv`, `lstm_subject_predictions.csv` | Voti separati; ora riuniti in `subject_predictions.csv` |
| `subject_inventory.csv`, `subject_split.csv` | Anagrafica operativa e divisione; ora l'essenziale è in `subject_split.csv` |
| `task_audit.csv`, `dataset_summary.json` | Controlli e conteggi dei CSV; il riepilogo conserva i conteggi complessivi |
| `task_features.csv`, `feature_names.json` | Feature estratte e loro nomi; ora vengono utilizzate in memoria senza salvarle |
| `svm_validation.csv` | Configurazioni SVM provate; ora elencate in `summary.txt` |
| `run_config.json`, `environment.json` | Parametri e versioni delle librerie; ora riportati in `summary.txt` |
| `svm_model.joblib`, `sequence_scaler.joblib`, `lstm_model.pt` | Modelli e scaler per un futuro riutilizzo; ora rimangono in memoria durante il confronto |
| `model_comparison.png`, `lstm_training.png` | Grafici aggiuntivi; metriche e storico rimangono nei rispettivi CSV |
| `execution.log` | Copia dell'output del terminale nell'esempio distribuito; non era creato dal normale `main.py` |

## Che cosa fa ciascun file Python

| File | Responsabilità |
|---|---|
| `main.py` | Coordina il flusso completo: dati, divisione, due rappresentazioni, training, test e salvataggio |
| `config.py` | Contiene `Settings`: seme, quote dei dati, lunghezza delle sequenze, dimensione LSTM, epoche, parametri SVM e frequenza di stampa |
| `data_loader.py` | Legge anagrafiche e CSV; applica le label; mantiene separato ogni task ed esclude i CSV vuoti |
| `preprocessing.py` | Ordina e pulisce i campioni, individua gli stroke, ricampiona le sequenze, standardizza i canali e divide i soggetti |
| `feature_engineering.py` | Produce 22 feature per ogni task: statistiche, durata, velocità, accelerazione e caratteristiche degli stroke |
| `models.py` | Definisce la rete LSTM, addestra i due modelli, sceglie la configurazione SVM e i pesi dell'epoca LSTM migliore |
| `evaluation.py` | Combina i voti dei task, calcola le metriche, affianca i risultati dei due modelli e scrive i sei file di testo/CSV |
| `plots.py` | Disegna e salva le quattro matrici di confusione in un unico PNG |
| `test_project.py` | Controlli facoltativi del codice; non fa parte dei passaggi necessari per avviare il confronto |

`requirements.txt` elenca i pacchetti necessari. `GUIDA_PROGETTO.md` spiega
preprocessing, feature e modelli con le formule. Per leggere il codice conviene
partire dai passaggi numerati di `main.py`, aprendo poi la funzione chiamata
nel relativo file. I commenti descrivono il ruolo dei dati, le forme degli array,
le operazioni sui pesi e le ragioni delle scelte principali.
