# Assistente del laboratorio

Applicazione Python da terminale che conversa con un modello servito da LM Studio sull'host.
Il flusso è un grafo LangGraph esplicito. La cronologia resta in memoria per la durata della sessione e si perde alla chiusura.

I comandi sono per PowerShell (Windows) e si eseguono dalla cartella `assistente-laboratorio/`.

## Prerequisiti

- Python 3.10
- LM Studio con il server avviato (porta predefinita 1234) e un modello di chat caricato

## Configurazione

Copiare il file di esempio:

```powershell
Copy-Item .env.example .env
```

Il file `.env` resta fuori da Git. Contiene tre variabili, tutte obbligatorie:

| Variabile | Esempio | Significato |
|---|---|---|
| `LM_BASE_URL` | `http://127.0.0.1:1234/v1` | indirizzo dell'API compatibile OpenAI di LM Studio |
| `LM_MODEL` | `gemma-4-e2b-it` | id del modello, come restituito dal server |
| `MAX_STEPS` | `5` | numero massimo di esecuzioni dello strumento per domanda (intero > 0) |

L'id del modello va copiato da quelli che il server restituisce, senza inventarlo:

```powershell
(Invoke-RestMethod http://127.0.0.1:1234/v1/models).data.id
```

Le variabili d'ambiente già impostate hanno la precedenza sul file `.env`.

## Installazione

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

I comandi usano direttamente `.\.venv\Scripts\python.exe`, quindi non serve attivare l'ambiente (e non c'è il problema dell'execution policy di PowerShell).

## Avvio

```powershell
.\.venv\Scripts\python.exe -m app.cli
```

Si scrive dopo `Tu> `. Per uscire si scrive `esci`, oppure si usa Ctrl+Z seguito da Invio, oppure Ctrl+C.
I log (livello INFO) vanno su stderr e riportano l'inizio e la fine di ogni chiamata al modello.

## Test automatici

Non serve un modello attivo:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Diagnosi

- **`Configurazione non valida: ...`** (codice di uscita 2): manca una variabile nel `.env` oppure `MAX_STEPS` non è un intero maggiore di zero. Il messaggio indica quale.
- **`Errore: impossibile contattare il modello...`**: LM Studio non è avviato o `LM_BASE_URL` è sbagliato. Prima dell'errore il client riprova due volte (righe `Retrying request` nei log). Per controllare che il server risponda: `(Invoke-RestMethod http://127.0.0.1:1234/v1/models).data.id`.
- **Id del modello sbagliato**: nelle prove, con un `LM_MODEL` inesistente LM Studio non ha restituito errori e ha risposto con il modello già caricato. Un id sbagliato quindi può passare inosservato. Va controllato che `LM_MODEL` coincida con uno degli id restituiti dal comando qui sopra.
- **La prima risposta è lenta**: LM Studio può impiegare tempo a caricare il modello alla prima richiesta (nelle prove fino a circa due minuti). Il timeout del client è di 300 secondi.
