# Assistente del laboratorio — Specifica di design

- Data: 2026-11-02
- Stato: in revisione (da approvare prima del piano di implementazione)
- Ambito: prima versione del progetto del corso «Deep Learning e LLM: costruire chatbot e agenti»

## 1. Obiettivo

Applicazione Python da terminale che conversa con un modello servito da LM Studio sull'host. Il flusso è un grafo LangGraph esplicito. Il modello può richiedere lo strumento orologio `tooltime` tramite tool calling. Il grafo lo esegue davvero, restituisce il risultato al modello e ottiene la risposta finale. La cronologia vive in memoria per la durata della sessione e alla chiusura si perde. L'applicazione si avvia sia sull'host sia con Docker Compose.

### Criteri di successo

1. Alla domanda «Che ore sono adesso a Tokyo?», `gemma-4-e2b-it` chiama `tooltime` con `Asia/Tokyo`. Nei log compare la riga della chiamata e la risposta finale riporta l'ora corretta. La prova va eseguita sia sull'host sia in Docker.
2. Nel terminale non compaiono le righe `HTTP Request: POST ...`.
3. Nella stessa sessione l'assistente ricorda le domande precedenti.
4. Il numero di esecuzioni dello strumento per domanda non supera mai `MAX_STEPS`.
5. `python -m pytest` passa senza un modello attivo, sia sull'host sia nel container.

### Fuori ambito

MCP, RAG, altri strumenti, interfacce web o grafiche, servizi cloud, persistenza della cronologia, streaming delle risposte, checkpointer LangGraph, framework o strumenti non concordati (uv, pyproject.toml, python-dotenv).

## 2. Ambiente e dipendenze

- Python 3.10 sull'host (versione installata: 3.10.0). Ambiente virtuale `.venv/` dentro `assistente-laboratorio/`, creato nel primo incremento del piano e non versionato.
- Gestione con `venv` + `pip`. Un solo file `requirements.txt` con versioni fissate:

| Pacchetto | Versione | Motivo |
|---|---|---|
| `langgraph` | `1.2.17` | grafo esplicito: stato, nodi, archi condizionali |
| `langchain-openai` | `1.7.2` | client `ChatOpenAI` verso l'API compatibile OpenAI di LM Studio, con `bind_tools` |
| `tzdata` | `2026.5` | database dei fusi orari per `zoneinfo`, che manca su Windows e in `python:3.10-slim` |
| `pytest` | `9.1.1` | test automatici |

Le altre librerie (`langchain-core`, `openai`, `httpx`, ...) arrivano come dipendenze transitive.

- LM Studio resta sull'host e non va containerizzato. Il modello di chat disponibile, restituito da `GET /v1/models`, è `gemma-4-e2b-it`.

## 3. Struttura del progetto

```
assistente-laboratorio/
├── CLAUDE.md
├── README.md
├── HANDOFF.md
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── compose.yaml
├── docs/superpowers/
│   ├── specs/      (questa specifica)
│   └── plans/      (piano di implementazione)
├── app/
│   ├── __init__.py
│   ├── config.py   # lettura di .env e dell'ambiente, validazione
│   ├── llm.py      # creazione del client ChatOpenAI con tooltime collegato
│   ├── tools.py    # strumento tooltime
│   ├── graph.py    # stato, nodi, instradamento, costruzione del grafo
│   └── cli.py      # ciclo da terminale, logging, entry point
└── tests/
    ├── test_config.py
    ├── test_tools.py
    ├── test_llm.py
    ├── test_graph.py
    ├── test_cli.py
    └── test_reali.py   # LM Studio reale, saltato salvo RUN_REAL_TESTS=1
```

Avvio: `python -m app.cli`, eseguito dalla radice `assistente-laboratorio/`.

Dipendenze tra moduli: `cli` → `config`, `llm`, `graph`; `llm` → `tools`; `graph` → `tools`. `config` e `tools` non dipendono da altri moduli del progetto. `graph` riceve il modello già pronto come parametro, così nei test si può sostituire con un modello finto.

## 4. Configurazione (`app/config.py`)

### Variabili

Sono tutte obbligatorie e non ce ne sono altre:

| Variabile | Esempio in `.env.example` | Regola |
|---|---|---|
| `LM_BASE_URL` | `http://127.0.0.1:1234/v1` | non vuota |
| `LM_MODEL` | `gemma-4-e2b-it` | non vuota; deve essere l'id restituito da `GET /v1/models` |
| `MAX_STEPS` | `5` | intero > 0 |

### Lettura

- Un parser scritto da noi legge `.env` dalla directory corrente. Gestisce righe `CHIAVE=valore`, ignora righe vuote e commenti che iniziano con `#` e toglie gli spazi attorno a chiave e valore.
- **Le variabili già presenti nell'ambiente hanno la precedenza sul file.** Così funziona l'override di `LM_BASE_URL` in Compose.
- Se il file `.env` non esiste non è un errore, a patto che le variabili siano nell'ambiente: è il caso del container, dove `.env` non viene copiato e le variabili arrivano da `env_file`.
- La funzione restituisce un oggetto immutabile `Config(base_url: str, model: str, max_steps: int)`.

### Errori

Se una variabile manca o non è valida viene sollevato `ConfigError` con un messaggio chiaro, per esempio:
- `Configurazione non valida: manca LM_MODEL`
- `Configurazione non valida: MAX_STEPS deve essere un intero > 0 (trovato "abc")`

La CLI stampa il messaggio su stderr ed esce con codice 2, senza traceback.

## 5. Strumento `tooltime` (`app/tools.py`)

- Firma: `tooltime(fuso_orario: str) -> str`. È uno strumento LangChain (`@tool`) con una descrizione che spiega al modello di passare un nome di fuso IANA, per esempio `Europe/Rome` o `Asia/Tokyo`.
- **Il fuso orario è obbligatorio**, senza valore predefinito né configurazione: lo sceglie il modello.
- Risultato: data e ora ISO 8601 al secondo, con offset da UTC, seguite dal nome del fuso. Esempi:
  - `2026-11-01T19:42:10+01:00 Europe/Rome`
  - `2026-11-02T03:42:10+09:00 Asia/Tokyo`
- L'istante corrente si ottiene da una funzione `adesso()` (UTC, timezone-aware) che i test possono sostituire con un istante fisso.
- Un fuso non valido o vuoto non solleva eccezioni: lo strumento restituisce un testo d'errore, per esempio `Errore: fuso orario sconosciuto "Mars/Base". Usa un nome IANA come Europe/Rome.`, che arriva al modello come risultato.

## 6. Collegamento al modello (`app/llm.py`)

- `crea_llm(config) -> Runnable`: costruisce il client e restituisce `.bind_tools([tooltime])`:
  ```python
  ChatOpenAI(
      base_url=config.base_url,
      model=config.model,
      api_key="lm-studio",
      temperature=0.2,
      timeout=300,
      max_tokens=4096,
  )
  ```
- Parametri fissi nel codice, non configurabili:
  - `temperature=0.2` per risposte stabili;
  - `timeout=300` secondi, perché un modello locale che ragiona può essere lento;
  - `max_tokens=4096`, perché il modello ragiona prima di rispondere e il ragionamento consuma token: con un limite basso la risposta visibile resterebbe vuota.
- `"lm-studio"` è un segnaposto fisso nel codice. LM Studio non richiede chiavi e non esistono credenziali reali.
- La creazione non fa chiamate di rete.

## 7. Grafo (`app/graph.py`)

Un vero `StateGraph` LangGraph, costruito esplicitamente senza helper già pronti (niente `ToolNode`, `tools_condition` o `create_react_agent`).

### Stato

```python
class Stato(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int   # esecuzioni di tooltime nella domanda corrente
```

### Topologia

```
START → modello ──instrada──┬─→ END        (nessuna tool call)
           ↑                ├─→ strumenti ─→ modello
           │                └─→ limite ────→ END
           └────────────────────┘
```

Nodi: `modello`, `strumenti`, `limite`. Arco condizionale: `instrada`. La funzione `costruisci_grafo(llm, max_steps)` restituisce il grafo compilato senza checkpointer.

### Prompt di sistema

> Sei l'assistente del laboratorio. Per conoscere data o ora attuali usa sempre lo strumento tooltime con un fuso orario IANA (per esempio Europe/Rome). Se l'utente non indica un luogo o un fuso, chiediglielo. Non inventare mai l'ora.

Il prompt si aggiunge in testa ai messaggi solo al momento della chiamata al modello. Non entra nella cronologia dello stato, quindi non si ripete a ogni turno.

### Nodo `modello`

1. Registra a livello `INFO` l'inizio della chiamata con il numero di messaggi inviati, prompt di sistema compreso. Per esempio: `chiamata al modello: 3 messaggi inviati`.
2. Chiama `llm.invoke([SystemMessage(PROMPT)] + state["messages"])`.
3. Registra a livello `INFO` la fine della chiamata con il numero di tool call richieste. Per esempio: `risposta del modello: 1 tool call richieste`.
4. **Risposta vuota:** se la risposta non ha tool call e il suo testo è vuoto o fatto solo di spazi, la sostituisce con un `AIMessage` contenente il testo fisso `Il modello ha restituito una risposta vuota: di solito i token sono finiti nel ragionamento. Riprova.` e registra un `WARNING`.
5. Restituisce `{"messages": [risposta]}`. Un messaggio sostituito arriva a `instrada` senza tool call e quindi porta a `END`.

### Arco condizionale `instrada`

Guarda l'ultimo messaggio (un `AIMessage`):
1. Senza `tool_calls` → `END`.
2. Con `state["steps"] + len(tool_calls) > max_steps` → `"limite"`.
3. Altrimenti → `"strumenti"`.

Il limite conta **ogni singola esecuzione** dello strumento: due tool call nello stesso messaggio valgono 2. La regola 2 garantisce che il limite non venga mai superato, nemmeno con più chiamate nello stesso messaggio.

### Nodo `strumenti`

Per ogni tool call dell'ultimo `AIMessage`:
- se il nome è `tooltime` e gli argomenti sono validi, esegue lo strumento;
- se il nome è sconosciuto o gli argomenti non sono validi (per esempio manca `fuso_orario`), produce un testo d'errore senza crash;
- aggiunge un `ToolMessage(content=risultato, tool_call_id=..., name=...)`;
- registra una riga `INFO`, per esempio: `tool call tooltime args={'fuso_orario': 'Asia/Tokyo'} → 2026-11-02T03:42:10+09:00 Asia/Tokyo` (gli errori come `WARNING`).

Restituisce i `ToolMessage` e `steps` aumentato del numero di tool call elaborate.

### Nodo `limite`

1. Chiude ogni tool call in sospeso dell'ultimo `AIMessage` con un `ToolMessage` d'errore: `Errore: strumento non eseguito, limite di passaggi raggiunto.` In questo modo la cronologia resta valida per il modello, perché ogni tool call ha la sua risposta.
2. Aggiunge un `AIMessage` con il testo fisso: `Ho raggiunto il limite di N passaggi con gli strumenti e mi fermo qui senza una risposta completa.`, dove N è `max_steps`.
3. Registra un `WARNING` nei log.
4. Prosegue verso `END`. La domanda termina e la sessione continua con quella successiva.

### Rete di sicurezza

La CLI passa a `invoke` il parametro `config={"recursion_limit": 2 * max_steps + 5}`, così il limite interno di LangGraph non scatta prima del nostro quando `MAX_STEPS` è grande.

## 8. Cronologia della sessione

- La CLI tiene `cronologia: list[AnyMessage]` in memoria, senza checkpointer né persistenza.
- A ogni domanda: `risultato = grafo.invoke({"messages": cronologia + [HumanMessage(testo)], "steps": 0}, config=...)`, poi `cronologia = risultato["messages"]`.
- Senza checkpointer ogni `invoke` parte da uno stato vuoto. L'input viene applicato una sola volta (nessun messaggio duplicato) e `steps` riparte da 0 a ogni domanda.
- Alla chiusura del programma la cronologia si perde.

## 9. Interfaccia da terminale (`app/cli.py`)

- Avvio:
  1. carica la configurazione;
  2. configura i log;
  3. crea il modello e il grafo;
  4. stampa una riga di benvenuto con il modello e `MAX_STEPS`.
- Ciclo: prompt `Tu> `, risposta `Assistente> <testo dell'ultimo AIMessage>`. Le risposte non sono in streaming.
- L'input vuoto viene ignorato.
- Per uscire si scrive `esci` oppure si usa EOF (Ctrl+Z e Invio su Windows, Ctrl+D su Linux) o Ctrl+C. Il programma saluta ed esce senza traceback.

### Log

- Su stderr, sempre a livello `INFO`, con il formato `%(asctime)s %(levelname)s %(name)s: %(message)s`.
- Righe `INFO` attese per ogni domanda:
  - inizio e fine di ogni chiamata al modello, con il numero di messaggi inviati e il numero di tool call richieste (§7, nodo `modello`);
  - una riga per ogni esecuzione dello strumento (§7, nodo `strumenti`).
- Righe `WARNING`: risposta vuota sostituita, errori di strumento, limite raggiunto.
- Il logger `httpx` viene portato a `WARNING`. Nella prova reale va verificato che le righe `HTTP Request: POST ...` non compaiano più. Se con le versioni installate arrivano da un logger con un altro nome, si individua quello effettivo e si silenzia anche quello.

### Struttura testabile

- `main() -> int` esegue in ordine:
  1. carica la configurazione e, se non è valida, stampa il messaggio su stderr e restituisce 2;
  2. configura i log;
  3. crea modello e grafo;
  4. avvia il ciclo.

  Il blocco `if __name__ == "__main__"` chiama `sys.exit(main())`.
- `ciclo(grafo, max_steps, leggi=input, scrivi=print)` contiene il ciclo di conversazione e la gestione degli errori. Riceve il grafo e le funzioni di input/output come parametri, così i test lo possono pilotare con un grafo finto e input scriptati.

### Errori durante una domanda

Se LM Studio non è raggiungibile o risponde con un errore (eccezioni di `openai`, per esempio `APIConnectionError` o `APIStatusError`):
- il programma registra un `ERROR` nei log;
- stampa all'utente un messaggio che suggerisce di controllare LM Studio e `LM_BASE_URL`;
- scarta la domanda fallita senza modificare la cronologia;
- prosegue con la domanda successiva.

All'avvio non si controlla se il server è raggiungibile.

## 10. Docker

### `Dockerfile`

- Immagine base: `python:3.10-slim`, la stessa versione di Python dell'host.
- `WORKDIR /srv`.
- Copia `requirements.txt` ed esegue `pip install --no-cache-dir -r requirements.txt`, quindi anche pytest: i test si possono eseguire nel container.
- Copia `app/` e `tests/`.
- `CMD ["python", "-m", "app.cli"]`.

### `compose.yaml`

Un servizio `assistente`:
- `build: .`
- `env_file: .env`
- `environment: LM_BASE_URL: http://host.docker.internal:1234/v1` (unico override)
- `extra_hosts: ["host.docker.internal:host-gateway"]`
- `stdin_open: true`, `tty: true`

### `.dockerignore`

Esclude `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.git` e `*.gguf`. Il `.env` non entra mai nell'immagine.

### Comandi

- Chat interattiva: `docker compose run --rm assistente` (`up` non va bene per l'input da tastiera).
- Test nel container: `docker compose run --rm assistente python -m pytest`.
- Prova reale nel container: `docker compose run --rm -e RUN_REAL_TESTS=1 assistente python -m pytest tests/test_reali.py -v`.

### Punto da verificare

Se LM Studio ascolta solo su 127.0.0.1, il container potrebbe non raggiungerlo tramite `host.docker.internal`. In quel caso il README documenta come abilitare l'opzione «Serve on Local Network» di LM Studio.

## 11. Test

### Automatici, senza modello attivo (`python -m pytest`)

**`test_config.py`**
- configurazione valida letta da file;
- ogni variabile mancante produce `ConfigError` con il nome della variabile;
- `MAX_STEPS` uguale a `0`, `-1` o `abc` produce `ConfigError`;
- l'ambiente ha la precedenza sul file;
- commenti e righe vuote vengono ignorati;
- un file `.env` assente con le variabili nell'ambiente è valido.

**`test_tools.py`** (istante fisso)
- `Europe/Rome` in inverno → offset `+01:00`;
- `Europe/Rome` in estate → offset `+02:00`;
- `Asia/Tokyo` → offset `+09:00`;
- formato esatto `AAAA-MM-GGThh:mm:ss±hh:mm Fuso`;
- un fuso non valido restituisce un testo d'errore senza eccezioni.

**`test_llm.py`**
- il client è costruito con `base_url`, `model` e chiave segnaposto presi dalla configurazione;
- `temperature=0.2`, `timeout=300` e `max_tokens=4096`;
- `tooltime` è collegato;
- nessuna chiamata di rete.

**`test_graph.py`** (modello finto che restituisce `AIMessage` scriptati e registra i messaggi ricevuti)
1. Risposta diretta: nessuno strumento eseguito, si va a `END`.
2. Un giro con lo strumento: `ToolMessage` con il risultato atteso, risposta finale, `steps == 1` e riga di log `tool call tooltime ...` verificata con `caplog`.
3. Strumento sconosciuto e argomenti non validi: `ToolMessage` d'errore, nessun crash.
4. **Più chiamate oltre il limite:** con `max_steps=2` il modello restituisce un messaggio con 3 tool call. Il grafo va dritto a `limite` e nessuno strumento viene eseguito (l'orologio finto non viene mai chiamato). Tutte e 3 le tool call ricevono un `ToolMessage` d'errore, l'ultimo messaggio è il testo fisso con N=2 e nei log c'è un `WARNING`.
5. **Limite raggiunto con chiamate singole:** con `max_steps=2` il modello chiede una tool call alla volta. Le prime due vengono eseguite (`steps == 2`), la terza porta a `limite` senza essere eseguita, con il `ToolMessage` d'errore e il testo fisso.
6. **Due turni senza duplicazioni:** stesso ciclo della CLI. Il turno 1 passa dallo strumento (Human, AI con tool call, Tool, AI → 4 messaggi), il turno 2 è una risposta diretta (→ 6 messaggi). Gli id dei messaggi sono unici, `steps` vale 1 dopo il turno 1 e 0 dopo il turno 2.
7. Il prompt di sistema arriva al modello come primo messaggio ma non compare nella cronologia restituita.
8. **Risposta vuota:** il modello finto restituisce un `AIMessage` con testo vuoto (o solo spazi) e senza tool call. L'ultimo messaggio della cronologia è l'avviso fisso, il grafo termina e nei log c'è un `WARNING`.
9. **Log del nodo modello:** con `caplog` si verificano le righe `INFO` di inizio e fine chiamata, con il numero corretto di messaggi inviati e di tool call richieste.

**`test_cli.py`** (nessun modello attivo)
1. **Configurazione non valida:** si esegue `main()` in una directory temporanea con `MAX_STEPS=abc`, usando `monkeypatch` su ambiente e directory corrente. Il valore restituito è 2, il messaggio `Configurazione non valida: ...` è su stderr (verificato con `capsys`) e non su stdout.
2. **Errore di connessione durante una domanda:** si esegue `ciclo` con un grafo finto e input scriptati (`domanda 1`, `domanda 2`, `esci`). Il grafo solleva `openai.APIConnectionError` alla prima domanda e risponde normalmente alla seconda. Si verifica che:
   - compare il messaggio per l'utente su LM Studio e `LM_BASE_URL`;
   - la cronologia passata al grafo nella seconda domanda non contiene la prima domanda fallita;
   - la seconda domanda riceve risposta, quindi il ciclo è continuato.

### Reali, attivati esplicitamente (`tests/test_reali.py`)

- Il test viene saltato se `RUN_REAL_TESTS` non vale `1`.
- Usa la configurazione reale e chiede «Che ore sono adesso a Tokyo?» al modello `gemma-4-e2b-it`.
- Verifica che nel risultato ci sia un `ToolMessage` di `tooltime` il cui contenuto contiene `Asia/Tokyo`, e che l'ultimo messaggio sia un `AIMessage` con testo non vuoto.
- Se il modello non chiama lo strumento, il test fallisce e il fallimento va riportato com'è, senza aggirarlo. Una risposta simulata non dimostra che il modello locale sappia usare lo strumento.
- Comando PowerShell sull'host:
  ```powershell
  $env:RUN_REAL_TESTS = "1"; python -m pytest tests/test_reali.py -v; Remove-Item Env:RUN_REAL_TESTS
  ```
- Nel container: vedi §10.

### Prove manuali

Documentate nel README, sull'host e in Docker:
1. avviare la chat e chiedere «Che ore sono adesso a Tokyo?»;
2. controllare la riga `tool call tooltime ...` nei log;
3. controllare che non compaiano righe `HTTP Request: POST`;
4. fare una seconda domanda che si riferisce alla prima per verificare la cronologia.

## 12. File di supporto e documentazione

- **`.gitignore`**: `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.gguf` (pesi dei modelli).
- **`.env.example`**: esattamente `LM_BASE_URL=http://127.0.0.1:1234/v1`, `LM_MODEL=gemma-4-e2b-it`, `MAX_STEPS=5`, senza credenziali né altre variabili.
- **`README.md`**, che riporta solo comandi verificati, anche per PowerShell:
  - prerequisiti;
  - configurazione (copia di `.env.example`, ricerca dell'id del modello con `GET /v1/models`);
  - creazione di `.venv` e installazione;
  - avvio sull'host e in Docker;
  - test automatici sull'host e nel container;
  - prove reali, con il comando PowerShell per `RUN_REAL_TESTS=1` e la prova nel container;
  - diagnosi: LM Studio non raggiungibile, id del modello errato, container che non raggiunge l'host.
- **`HANDOFF.md`** nella radice di `assistente-laboratorio/`, da compilare a fine lavoro con:
  - stato e ultimo commit;
  - cosa funziona;
  - decisioni da ricordare;
  - prove eseguite con esito: automatiche, LM Studio reale, Docker reale, indicando «non eseguita» dove una prova non è stata fatta;
  - problemi aperti;
  - prossimi passi.
- **`CLAUDE.md`**: aggiornato con la struttura e i comandi di test quando diventano stabili.

## 13. Metodo di lavoro

- Incrementi piccoli, ciascuno verificato prima del commit.
- Prima di ogni commit: `git status`, controllo del diff, staging selettivo dei soli file pertinenti.
- Commit in Conventional Commits in italiano.
- Nessun push senza richiesta esplicita.
- Si lavora nel branch corrente (`main`), senza branch o worktree nuovi.
