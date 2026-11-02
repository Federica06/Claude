# Handoff — Assistente del laboratorio

Aggiornato il 2026-11-02, a fine Lezione 4.

## Stato e ultimo commit

- Branch `main`. Il push lo esegue l'utente a fine lavoro.
- Ultimo commit di codice: `8d49d0d feat(docker): avvio con Docker Compose verso LM Studio sull'host`. Dopo di esso ci sono solo commit di documentazione.
- Il commit finale per la consegna non può essere scritto in questo file: si ricava con `git rev-parse --short HEAD`.
- Implementati tutti e quattro gli incrementi del piano (`docs/superpowers/plans/2026-11-02-assistente-laboratorio.md`), secondo la specifica `docs/superpowers/specs/2026-11-02-assistente-laboratorio-design.md`.

## Cosa funziona

- Chat da terminale (`python -m app.cli`) con `gemma-4-e2b-it` servito da LM Studio sull'host.
- Grafo LangGraph esplicito: nodi `modello`, `strumenti` e `limite`, arco condizionale `instrada`, nessun helper prebuilt.
- Cronologia in memoria per la sessione, senza duplicazioni tra i turni. `steps` riparte da 0 a ogni domanda.
- Strumento `tooltime(fuso_orario)`: data e ora ISO con offset da UTC e nome del fuso. Le chiamate compaiono nei log.
- Limite di `MAX_STEPS` esecuzioni dello strumento per domanda, mai superato. Le tool call in sospeso vengono chiuse con un errore e la risposta finale è un testo fisso.
- Risposta vuota del modello sostituita da un avviso. Errori di connessione gestiti senza perdere la sessione. Configurazione non valida con codice di uscita 2.
- Avvio e test anche con Docker Compose. Dal container LM Studio si raggiunge tramite `host.docker.internal`.

## Decisioni da ricordare

- Variabili solo in `.env`: `LM_BASE_URL`, `LM_MODEL`, `MAX_STEPS`. L'ambiente ha la precedenza, e così funziona l'override di `LM_BASE_URL` in `compose.yaml`. Il parser è scritto da noi: legge UTF-8 con BOM e toglie le virgolette esterne.
- Il fuso di `tooltime` è obbligatorio, senza valore predefinito: lo sceglie il modello.
- Instradamento: `steps + len(tool_calls) > max_steps` → `limite`. Si conta ogni singola esecuzione, anche quelle che finiscono in errore.
- Parametri fissi del client: `temperature=0.2`, `timeout=300`, `max_tokens=4096` (il ragionamento del modello consuma token).
- Le richieste HTTP vengono registrate dal logger `httpx2` (oltre a `httpx`). Entrambi sono portati a `WARNING`.
- Nei log l'esecuzione dello strumento usa `->` e non `→`: su Windows lo stderr rediretto usa cp1252.
- Con un `LM_MODEL` inesistente LM Studio non dà errore e risponde con il modello caricato.
- Sono fissate solo le quattro dipendenze dirette. Le transitive possono variare tra un'installazione e l'altra: per esempio `langsmith` era 0.14.16 sull'host e 0.14.17 nell'immagine Docker.

## Prove eseguite

| Prova | Dove | Chi | Esito |
|---|---|---|---|
| Test automatici (`pytest`) | host | Claude e utente | 40 passed, 1 skipped |
| Test automatici (`pytest`) | container | Claude e utente | 40 passed, 1 skipped |
| Chat: memoria del nome su due turni | host, interattiva | utente | superata |
| Assenza delle righe `HTTP Request: POST` | host, interattiva | utente | superata |
| Server irraggiungibile (porta sbagliata), uscita con Ctrl+C | host, interattiva | utente | superata |
| Configurazione non valida → codice 2 | host | Claude e utente | superata |
| `RUN_REAL_TESTS=1` (Tokyo → `tooltime` con `Asia/Tokyo`) | host | Claude e utente | superata |
| Chat Tokyo poi Roma con le righe `tool call` nei log | host, interattiva | utente | superata |
| `RUN_REAL_TESTS=1` | container | Claude e utente | superata (tool call `tooltime` con `Asia/Tokyo`) |
| Chat Tokyo poi Roma | container, input tramite pipe (`-T`) | Claude | superata |
| `.env` assente nell'immagine | container | Claude e utente | superata |
| Build dell'immagine (`docker compose build`) | host | Claude e utente | superata |
| Chat interattiva con TTY Tokyo poi Roma (`docker compose run --rm assistente`) | container | utente | superata |
| Limite di passaggi con il modello vero | host | — | non eseguita (coperto solo dai test automatici) |

## Problemi aperti

- Il nodo `limite` non è mai stato visto scattare con il modello reale: `gemma-4-e2b-it` chiede un solo orologio per domanda.
- La prima richiesta della sessione può richiedere fino a circa 2 minuti mentre LM Studio carica il modello.
- Nel container i timestamp dei log sono in UTC.

## Problemi noti rimandati (dalla revisione finale)

La revisione finale indipendente è stata eseguita: nessun problema critico. I problemi seguenti sono stati annotati e non corretti, per decisione dell'utente.

- **Importante** — Tool call malformata (argomenti non JSON): langchain la mette in `invalid_tool_calls`, `instrada` va a `END` e la tool call resta senza risposta in cronologia; un server rigoroso potrebbe rifiutare tutti i turni successivi (non verificato con LM Studio) (`app/graph.py`).
- **Minore** — Con lo stdout rediretto su Windows, una risposta con caratteri fuori da cp1252 (per esempio emoji) chiude la sessione con un traceback; console interattiva e Docker non sono interessati (`app/cli.py`).
- **Minore** — Ogni errore dell'API è presentato come problema di connessione, anche un 400 per contesto esaurito in una sessione lunga (`app/cli.py`).
- **Minore** — Un `.env` non UTF-8 (per esempio UTF-16, come lo scrive `"..." > .env` in PowerShell 5.1) produce un traceback invece del codice di uscita 2 (`app/config.py`).
- **Minore** — Con `fuso_orario="Europe"` (una cartella di tzdata) il messaggio d'errore dello strumento contiene un percorso assoluto dell'host (`app/tools.py`).
- **Minore** — Funziona solo `python -m pytest`, non `pytest` da solo; i documenti indicano già `python -m pytest`.
- **Minore** — Compose legge `env_file` con un proprio parser, diverso da `app/config.py`; con il `.env.example` attuale non ci sono effetti (non verificato) (`compose.yaml`).

## Prossimi passi

- Se si vuole, verificare il limite con il modello vero (`MAX_STEPS=1` e una domanda su più città).
- Valutare la correzione dei problemi rimandati, a partire dalla tool call malformata.
- Lezioni successive: altri strumenti, MCP, workflow e RAG (non anticipati in questa versione).
