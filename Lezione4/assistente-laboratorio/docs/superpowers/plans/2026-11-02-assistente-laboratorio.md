# Assistente del laboratorio — Piano di implementazione

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** chat da terminale in Python con un grafo LangGraph esplicito. Usa `gemma-4-e2b-it` su LM Studio (host) e lo strumento `tooltime`, con un limite di passaggi. La cronologia vive in memoria per la sessione e l'applicazione si avvia anche con Docker Compose.

**Architecture:** `StateGraph` costruito a mano con i nodi `modello`, `strumenti` e `limite` e l'arco condizionale `instrada`. Il modello viene iniettato in `costruisci_grafo`, così i test usano un modello finto. La CLI tiene la cronologia e a ogni domanda richiama il grafo con `steps=0`.

**Tech Stack:** Python 3.10, `langgraph==1.2.17`, `langchain-openai==1.7.2`, `tzdata==2026.5`, `pytest==9.1.1`, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-02-assistente-laboratorio-design.md` (commit `51b3a71`)

## Global Constraints

- Tutti i percorsi sono relativi ad `assistente-laboratorio/`. Non si toccano file fuori da questa cartella.
- Python 3.10, ambiente in `.venv/`. I comandi usano `.\.venv\Scripts\python.exe` per non dipendere dall'execution policy di PowerShell.
- Un solo `requirements.txt`, con esattamente `langgraph==1.2.17`, `langchain-openai==1.7.2`, `tzdata==2026.5` e `pytest==9.1.1`. Nessuna altra dipendenza: niente `python-dotenv`, `uv` o `pyproject.toml`.
- Variabili: solo `LM_BASE_URL`, `LM_MODEL` e `MAX_STEPS`, tutte obbligatorie. `MAX_STEPS` deve essere un intero > 0.
- Chiave API: il segnaposto `"lm-studio"`. Nessuna credenziale reale e nessun `.env` versionato.
- Niente `ToolNode`, `tools_condition`, `create_react_agent`, checkpointer o streaming.
- Log su stderr a livello `INFO` con formato `%(asctime)s %(levelname)s %(name)s: %(message)s`. Il logger `httpx` (e quelli equivalenti eventualmente trovati) va portato a `WARNING`.
- Testi fissi, da copiare così come sono:
  - **limite:** `Ho raggiunto il limite di {N} passaggi con gli strumenti e mi fermo qui senza una risposta completa.`
  - **tool call non eseguita:** `Errore: strumento non eseguito, limite di passaggi raggiunto.`
  - **risposta vuota:** `Il modello ha restituito una risposta vuota: di solito i token sono finiti nel ragionamento. Riprova.`
  - **errore di connessione:** `Errore: impossibile contattare il modello. Controlla che LM Studio sia avviato e che LM_BASE_URL sia corretto.`
- Commit in Conventional Commits in italiano, chiusi dalla riga `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Prima di ogni commit: `git status --short .` e `git diff --cached --stat`, poi `git add` solo dei file elencati nel task. Nessun push.
- Nel README vanno solo comandi eseguiti davvero con successo durante il task.

## Review Focus

1. **`.env` salvato da PowerShell con BOM UTF-8** (`Out-File`/`Set-Content -Encoding utf8` su 5.1). La prima chiave non deve diventare `﻿LM_BASE_URL` e la configurazione deve funzionare lo stesso. Il file si legge con `encoding="utf-8-sig"`. Test nel Task 1.
2. **Valori con `=` o tra virgolette nel `.env`.** La riga si divide sul primo `=`. Le virgolette `"..."` o `'...'` che racchiudono l'intero valore vengono tolte. Test nel Task 1.
3. **Argomento `fuso_orario` assente, `null` o non stringa in una tool call del modello piccolo.** Deve produrre un `ToolMessage` d'errore, non un'eccezione che chiude la sessione. Test nel Task 6.
4. **Input fatto solo di spazi.** Va ignorato come quello vuoto, senza chiamare il grafo. Test nel Task 4.
5. **Ctrl+C durante l'attesa del modello**, che con `timeout=300` può essere lunga. L'uscita deve essere pulita, con saluto e senza traceback, come per Ctrl+C al prompt. Test nel Task 4.

---

## File e responsabilità

| File | Responsabilità | Task |
|---|---|---|
| `.gitignore`, `requirements.txt`, `.env.example` | igiene del repository e dipendenze | 1 |
| `app/__init__.py`, `app/config.py` | `Config`, `ConfigError`, `carica_config` | 1 |
| `app/graph.py` | stato, prompt, nodi, instradamento, `costruisci_grafo` | 2, 6 |
| `tests/conftest.py` | `ModelloFinto` condiviso dai test | 2 |
| `app/llm.py` | `crea_llm` | 3, 6 |
| `app/cli.py` | `configura_log`, `ciclo`, `main` | 4 |
| `app/tools.py` | `adesso`, `tooltime` | 5 |
| `tests/test_reali.py` | prova con LM Studio reale | 7 |
| `Dockerfile`, `compose.yaml`, `.dockerignore` | container | 8 |
| `README.md`, `HANDOFF.md`, `CLAUDE.md` | documentazione | 4, 7, 8, 9 |

---

# Incremento 1 — Grafo con risposta fissa

Obiettivo: progetto installabile, configurazione validata e un `StateGraph` reale `START → modello → END`, provato con un modello finto che dà una risposta fissa. Nessuna rete.

### Task 1: Ambiente, dipendenze e configurazione

**Files:**
- Create: `.gitignore`, `requirements.txt`, `.env.example`, `app/__init__.py` (vuoto), `app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Config` (dataclass frozen: `base_url: str`, `model: str`, `max_steps: int`);
  - `class ConfigError(Exception)`;
  - `carica_config(percorso: Path = Path(".env"), ambiente: Mapping[str, str] | None = None) -> Config`. Se `ambiente is None` usa `os.environ`.

- [ ] **Step 1: Creare i file di supporto**
  - `.gitignore` contiene le righe `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.gguf`.
  - `requirements.txt` contiene le quattro righe dei Global Constraints.
  - `.env.example` contiene esattamente:
    ```
    LM_BASE_URL=http://127.0.0.1:1234/v1
    LM_MODEL=gemma-4-e2b-it
    MAX_STEPS=5
    ```

- [ ] **Step 2: Creare `.venv` e installare**

  Run: `python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt`

  Expected: installazione completata. Se una versione fissata non esiste, ci si ferma e si avvisa l'utente, senza cambiare versione di propria iniziativa.

  Run: `git status --short .`

  Expected: `.venv/` non compare.

- [ ] **Step 3: Scrivere i test che falliscono** in `tests/test_config.py`. Ogni test scrive un `.env` in `tmp_path` e passa `ambiente={}` (o un dict) esplicito.
  - `test_config_valida_da_file`: le tre righe di `.env.example` → `Config("http://127.0.0.1:1234/v1", "gemma-4-e2b-it", 5)`.
  - `test_variabile_mancante` (parametrizzato sulle tre chiavi): manca la chiave → `ConfigError`, con il nome della chiave nel messaggio e il messaggio che inizia con `"Configurazione non valida: "`.
  - `test_max_steps_non_valido` (parametrizzato su `"0"`, `"-1"`, `"abc"`): `ConfigError`, con `MAX_STEPS` e il valore trovato nel messaggio.
  - `test_ambiente_ha_precedenza`: il file ha `LM_BASE_URL=http://127.0.0.1:1234/v1`, `ambiente={"LM_BASE_URL": "http://host.docker.internal:1234/v1"}` → vince l'ambiente.
  - `test_commenti_e_righe_vuote`: righe `# commento` e righe vuote vengono ignorate.
  - `test_file_assente_con_ambiente`: il percorso non esiste e le tre variabili sono in `ambiente` → `Config` valido.
  - `test_bom_utf8`: file scritto con `encoding="utf-8-sig"` → `Config` valido (Review Focus 1).
  - `test_valori_con_uguale_e_virgolette`: `LM_BASE_URL="http://x:1/v1?a=b"` → `"http://x:1/v1?a=b"`; `LM_MODEL='gemma-4-e2b-it'` → senza apici (Review Focus 2).

- [ ] **Step 4: Verificare che falliscano**

  Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`

  Expected: FAIL, `ModuleNotFoundError: app.config`.

- [ ] **Step 5: Implementare `app/config.py`**
  - Legge il file con `utf-8-sig` e divide ogni riga sul primo `=`, togliendo spazi e virgolette esterne.
  - Unisce i valori con `{**file, **ambiente}`, così l'ambiente vince sul file.
  - Valida nell'ordine `LM_BASE_URL`, `LM_MODEL`, `MAX_STEPS`.
  - Messaggi:
    - variabile mancante: `Configurazione non valida: manca {CHIAVE}`;
    - `MAX_STEPS` non valido: `Configurazione non valida: MAX_STEPS deve essere un intero > 0 (trovato "{valore}")`.

- [ ] **Step 6: Verificare che passino**

  Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`

  Expected: tutti PASS.

- [ ] **Step 7: Commit**

  `git add .gitignore requirements.txt .env.example app/__init__.py app/config.py tests/test_config.py`

  Messaggio: `feat(config): lettura e validazione di .env con dipendenze fissate`

### Task 2: Grafo minimo con nodo `modello`

**Files:**
- Create: `app/graph.py`, `tests/conftest.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Produces:
  - `Stato` (TypedDict: `messages: Annotated[list[AnyMessage], add_messages]`, `steps: int`);
  - `PROMPT: str` (testo esatto della spec §7);
  - `RISPOSTA_VUOTA: str`;
  - `costruisci_grafo(llm, max_steps: int)`, che restituisce il grafo compilato. `llm` è qualsiasi oggetto con `.invoke(list[BaseMessage]) -> AIMessage`.
- Produces (test): in `tests/conftest.py`, `class ModelloFinto` con:
  - `__init__(self, risposte: list[AIMessage])`;
  - `invoke(messages)`, che salva `list(messages)` in `self.chiamate` e restituisce la risposta successiva della lista.

- [ ] **Step 1: Scrivere i test che falliscono** in `tests/test_graph.py`. Tutti invocano `grafo.invoke({"messages": [HumanMessage("ciao")], "steps": 0})`.
  - `test_risposta_fissa`: con `ModelloFinto([AIMessage("Risposta fissa")])` l'ultimo messaggio è `"Risposta fissa"`, ci sono 2 messaggi e `steps == 0`.
  - `test_prompt_di_sistema_non_in_cronologia`: `modello.chiamate[0][0]` è un `SystemMessage` con contenuto `PROMPT`, e nessun `SystemMessage` compare in `risultato["messages"]`.
  - `test_risposta_vuota_sostituita` (parametrizzato su `""` e `"   "`): l'ultimo messaggio è `AIMessage(RISPOSTA_VUOTA)` e `caplog` contiene un `WARNING`.
  - `test_log_nodo_modello`: `caplog` a livello `INFO` contiene `chiamata al modello: 2 messaggi inviati` e `risposta del modello: 0 tool call richieste`.

- [ ] **Step 2: Verificare che falliscano**

  Run: `.\.venv\Scripts\python.exe -m pytest tests/test_graph.py -v`

  Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implementare `app/graph.py`**
  - `StateGraph(Stato)` con il nodo `"modello"` e gli archi `START → "modello"` e `"modello" → END`. Il grafo si compila senza checkpointer.
  - Il nodo `modello` segue la spec §7, passi 1–5. Usa `logger = logging.getLogger(__name__)`.
  - `max_steps` per ora non viene usato: serve al Task 6.

- [ ] **Step 4: Verificare che passino**

  Run: `.\.venv\Scripts\python.exe -m pytest -v`

  Expected: tutti PASS.

- [ ] **Step 5: Commit**

  `git add app/graph.py tests/conftest.py tests/test_graph.py`

  Messaggio: `feat(grafo): StateGraph con nodo modello e risposta vuota gestita`

**Verifica dell'incremento 1:** solo test senza modello. Nessuna prova reale.

---

# Incremento 2 — Collegamento a LM Studio e cronologia

Obiettivo: chat reale con `gemma-4-e2b-it` sull'host, cronologia in memoria, errori gestiti e log puliti. Lo strumento non c'è ancora.

### Task 3: Client del modello

**Files:**
- Create: `app/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Config` (Task 1).
- Produces: `crea_llm(config: Config)`. Per ora restituisce il `ChatOpenAI`; dal Task 6 restituirà il `RunnableBinding` con lo strumento collegato.

- [ ] **Step 1: Scrivere il test che fallisce**

  `test_parametri_client`. Con `Config("http://127.0.0.1:1234/v1", "gemma-4-e2b-it", 5)` verifica:
  - `openai_api_base == "http://127.0.0.1:1234/v1"`;
  - `model_name == "gemma-4-e2b-it"`;
  - `openai_api_key.get_secret_value() == "lm-studio"`;
  - `temperature == 0.2`, `request_timeout == 300`, `max_tokens == 4096`.

  La creazione non fa chiamate di rete.

- [ ] **Step 2: Verificare che fallisca**

  Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm.py -v`

  Expected: FAIL.

- [ ] **Step 3: Implementare `crea_llm`** con i parametri della spec §6.

- [ ] **Step 4: Verificare che passi**

  Run: `.\.venv\Scripts\python.exe -m pytest -v`

  Expected: tutti PASS.

- [ ] **Step 5: Commit**

  `git add app/llm.py tests/test_llm.py`

  Messaggio: `feat(llm): client ChatOpenAI verso LM Studio`

### Task 4: CLI con cronologia, errori e log

**Files:**
- Create: `app/cli.py`, `README.md`
- Test: `tests/test_cli.py`, `tests/test_graph.py` (aggiunta)

**Interfaces:**
- Consumes: `carica_config`, `ConfigError` (Task 1); `costruisci_grafo` (Task 2); `crea_llm` (Task 3).
- Produces:
  - `configura_log() -> None`;
  - `ciclo(grafo, max_steps: int, leggi=input, scrivi=print) -> None`;
  - `main() -> int`;
  - `ERRORE_CONNESSIONE: str`.

- [ ] **Step 1: Scrivere i test che falliscono**

  In `tests/test_cli.py`:
  - `test_config_non_valida_esce_con_2`:
    - `monkeypatch.chdir(tmp_path)`;
    - un `.env` con `MAX_STEPS=abc`;
    - `monkeypatch.delenv` delle tre variabili.

    Atteso: `main() == 2`, `"Configurazione non valida:"` presente in `capsys.readouterr().err` e assente da `.out`.
  - `test_errore_connessione_cronologia_intatta`:
    - un grafo finto con `invoke(input, config=None)` che registra `input`;
    - alla prima chiamata solleva `openai.APIConnectionError(request=httpx.Request("POST", "http://x"))`;
    - alla seconda restituisce `{"messages": input["messages"] + [AIMessage("ok")], "steps": 0}`;
    - input: `["domanda 1", "domanda 2", "esci"]`.

    Atteso:
    - `ERRORE_CONNESSIONE` compare tra gli output;
    - il secondo input ha esattamente 1 messaggio, `HumanMessage("domanda 2")`;
    - `"Assistente> ok"` compare tra gli output.
  - `test_input_vuoto_o_spazi_ignorato`: con gli input `["", "   ", "esci"]` il grafo non viene mai chiamato (Review Focus 4).
  - `test_ctrl_c_durante_risposta`: il grafo solleva `KeyboardInterrupt`. `ciclo` termina senza eccezioni e stampa un saluto (Review Focus 5).
  - `test_eof_termina`: `leggi` solleva `EOFError` → uscita pulita.

  In `tests/test_graph.py`:
  - `test_due_turni_senza_duplicazioni_risposta_diretta`. Si esegue lo stesso ciclo della CLI con due risposte dirette. Atteso: 2 messaggi e poi 4, id unici, `steps == 0` dopo ogni turno. Il Task 6 lo estende con lo strumento.

- [ ] **Step 2: Verificare che falliscano**

  Run: `.\.venv\Scripts\python.exe -m pytest -v`

  Expected: i test nuovi FAIL.

- [ ] **Step 3: Implementare `app/cli.py`** secondo la spec §8–§9.
  - `ciclo` passa `config={"recursion_limit": 2 * max_steps + 5}` a ogni `invoke`.
  - `ciclo` cattura `openai.APIError` (registra `ERROR`, stampa `ERRORE_CONNESSIONE` e non cambia la cronologia).
  - `ciclo` cattura `KeyboardInterrupt` ed `EOFError` (saluto e `return`).
  - `configura_log` usa `basicConfig` su stderr e porta `httpx` a `WARNING`.
  - `main` restituisce 0 quando l'utente esce normalmente.

- [ ] **Step 4: Verificare che passino**

  Run: `.\.venv\Scripts\python.exe -m pytest -v`

  Expected: tutti PASS.

- [ ] **Step 5 (prova reale, LM Studio attivo): chat sull'host**

  Preparazione: `Copy-Item .env.example .env`. Poi confermare l'id del modello con `(Invoke-RestMethod http://127.0.0.1:1234/v1/models).data.id`: deve comparire `gemma-4-e2b-it`.

  Run: `.\.venv\Scripts\python.exe -m app.cli`. Domande: «Mi chiamo Federica.» e poi «Come mi chiamo?».

  Expected:
  - la seconda risposta usa il nome;
  - compaiono le righe `INFO` di inizio e fine chiamata;
  - **non** compaiono righe `HTTP Request: POST`. Se compaiono, si legge il nome del logger nella riga (campo `%(name)s`), lo si aggiunge a `configura_log` e si ripete la prova.

  Poi si chiude LM Studio (o si stoppa il server) e si fa una domanda: atteso `ERRORE_CONNESSIONE`, con la sessione ancora viva. Gli esiti vanno annotati per `HANDOFF.md`.

- [ ] **Step 6: README, prima versione**

  Sezioni: prerequisiti, configurazione (`Copy-Item`, `Invoke-RestMethod` su `/v1/models`), creazione di `.venv` e installazione, avvio sull'host, test automatici, diagnosi (server non raggiungibile, id del modello errato). Vanno inseriti solo i comandi eseguiti negli Step 2–5 di questo task e nel Task 1.

- [ ] **Step 7: Commit**

  `git add app/cli.py tests/test_cli.py tests/test_graph.py README.md`

  Messaggio: `feat(cli): chat da terminale con cronologia di sessione e gestione errori`

**Verifica dell'incremento 2:**
- test senza modello: tutta la suite;
- prova reale: Task 4, Step 5 (LM Studio sull'host).

---

# Incremento 3 — Strumento orologio con instradamento

Obiettivo: `tooltime` reale, nodi `strumenti` e `limite`, arco `instrada` con il limite e prova reale con `gemma-4-e2b-it`.

### Task 5: Strumento `tooltime`

**Files:**
- Create: `app/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces:
  - `adesso() -> datetime` (UTC, aware);
  - `tooltime`, strumento LangChain (`@tool`) con argomento `fuso_orario: str` e risultato `str`;
  - si invoca con `tooltime.invoke({"fuso_orario": "Asia/Tokyo"})`.

- [ ] **Step 1: Scrivere i test che falliscono**

  Tutti fanno `monkeypatch.setattr("app.tools.adesso", lambda: <istante fisso UTC>)`.
  - `test_roma_inverno`: `2026-11-01T18:42:10Z` → `"2026-11-01T19:42:10+01:00 Europe/Rome"`.
  - `test_roma_estate`: `2026-07-01T10:00:00Z` → `"2026-07-01T12:00:00+02:00 Europe/Rome"`.
  - `test_tokyo`: `2026-11-01T18:42:10Z` → `"2026-11-02T03:42:10+09:00 Asia/Tokyo"`.
  - `test_formato`: il risultato rispetta `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2} \S+$`.
  - `test_fuso_non_valido` (parametrizzato su `"Mars/Base"` e `""`): il risultato inizia con `"Errore: fuso orario sconosciuto"` e contiene `Europe/Rome`, senza eccezioni.

- [ ] **Step 2: Verificare che falliscano**

  Run: `.\.venv\Scripts\python.exe -m pytest tests/test_tools.py -v`

  Expected: FAIL.

- [ ] **Step 3: Implementare `app/tools.py`**
  - Usa `zoneinfo.ZoneInfo` e cattura `ZoneInfoNotFoundError` e `ValueError`.
  - Formato: `isoformat(timespec="seconds")` + `" "` + nome del fuso.
  - Messaggio d'errore della spec §5.
  - La docstring (descrizione per il modello) chiede un nome IANA, per esempio `Europe/Rome` o `Asia/Tokyo`.
  - `tooltime` deve leggere `adesso` dal modulo in fase di esecuzione, così il monkeypatch funziona.

- [ ] **Step 4: Verificare che passino**

  Run: `.\.venv\Scripts\python.exe -m pytest -v`

  Expected: tutti PASS.

- [ ] **Step 5: Commit**

  `git add app/tools.py tests/test_tools.py`

  Messaggio: `feat(strumenti): strumento tooltime con fuso orario esplicito`

### Task 6: Instradamento, nodi `strumenti` e `limite`, strumento collegato al modello

**Files:**
- Modify: `app/graph.py`, `app/llm.py`
- Test: `tests/test_graph.py`, `tests/test_llm.py`

**Interfaces:**
- Consumes: `tooltime`, `adesso` (Task 5).
- Produces:
  - `costruisci_grafo(llm, max_steps)` (stessa firma) con la topologia completa della spec §7;
  - `TESTO_LIMITE` (template con `{n}`) e `ERRORE_LIMITE: str`;
  - `crea_llm` ora restituisce `ChatOpenAI(...).bind_tools([tooltime])`.

- [ ] **Step 1: Scrivere i test che falliscono**

  Helper di test: `chiamata(id, args={"fuso_orario": "Asia/Tokyo"}, nome="tooltime")` restituisce il dict della tool call. Tutti i test di questo task fanno il monkeypatch di `app.tools.adesso` con un istante fisso e contano le sue chiamate.
  - `test_un_giro_con_strumento`. Modello: `[AIMessage("", tool_calls=[chiamata("c1")]), AIMessage("Sono le 03:42")]`. Atteso:
    - la sequenza è Human, AI, Tool(`tool_call_id="c1"`, contenuto `"2026-11-02T03:42:10+09:00 Asia/Tokyo"`), AI;
    - `steps == 1`;
    - `caplog` contiene `tool call tooltime` e `Asia/Tokyo`;
    - la seconda chiamata al modello include il `ToolMessage`.
  - `test_strumento_sconosciuto` e `test_argomenti_non_validi`. Argomenti parametrizzati: `{}`, `{"fuso_orario": None}`, `{"fuso_orario": 42}` (Review Focus 3). Atteso: un `ToolMessage` che inizia con `"Errore"`, nessuna eccezione, un `WARNING` e `steps == 1`.
  - `test_tre_chiamate_oltre_limite`. `max_steps=2`, modello: `[AIMessage("", tool_calls=[chiamata("a"), chiamata("b"), chiamata("c")])]`. Atteso:
    - `adesso` chiamato 0 volte;
    - 3 `ToolMessage` con `ERRORE_LIMITE` e id `a`, `b`, `c`;
    - ultimo messaggio `TESTO_LIMITE.format(n=2)`;
    - `WARNING` in `caplog`;
    - il modello è stato chiamato 1 volta.
  - `test_limite_raggiunto_con_chiamate_singole`. `max_steps=2`, modello: tre `AIMessage` con una tool call ciascuno. Atteso:
    - `adesso` chiamato 2 volte;
    - la terza tool call riceve `ERRORE_LIMITE`;
    - ultimo messaggio il testo del limite;
    - `steps == 2`.
  - `test_due_turni_senza_duplicazioni`. Si estende il test del Task 4: il turno 1 passa dallo strumento (4 messaggi), il turno 2 è diretto (6 messaggi). Id unici, `steps` vale 1 e poi 0.

  In `tests/test_llm.py` si aggiorna `test_parametri_client`:
  - i parametri si leggono da `llm.bound`;
  - in più `llm.kwargs["tools"][0]["function"]["name"] == "tooltime"`.

- [ ] **Step 2: Verificare che falliscano**

  Run: `.\.venv\Scripts\python.exe -m pytest -v`

  Expected: i test nuovi e quello aggiornato FAIL.

- [ ] **Step 3: Implementare**

  In `graph.py`:
  - si sostituisce l'arco `modello → END` con `add_conditional_edges("modello", instrada, ["strumenti", "limite", END])` e si aggiungono `strumenti → modello` e `limite → END`;
  - `instrada` è una closure su `max_steps` con la regola della spec §7;
  - `strumenti` usa una mappa `{"tooltime": tooltime}` e cattura le eccezioni di validazione di `tooltime.invoke` (registrandole come `WARNING`);
  - `limite` segue la spec §7.

  In `llm.py` si aggiunge `bind_tools([tooltime])`.

- [ ] **Step 4: Verificare che passino**

  Run: `.\.venv\Scripts\python.exe -m pytest -v`

  Expected: tutti PASS.

- [ ] **Step 5: Commit**

  `git add app/graph.py app/llm.py tests/test_graph.py tests/test_llm.py`

  Messaggio: `feat(grafo): instradamento verso strumenti e limite di passaggi`

### Task 7: Prova reale con LM Studio

**Files:**
- Create: `tests/test_reali.py`
- Modify: `README.md`

- [ ] **Step 1: Scrivere `tests/test_reali.py`**
  - Il modulo è saltato con `pytest.mark.skipif(os.environ.get("RUN_REAL_TESTS") != "1", reason=...)`.
  - `test_tokyo_usa_tooltime`:
    - `config = carica_config()`;
    - `grafo = costruisci_grafo(crea_llm(config), config.max_steps)`;
    - invoca con «Che ore sono adesso a Tokyo?» e `steps=0`.
  - Atteso: almeno un `ToolMessage` con `name == "tooltime"` e `"Asia/Tokyo"` nel contenuto; ultimo messaggio un `AIMessage` con testo non vuoto e diverso da `RISPOSTA_VUOTA`.

- [ ] **Step 2: Verificare che senza attivazione venga saltato**

  Run: `.\.venv\Scripts\python.exe -m pytest -v`

  Expected: tutti PASS e `test_reali` SKIPPED.

- [ ] **Step 3 (prova reale, LM Studio attivo): test reale sull'host**

  Run: `$env:RUN_REAL_TESTS = "1"; .\.venv\Scripts\python.exe -m pytest tests/test_reali.py -v; Remove-Item Env:RUN_REAL_TESTS`

  Expected: PASS. Se fallisce perché il modello non chiama lo strumento, si riporta il fallimento con l'output reale, senza aggirarlo. Prima di cambiare il prompt si chiede all'utente.

- [ ] **Step 4 (prova reale, LM Studio attivo): chat manuale sull'host**

  Si chiede «Che ore sono adesso a Tokyo?», poi «E a Roma?».

  Expected:
  - righe `tool call tooltime args={'fuso_orario': 'Asia/Tokyo'} → ...`;
  - risposte coerenti con l'ora reale;
  - nessun `HTTP Request: POST`.

- [ ] **Step 5: README**

  Si aggiungono la sezione «Prove reali», con il comando PowerShell dello Step 3, e l'esempio di log atteso.

- [ ] **Step 6: Commit**

  `git add tests/test_reali.py README.md`

  Messaggio: `test(reali): prova di tooltime con il modello di LM Studio`

**Verifica dell'incremento 3:**
- test senza modello: tutta la suite;
- prove reali: Task 7, Step 3–4.

---

# Incremento 4 — Docker Compose e verifica completa

### Task 8: Container

**Files:**
- Create: `Dockerfile`, `compose.yaml`, `.dockerignore`
- Modify: `README.md`

- [ ] **Step 1: Scrivere i file** come da spec §10:
  - `python:3.10-slim`, `WORKDIR /srv`;
  - `COPY requirements.txt` + `pip install --no-cache-dir`, poi `COPY app/ app/` e `COPY tests/ tests/`;
  - `CMD ["python", "-m", "app.cli"]`;
  - in `compose.yaml` il servizio `assistente` con `env_file`, l'override di `LM_BASE_URL`, `extra_hosts`, `stdin_open` e `tty`;
  - in `.dockerignore`: `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.git`, `*.gguf`.

- [ ] **Step 2 (Docker attivo): build**

  Run: `docker compose build`

  Expected: build riuscita.

- [ ] **Step 3 (Docker attivo): test senza modello nel container**

  Run: `docker compose run --rm assistente python -m pytest -v`

  Expected: tutti PASS e `test_reali` SKIPPED.

  Run: `docker compose run --rm assistente ls -a /srv`

  Expected: `.env` assente.

- [ ] **Step 4 (Docker + LM Studio attivi): test reale nel container**

  Run: `docker compose run --rm -e RUN_REAL_TESTS=1 assistente python -m pytest tests/test_reali.py -v`

  Expected: PASS. Se fallisce con un errore di connessione, si abilita «Serve on Local Network» in LM Studio, si ripete e si documenta l'esito.

- [ ] **Step 5 (Docker + LM Studio attivi): chat manuale nel container**

  Run: `docker compose run --rm assistente`

  Stesse domande e stesse attese del Task 7, Step 4.

- [ ] **Step 6: README**

  Si aggiungono le sezioni «Avvio con Docker», «Test nel container» e «Prova reale nel container», e la voce di diagnosi «il container non raggiunge LM Studio», con i comandi verificati negli Step 2–5.

- [ ] **Step 7: Commit**

  `git add Dockerfile compose.yaml .dockerignore README.md`

  Messaggio: `feat(docker): avvio con Docker Compose verso LM Studio sull'host`

### Task 9: Verifica completa e documentazione finale

**Files:**
- Create: `HANDOFF.md`
- Modify: `CLAUDE.md`, `README.md` (solo correzioni)

- [ ] **Step 1: Suite completa sull'host e nel container**

  Run: `.\.venv\Scripts\python.exe -m pytest -v` e `docker compose run --rm assistente python -m pytest -v`

  Expected: tutti PASS in entrambi.

- [ ] **Step 2: Rilettura del README**

  Ogni comando del README deve essere stato eseguito in questo piano. Quelli non verificati si tolgono o si segnalano.

- [ ] **Step 3: Aggiornare `CLAUDE.md`**

  Si aggiungono una sezione con la struttura (`app/`, `tests/`), i comandi di test (host, container, prove reali) e l'avvio. Non si annota l'avanzamento.

- [ ] **Step 4: Scrivere `HANDOFF.md`**

  Contenuto, come da spec §12:
  - stato e ultimo commit;
  - cosa funziona;
  - decisioni da ricordare;
  - prove eseguite con esito: automatiche host, automatiche container, LM Studio reale host, Docker reale, con «non eseguita» dove manca una prova;
  - problemi aperti;
  - prossimi passi.

- [ ] **Step 5: Commit**

  `git add HANDOFF.md CLAUDE.md README.md`

  Messaggio: `docs: handoff, struttura e comandi di test`

---

## Riepilogo delle verifiche

| Incremento | Senza modello (`pytest`) | LM Studio reale | Docker |
|---|---|---|---|
| 1 Grafo con risposta fissa | config, grafo minimo | — | — |
| 2 LM Studio e cronologia | llm, cli, due turni | chat, memoria, log `httpx`, errore di connessione | — |
| 3 Orologio e instradamento | tools, strumenti, limite, due turni con strumento | `test_reali` + chat Tokyo/Roma | — |
| 4 Docker e verifica | suite nel container | `test_reali` nel container | build, chat interattiva, `.env` assente |
