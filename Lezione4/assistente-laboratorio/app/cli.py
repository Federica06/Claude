"""Interfaccia da terminale: ciclo di conversazione con cronologia di sessione."""

import logging
import sys

import openai
from langchain_core.messages import HumanMessage

from app.config import ConfigError, carica_config
from app.graph import costruisci_grafo
from app.llm import crea_llm

logger = logging.getLogger(__name__)

ERRORE_CONNESSIONE = (
    "Errore: impossibile contattare il modello. "
    "Controlla che LM Studio sia avviato e che LM_BASE_URL sia corretto."
)
SALUTO = "Arrivederci!"


def configura_log() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # le righe "HTTP Request: POST ..." coprirebbero la conversazione;
    # con le versioni installate arrivano dal logger httpx2
    for nome in ("httpx", "httpx2"):
        logging.getLogger(nome).setLevel(logging.WARNING)


def ciclo(grafo, max_steps: int, leggi=input, scrivi=print) -> None:
    cronologia = []  # solo in memoria: si perde alla chiusura
    config = {"recursion_limit": 2 * max_steps + 5}
    while True:
        try:
            testo = leggi("Tu> ").strip()
            if not testo:
                continue
            if testo.lower() == "esci":
                break
            risultato = grafo.invoke(
                {"messages": cronologia + [HumanMessage(testo)], "steps": 0}, config=config
            )
        except (EOFError, KeyboardInterrupt):
            break
        except openai.APIError as errore:
            # la domanda fallita viene scartata: la cronologia resta com'era
            logger.error("errore nella chiamata al modello: %s", errore)
            scrivi(ERRORE_CONNESSIONE)
            continue
        cronologia = risultato["messages"]
        scrivi(f"Assistente> {cronologia[-1].content}")
    scrivi(SALUTO)


def main() -> int:
    try:
        config = carica_config()
    except ConfigError as errore:
        print(errore, file=sys.stderr)
        return 2
    configura_log()
    grafo = costruisci_grafo(crea_llm(config), config.max_steps)
    print(f"Assistente del laboratorio: modello {config.model}, MAX_STEPS={config.max_steps}.")
    print('Scrivi "esci" per terminare.')
    ciclo(grafo, config.max_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
