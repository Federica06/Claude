"""Grafo LangGraph dell'assistente: stato, nodi e instradamento."""

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

PROMPT = (
    "Sei l'assistente del laboratorio. Per conoscere data o ora attuali usa sempre "
    "lo strumento tooltime con un fuso orario IANA (per esempio Europe/Rome). "
    "Se l'utente non indica un luogo o un fuso, chiediglielo. Non inventare mai l'ora."
)
RISPOSTA_VUOTA = (
    "Il modello ha restituito una risposta vuota: di solito i token sono finiti "
    "nel ragionamento. Riprova."
)


class Stato(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int  # esecuzioni di tooltime nella domanda corrente


def _testo_vuoto(contenuto) -> bool:
    if isinstance(contenuto, str):
        return not contenuto.strip()
    return not contenuto  # contenuto a blocchi: vuoto solo se la lista è vuota


def costruisci_grafo(llm, max_steps: int):
    """Compila il grafo senza checkpointer; llm è un oggetto con .invoke(messages)."""

    def modello(state: Stato) -> dict:
        messaggi = [SystemMessage(PROMPT)] + state["messages"]
        logger.info("chiamata al modello: %d messaggi inviati", len(messaggi))
        risposta = llm.invoke(messaggi)
        logger.info("risposta del modello: %d tool call richieste", len(risposta.tool_calls))
        if not risposta.tool_calls and _testo_vuoto(risposta.content):
            logger.warning("risposta vuota del modello, sostituita con un avviso")
            risposta = AIMessage(RISPOSTA_VUOTA)
        return {"messages": [risposta]}

    grafo = StateGraph(Stato)
    grafo.add_node("modello", modello)
    grafo.add_edge(START, "modello")
    grafo.add_edge("modello", END)
    return grafo.compile()
