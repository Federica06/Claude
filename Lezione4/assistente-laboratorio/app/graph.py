"""Grafo LangGraph dell'assistente: stato, nodi e instradamento."""

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.tools import tooltime

logger = logging.getLogger(__name__)

STRUMENTI = {"tooltime": tooltime}

PROMPT = (
    "Sei l'assistente del laboratorio. Per conoscere data o ora attuali usa sempre "
    "lo strumento tooltime con un fuso orario IANA (per esempio Europe/Rome). "
    "Se l'utente non indica un luogo o un fuso, chiediglielo. Non inventare mai l'ora."
)
RISPOSTA_VUOTA = (
    "Il modello ha restituito una risposta vuota: di solito i token sono finiti "
    "nel ragionamento. Riprova."
)
TESTO_LIMITE = (
    "Ho raggiunto il limite di {n} passaggi con gli strumenti e mi fermo qui "
    "senza una risposta completa."
)
ERRORE_LIMITE = "Errore: strumento non eseguito, limite di passaggi raggiunto."


class Stato(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int  # esecuzioni di tooltime nella domanda corrente


def _testo_vuoto(contenuto) -> bool:
    if isinstance(contenuto, str):
        return not contenuto.strip()
    return not contenuto  # contenuto a blocchi: vuoto solo se la lista è vuota


def _esegui(chiamata: dict) -> str:
    """Esegue una tool call; ogni problema diventa un testo d'errore per il modello."""
    strumento = STRUMENTI.get(chiamata["name"])
    if strumento is None:
        risultato = f'Errore: strumento sconosciuto "{chiamata["name"]}".'
    else:
        try:
            risultato = strumento.invoke(chiamata["args"])
        except Exception as errore:  # argomenti non validi: il modello deve poter correggersi
            risultato = f"Errore: argomenti non validi per {chiamata['name']}: {errore}"
    livello = logging.WARNING if risultato.startswith("Errore") else logging.INFO
    logger.log(livello, "tool call %s args=%s -> %s", chiamata["name"], chiamata["args"], risultato)
    return risultato


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

    def strumenti(state: Stato) -> dict:
        chiamate = state["messages"][-1].tool_calls
        risposte = [
            ToolMessage(_esegui(c), tool_call_id=c["id"], name=c["name"]) for c in chiamate
        ]
        return {"messages": risposte, "steps": state["steps"] + len(chiamate)}

    def limite(state: Stato) -> dict:
        # ogni tool call deve avere una risposta, altrimenti la cronologia non è valida per il modello
        chiuse = [
            ToolMessage(ERRORE_LIMITE, tool_call_id=c["id"], name=c["name"])
            for c in state["messages"][-1].tool_calls
        ]
        logger.warning("limite di %d passaggi con gli strumenti raggiunto", max_steps)
        return {"messages": chiuse + [AIMessage(TESTO_LIMITE.format(n=max_steps))]}

    def instrada(state: Stato) -> str:
        chiamate = state["messages"][-1].tool_calls
        if not chiamate:
            return END
        # conta ogni singola esecuzione: il limite non va mai superato
        if state["steps"] + len(chiamate) > max_steps:
            return "limite"
        return "strumenti"

    grafo = StateGraph(Stato)
    grafo.add_node("modello", modello)
    grafo.add_node("strumenti", strumenti)
    grafo.add_node("limite", limite)
    grafo.add_edge(START, "modello")
    grafo.add_conditional_edges("modello", instrada, ["strumenti", "limite", END])
    grafo.add_edge("strumenti", "modello")
    grafo.add_edge("limite", END)
    return grafo.compile()
