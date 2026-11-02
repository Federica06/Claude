import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import PROMPT, RISPOSTA_VUOTA, costruisci_grafo
from tests.conftest import ModelloFinto


def esegui(modello, max_steps=5, messages=None):
    grafo = costruisci_grafo(modello, max_steps)
    return grafo.invoke({"messages": messages or [HumanMessage("ciao")], "steps": 0})


def test_risposta_fissa():
    risultato = esegui(ModelloFinto([AIMessage("Risposta fissa")]))
    assert risultato["messages"][-1].content == "Risposta fissa"
    assert len(risultato["messages"]) == 2
    assert risultato["steps"] == 0


def test_prompt_di_sistema_non_in_cronologia():
    modello = ModelloFinto([AIMessage("ok")])
    risultato = esegui(modello)
    primo = modello.chiamate[0][0]
    assert isinstance(primo, SystemMessage)
    assert primo.content == PROMPT
    assert not any(isinstance(m, SystemMessage) for m in risultato["messages"])


@pytest.mark.parametrize("testo", ["", "   "])
def test_risposta_vuota_sostituita(testo, caplog):
    with caplog.at_level(logging.INFO):
        risultato = esegui(ModelloFinto([AIMessage(testo)]))
    ultimo = risultato["messages"][-1]
    assert isinstance(ultimo, AIMessage)
    assert ultimo.content == RISPOSTA_VUOTA
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_log_nodo_modello(caplog):
    with caplog.at_level(logging.INFO):
        esegui(ModelloFinto([AIMessage("ok")]))
    assert "chiamata al modello: 2 messaggi inviati" in caplog.messages
    assert "risposta del modello: 0 tool call richieste" in caplog.messages
