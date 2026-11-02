import logging
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.graph import ERRORE_LIMITE, PROMPT, RISPOSTA_VUOTA, TESTO_LIMITE, costruisci_grafo
from tests.conftest import ModelloFinto


@pytest.fixture
def orologio(monkeypatch):
    """Istante fisso per tooltime; conta quante volte lo strumento legge l'ora."""
    letture = []

    def adesso():
        letture.append(1)
        return datetime(2026, 11, 1, 18, 42, 10, tzinfo=timezone.utc)

    monkeypatch.setattr("app.tools.adesso", adesso)
    return letture


def chiamata(id, args=None, nome="tooltime"):
    if args is None:
        args = {"fuso_orario": "Asia/Tokyo"}
    return {"name": nome, "args": args, "id": id, "type": "tool_call"}


def tool_messages(messages):
    return [m for m in messages if isinstance(m, ToolMessage)]


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


def turno(grafo, cronologia, testo):
    """Stesso schema della CLI: cronologia + nuova domanda, steps da zero."""
    return grafo.invoke({"messages": cronologia + [HumanMessage(testo)], "steps": 0})


def test_due_turni_senza_duplicazioni_risposta_diretta():
    grafo = costruisci_grafo(ModelloFinto([AIMessage("uno"), AIMessage("due")]), 5)

    risultato = turno(grafo, [], "prima")
    cronologia = risultato["messages"]
    assert len(cronologia) == 2
    assert risultato["steps"] == 0

    risultato = turno(grafo, cronologia, "seconda")
    cronologia = risultato["messages"]
    assert [m.content for m in cronologia] == ["prima", "uno", "seconda", "due"]
    assert len({m.id for m in cronologia}) == 4
    assert risultato["steps"] == 0


def test_due_turni_senza_duplicazioni(orologio):
    modello = ModelloFinto([
        AIMessage("", tool_calls=[chiamata("c1")]),
        AIMessage("A Tokyo sono le 03:42"),
        AIMessage("Prego"),
    ])
    grafo = costruisci_grafo(modello, 5)

    risultato = turno(grafo, [], "Che ore sono a Tokyo?")
    cronologia = risultato["messages"]
    assert [type(m) for m in cronologia] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert risultato["steps"] == 1

    risultato = turno(grafo, cronologia, "Grazie")
    cronologia = risultato["messages"]
    assert len(cronologia) == 6
    assert len({m.id for m in cronologia}) == 6
    assert risultato["steps"] == 0


def test_un_giro_con_strumento(orologio, caplog):
    modello = ModelloFinto([
        AIMessage("", tool_calls=[chiamata("c1")]),
        AIMessage("Sono le 03:42"),
    ])
    with caplog.at_level(logging.INFO):
        risultato = esegui(modello)
    messages = risultato["messages"]
    assert [type(m) for m in messages] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert messages[2].tool_call_id == "c1"
    assert messages[2].content == "2026-11-02T03:42:10+09:00 Asia/Tokyo"
    assert risultato["steps"] == 1
    assert any("tool call tooltime" in m and "Asia/Tokyo" in m for m in caplog.messages)
    assert any(isinstance(m, ToolMessage) for m in modello.chiamate[1])
    assert "risposta del modello: 1 tool call richieste" in caplog.messages


def test_strumento_sconosciuto(orologio, caplog):
    modello = ModelloFinto([
        AIMessage("", tool_calls=[chiamata("c1", nome="meteo")]),
        AIMessage("fine"),
    ])
    risultato = esegui(modello)
    (tool,) = tool_messages(risultato["messages"])
    assert tool.content.startswith("Errore")
    assert risultato["steps"] == 1
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("args", [{}, {"fuso_orario": None}, {"fuso_orario": 42}])
def test_argomenti_non_validi(orologio, caplog, args):
    modello = ModelloFinto([
        AIMessage("", tool_calls=[chiamata("c1", args=args)]),
        AIMessage("fine"),
    ])
    risultato = esegui(modello)
    (tool,) = tool_messages(risultato["messages"])
    assert tool.content.startswith("Errore")
    assert risultato["steps"] == 1
    assert risultato["messages"][-1].content == "fine"
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_tre_chiamate_oltre_limite(orologio, caplog):
    modello = ModelloFinto([
        AIMessage("", tool_calls=[chiamata("a"), chiamata("b"), chiamata("c")]),
    ])
    risultato = esegui(modello, max_steps=2)
    tools = tool_messages(risultato["messages"])
    assert orologio == []
    assert [t.tool_call_id for t in tools] == ["a", "b", "c"]
    assert all(t.content == ERRORE_LIMITE for t in tools)
    assert risultato["messages"][-1].content == TESTO_LIMITE.format(n=2)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert len(modello.chiamate) == 1


def test_limite_raggiunto_con_chiamate_singole(orologio):
    modello = ModelloFinto([
        AIMessage("", tool_calls=[chiamata("a")]),
        AIMessage("", tool_calls=[chiamata("b")]),
        AIMessage("", tool_calls=[chiamata("c")]),
    ])
    risultato = esegui(modello, max_steps=2)
    tools = tool_messages(risultato["messages"])
    assert len(orologio) == 2
    assert [t.tool_call_id for t in tools] == ["a", "b", "c"]
    assert tools[2].content == ERRORE_LIMITE
    assert risultato["messages"][-1].content == TESTO_LIMITE.format(n=2)
    assert risultato["steps"] == 2


def test_log_nodo_modello(caplog):
    with caplog.at_level(logging.INFO):
        esegui(ModelloFinto([AIMessage("ok")]))
    assert "chiamata al modello: 2 messaggi inviati" in caplog.messages
    assert "risposta del modello: 0 tool call richieste" in caplog.messages
