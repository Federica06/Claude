"""Prova con il modello reale di LM Studio: si attiva solo con RUN_REAL_TESTS=1."""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import carica_config
from app.graph import RISPOSTA_VUOTA, costruisci_grafo
from app.llm import crea_llm

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_TESTS") != "1",
    reason="prova reale: impostare RUN_REAL_TESTS=1 con LM Studio attivo",
)


def test_tokyo_usa_tooltime():
    config = carica_config()
    grafo = costruisci_grafo(crea_llm(config), config.max_steps)
    risultato = grafo.invoke(
        {"messages": [HumanMessage("Che ore sono adesso a Tokyo?")], "steps": 0},
        config={"recursion_limit": 2 * config.max_steps + 5},
    )
    messages = risultato["messages"]
    for m in messages:
        print(f"{type(m).__name__}: {m.content!r} {getattr(m, 'tool_calls', '')}")

    tools = [m for m in messages if isinstance(m, ToolMessage) and m.name == "tooltime"]
    assert tools, "il modello non ha chiamato tooltime"
    assert any("Asia/Tokyo" in t.content for t in tools)
    ultimo = messages[-1]
    assert isinstance(ultimo, AIMessage)
    assert ultimo.content.strip()
    assert ultimo.content != RISPOSTA_VUOTA
