import logging

import httpx
import openai
from langchain_core.messages import AIMessage, HumanMessage

from app.cli import ERRORE_CONNESSIONE, ciclo, configura_log, main


def lettore(righe):
    """Simula input(): restituisce le righe in ordine, poi EOF."""
    righe = list(righe)

    def leggi(prompt=""):
        if not righe:
            raise EOFError
        return righe.pop(0)

    return leggi


class GrafoFinto:
    """Registra gli input; ogni comportamento è un'eccezione da sollevare o un testo di risposta."""

    def __init__(self, comportamenti):
        self.comportamenti = list(comportamenti)
        self.input = []

    def invoke(self, input, config=None):
        self.input.append(input)
        azione = self.comportamenti.pop(0)
        if isinstance(azione, BaseException):
            raise azione
        return {"messages": input["messages"] + [AIMessage(azione)], "steps": 0}


def test_configura_log_silenzia_richieste_http():
    # con le versioni installate le righe "HTTP Request: POST" arrivano dal logger httpx2
    configura_log()
    for nome in ("httpx", "httpx2"):
        # livello impostato sul logger stesso: sotto pytest quello effettivo eredita WARNING dalla radice
        assert logging.getLogger(nome).level == logging.WARNING


def test_config_non_valida_esce_con_2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for chiave in ("LM_BASE_URL", "LM_MODEL", "MAX_STEPS"):
        monkeypatch.delenv(chiave, raising=False)
    (tmp_path / ".env").write_text(
        "LM_BASE_URL=http://127.0.0.1:1234/v1\nLM_MODEL=gemma-4-e2b-it\nMAX_STEPS=abc\n",
        encoding="utf-8",
    )
    assert main() == 2
    catturato = capsys.readouterr()
    assert "Configurazione non valida:" in catturato.err
    assert "Configurazione non valida:" not in catturato.out


def test_errore_connessione_cronologia_intatta():
    errore = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    grafo = GrafoFinto([errore, "ok"])
    uscite = []
    ciclo(grafo, 5, leggi=lettore(["domanda 1", "domanda 2", "esci"]), scrivi=uscite.append)

    assert ERRORE_CONNESSIONE in uscite
    secondo = grafo.input[1]["messages"]
    assert len(secondo) == 1
    assert isinstance(secondo[0], HumanMessage)
    assert secondo[0].content == "domanda 2"
    assert "Assistente> ok" in uscite


def test_input_vuoto_o_spazi_ignorato():
    grafo = GrafoFinto([])
    ciclo(grafo, 5, leggi=lettore(["", "   ", "esci"]), scrivi=lambda _: None)
    assert grafo.input == []


def test_ctrl_c_durante_risposta():
    grafo = GrafoFinto([KeyboardInterrupt()])
    uscite = []
    ciclo(grafo, 5, leggi=lettore(["domanda"]), scrivi=uscite.append)
    assert uscite[-1] == "Arrivederci!"


def test_eof_termina():
    uscite = []
    ciclo(GrafoFinto([]), 5, leggi=lettore([]), scrivi=uscite.append)
    assert uscite[-1] == "Arrivederci!"


def test_passa_steps_zero_e_recursion_limit():
    grafo = GrafoFinto(["ok"])
    configurazioni = []
    originale = grafo.invoke

    def invoke(input, config=None):
        configurazioni.append(config)
        return originale(input, config)

    grafo.invoke = invoke
    ciclo(grafo, 5, leggi=lettore(["domanda", "esci"]), scrivi=lambda _: None)
    assert grafo.input[0]["steps"] == 0
    assert configurazioni[0] == {"recursion_limit": 15}
