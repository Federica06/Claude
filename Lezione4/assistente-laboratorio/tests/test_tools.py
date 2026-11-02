import re
from datetime import datetime, timezone

import pytest

from app.tools import tooltime


def fissa(monkeypatch, *istante):
    monkeypatch.setattr("app.tools.adesso", lambda: datetime(*istante, tzinfo=timezone.utc))


def test_roma_inverno(monkeypatch):
    fissa(monkeypatch, 2026, 11, 1, 18, 42, 10)
    assert tooltime.invoke({"fuso_orario": "Europe/Rome"}) == "2026-11-01T19:42:10+01:00 Europe/Rome"


def test_roma_estate(monkeypatch):
    fissa(monkeypatch, 2026, 7, 1, 10, 0, 0)
    assert tooltime.invoke({"fuso_orario": "Europe/Rome"}) == "2026-07-01T12:00:00+02:00 Europe/Rome"


def test_tokyo(monkeypatch):
    fissa(monkeypatch, 2026, 11, 1, 18, 42, 10)
    assert tooltime.invoke({"fuso_orario": "Asia/Tokyo"}) == "2026-11-02T03:42:10+09:00 Asia/Tokyo"


def test_formato(monkeypatch):
    fissa(monkeypatch, 2026, 11, 1, 18, 42, 10)
    risultato = tooltime.invoke({"fuso_orario": "America/New_York"})
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2} \S+", risultato)


@pytest.mark.parametrize("fuso", ["Mars/Base", ""])
def test_fuso_non_valido(monkeypatch, fuso):
    fissa(monkeypatch, 2026, 11, 1, 18, 42, 10)
    risultato = tooltime.invoke({"fuso_orario": fuso})
    assert risultato.startswith("Errore: fuso orario sconosciuto")
    assert "Europe/Rome" in risultato
