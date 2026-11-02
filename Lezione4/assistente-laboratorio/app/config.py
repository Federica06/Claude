"""Configurazione da file .env e variabili d'ambiente (l'ambiente ha la precedenza)."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


class ConfigError(Exception):
    """Configurazione mancante o non valida."""


@dataclass(frozen=True)
class Config:
    base_url: str
    model: str
    max_steps: int


def _leggi_env(percorso: Path) -> dict:
    """Legge righe CHIAVE=valore; ignora righe vuote e commenti. File assente → dizionario vuoto."""
    valori = {}
    if not percorso.is_file():
        return valori
    # utf-8-sig: PowerShell 5.1 salva i file UTF-8 con BOM
    for riga in percorso.read_text(encoding="utf-8-sig").splitlines():
        riga = riga.strip()
        if not riga or riga.startswith("#") or "=" not in riga:
            continue
        chiave, valore = riga.split("=", 1)
        valore = valore.strip()
        if len(valore) >= 2 and valore[0] == valore[-1] and valore[0] in "\"'":
            valore = valore[1:-1]
        valori[chiave.strip()] = valore
    return valori


def carica_config(
    percorso: Path = Path(".env"), ambiente: Optional[Mapping[str, str]] = None
) -> Config:
    if ambiente is None:
        ambiente = os.environ
    valori = {**_leggi_env(percorso), **ambiente}

    for chiave in ("LM_BASE_URL", "LM_MODEL", "MAX_STEPS"):
        if not valori.get(chiave, "").strip():
            raise ConfigError(f"Configurazione non valida: manca {chiave}")

    grezzo = valori["MAX_STEPS"].strip()
    try:
        max_steps = int(grezzo)
    except ValueError:
        max_steps = 0
    if max_steps <= 0:
        raise ConfigError(
            f'Configurazione non valida: MAX_STEPS deve essere un intero > 0 (trovato "{grezzo}")'
        )

    return Config(valori["LM_BASE_URL"].strip(), valori["LM_MODEL"].strip(), max_steps)
