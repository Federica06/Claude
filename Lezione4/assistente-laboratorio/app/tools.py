"""Strumenti richiamabili dal modello tramite tool calling."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


def adesso() -> datetime:
    """Istante corrente in UTC; i test lo sostituiscono con un istante fisso."""
    return datetime.now(timezone.utc)


@tool
def tooltime(fuso_orario: str) -> str:
    """Restituisce data e ora attuali nel fuso orario indicato, con offset da UTC.

    fuso_orario: nome IANA del fuso, per esempio Europe/Rome o Asia/Tokyo.
    """
    try:
        zona = ZoneInfo(fuso_orario)
    except (ZoneInfoNotFoundError, ValueError):
        return (
            f'Errore: fuso orario sconosciuto "{fuso_orario}". '
            "Usa un nome IANA come Europe/Rome."
        )
    return f"{adesso().astimezone(zona).isoformat(timespec='seconds')} {fuso_orario}"
