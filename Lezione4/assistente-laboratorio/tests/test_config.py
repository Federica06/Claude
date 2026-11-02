import pytest

from app.config import Config, ConfigError, carica_config

VALIDO = "LM_BASE_URL=http://127.0.0.1:1234/v1\nLM_MODEL=gemma-4-e2b-it\nMAX_STEPS=5\n"
ATTESO = Config("http://127.0.0.1:1234/v1", "gemma-4-e2b-it", 5)


def scrivi_env(tmp_path, testo, encoding="utf-8"):
    percorso = tmp_path / ".env"
    percorso.write_text(testo, encoding=encoding)
    return percorso


def test_config_valida_da_file(tmp_path):
    assert carica_config(scrivi_env(tmp_path, VALIDO), ambiente={}) == ATTESO


@pytest.mark.parametrize("chiave", ["LM_BASE_URL", "LM_MODEL", "MAX_STEPS"])
def test_variabile_mancante(tmp_path, chiave):
    testo = "".join(r + "\n" for r in VALIDO.splitlines() if not r.startswith(chiave))
    with pytest.raises(ConfigError) as errore:
        carica_config(scrivi_env(tmp_path, testo), ambiente={})
    assert str(errore.value).startswith("Configurazione non valida: ")
    assert chiave in str(errore.value)


@pytest.mark.parametrize("valore", ["0", "-1", "abc"])
def test_max_steps_non_valido(tmp_path, valore):
    testo = VALIDO.replace("MAX_STEPS=5", f"MAX_STEPS={valore}")
    with pytest.raises(ConfigError) as errore:
        carica_config(scrivi_env(tmp_path, testo), ambiente={})
    assert "MAX_STEPS" in str(errore.value)
    assert f'"{valore}"' in str(errore.value)


def test_ambiente_ha_precedenza(tmp_path):
    config = carica_config(
        scrivi_env(tmp_path, VALIDO),
        ambiente={"LM_BASE_URL": "http://host.docker.internal:1234/v1"},
    )
    assert config.base_url == "http://host.docker.internal:1234/v1"


def test_commenti_e_righe_vuote(tmp_path):
    testo = "# configurazione\n\nLM_BASE_URL=http://127.0.0.1:1234/v1\n  # altro\n\nLM_MODEL=gemma-4-e2b-it\nMAX_STEPS=5\n"
    assert carica_config(scrivi_env(tmp_path, testo), ambiente={}) == ATTESO


def test_file_assente_con_ambiente(tmp_path):
    ambiente = {
        "LM_BASE_URL": "http://127.0.0.1:1234/v1",
        "LM_MODEL": "gemma-4-e2b-it",
        "MAX_STEPS": "5",
    }
    assert carica_config(tmp_path / "manca.env", ambiente=ambiente) == ATTESO


def test_bom_utf8(tmp_path):
    percorso = scrivi_env(tmp_path, VALIDO, encoding="utf-8-sig")
    assert carica_config(percorso, ambiente={}) == ATTESO


def test_valori_con_uguale_e_virgolette(tmp_path):
    testo = "LM_BASE_URL=\"http://x:1/v1?a=b\"\nLM_MODEL='gemma-4-e2b-it'\nMAX_STEPS=5\n"
    config = carica_config(scrivi_env(tmp_path, testo), ambiente={})
    assert config.base_url == "http://x:1/v1?a=b"
    assert config.model == "gemma-4-e2b-it"
