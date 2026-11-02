from app.config import Config
from app.llm import crea_llm


def test_parametri_client():
    llm = crea_llm(Config("http://127.0.0.1:1234/v1", "gemma-4-e2b-it", 5))
    assert llm.openai_api_base == "http://127.0.0.1:1234/v1"
    assert llm.model_name == "gemma-4-e2b-it"
    assert llm.openai_api_key.get_secret_value() == "lm-studio"
    assert llm.temperature == 0.2
    assert llm.request_timeout == 300
    assert llm.max_tokens == 4096
