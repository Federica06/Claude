from app.config import Config
from app.llm import crea_llm


def test_parametri_client():
    llm = crea_llm(Config("http://127.0.0.1:1234/v1", "gemma-4-e2b-it", 5))
    client = llm.bound
    assert client.openai_api_base == "http://127.0.0.1:1234/v1"
    assert client.model_name == "gemma-4-e2b-it"
    assert client.openai_api_key.get_secret_value() == "lm-studio"
    assert client.temperature == 0.2
    assert client.request_timeout == 300
    assert client.max_tokens == 4096
    assert llm.kwargs["tools"][0]["function"]["name"] == "tooltime"
