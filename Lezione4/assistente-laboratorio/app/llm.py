"""Client del modello servito da LM Studio (API compatibile con OpenAI)."""

from langchain_openai import ChatOpenAI

from app.config import Config
from app.tools import tooltime


def crea_llm(config: Config):
    # LM Studio non richiede chiavi: "lm-studio" è solo un segnaposto.
    # max_tokens alto: il modello ragiona prima di rispondere e il ragionamento consuma token.
    return ChatOpenAI(
        base_url=config.base_url,
        model=config.model,
        api_key="lm-studio",
        temperature=0.2,
        timeout=300,
        max_tokens=4096,
    ).bind_tools([tooltime])
