from langchain_core.messages import AIMessage


class ModelloFinto:
    """Modello che restituisce risposte scriptate e registra i messaggi ricevuti."""

    def __init__(self, risposte: list[AIMessage]):
        self.risposte = list(risposte)
        self.chiamate = []

    def invoke(self, messages):
        self.chiamate.append(list(messages))
        return self.risposte.pop(0)
