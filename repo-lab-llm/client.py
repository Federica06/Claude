"""Client minimo per il server locale di LM Studio.

Legge le domande da terminale e stampa la risposta in streaming.
Uscire con «esci» o Ctrl+C.
"""

import json
import os

import requests
from dotenv import load_dotenv

# --- Configurazione -------------------------------------------------------
load_dotenv()
base_url = os.getenv("LM_BASE_URL")
model = os.getenv("LM_MODEL")

# --- Messaggi -------------------------------------------------------------
system = {
    "role": "system",
    "content": "Sei un assistente utile. Rispondi in italiano, in modo chiaro e conciso.",
}


# --- Chiamata -------------------------------------------------------------
def chiedi(domanda):
    """Invia la domanda, stampa la risposta in streaming e la restituisce."""
    payload = {
        "model": model,
        "messages": [system, {"role": "user", "content": domanda}],
        "temperature": 0.7,
        "max_tokens": 512,
        "reasoning_effort": "none",
        "stream": True,
    }
    try:
        with requests.post(
            f"{base_url}/chat/completions", json=payload, stream=True, timeout=60
        ) as response:
            if not response.ok:
                response.content  # legge il corpo prima che il with chiuda la connessione
            response.raise_for_status()
            parti = []
            for riga in response.iter_lines():
                riga = riga.decode("utf-8")
                if not riga.startswith("data:"):
                    continue
                dati = riga[len("data:"):].strip()
                if dati == "[DONE]":
                    break
                scelte = json.loads(dati).get("choices") or [{}]
                frammento = scelte[0].get("delta", {}).get("content")
                if frammento:
                    print(frammento, end="", flush=True)
                    parti.append(frammento)
            print()
            return "".join(parti)
    except requests.exceptions.Timeout:
        print("Attesa scaduta: controllare server, carico e tempi.")
    except requests.exceptions.ConnectionError:
        print("Connessione fallita: controllare server, URL e porta.")
    except requests.exceptions.HTTPError:
        print("Errore HTTP:", response.status_code, response.text)
    return None


# --- Ciclo di input -------------------------------------------------------
try:
    while True:
        domanda = input("Domanda: ").strip()
        if not domanda:
            continue
        if domanda.lower() == "esci":
            break
        chiedi(domanda)
except (KeyboardInterrupt, EOFError):
    print()
print("Arrivederci.")
