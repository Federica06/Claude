# Stessa versione di Python dell'host
FROM python:3.10-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY tests/ tests/

CMD ["python", "-m", "app.cli"]
