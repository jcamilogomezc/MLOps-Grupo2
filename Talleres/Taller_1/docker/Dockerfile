FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models \
    APP_HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash appuser
WORKDIR ${APP_HOME}

# instalar deps
COPY requirements/requirements.txt .
RUN pip install -r requirements.txt

# copiar código y modelos (como ya están en el repo)
COPY app/ app/
COPY models/ /models/

RUN chown -R appuser:appuser ${APP_HOME} /models
USER appuser

EXPOSE 8989
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8989"]