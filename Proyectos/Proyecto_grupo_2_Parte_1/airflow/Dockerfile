# Dockerfile para Airflow con Python 3.12
FROM apache/airflow:3.0.6-python3.12

COPY requirements.txt .

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential default-libmysqlclient-dev gcc curl libgomp1 gosu \
 && rm -rf /var/lib/apt/lists/*
USER airflow


# Instalar dependencias de Python sin guardar en caché para reducir el tamaño de la imagen
RUN pip install --no-cache-dir -r requirements.txt