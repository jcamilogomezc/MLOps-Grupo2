# Dockerfile (uv + providers + ciencia de datos)
FROM apache/airflow:2.6.0

COPY requirements.txt .
# COPY dags/ /opt/airflow/dags/

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential default-libmysqlclient-dev gcc curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# chmod -R 755 /opt/airflow/dags/

USER airflow
# # Instalar uv (gestor rápido de deps)
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# ENV PATH="/home/airflow/.local/bin:${PATH}"


# # Verificar que uv quedó instalado (esto imprime versión en el log de build)
# RUN uv --version


# Instalar TODO con uv en el entorno del sistema (sin pip)
# RUN uv pip install --system --no-cache-dir \
#     mysqlclient==2.1.1 \
#     apache-airflow-providers-mysql==5.0.0 \
#     numpy==1.21.6 pandas==1.3.5 scikit-learn==1.0.2 joblib==1.2.0

RUN pip install -r requirements.txt

    # (Opcional) Verificación visible en el build log
# RUN python - <<'PY'
# import sys, numpy, pandas, sklearn, joblib, MySQLdb
# print("PY", sys.version.split()[0])
# print("numpy", numpy.__version__)
# print("pandas", pandas.__version__)
# print("sklearn", sklearn.__version__)
# print("joblib", joblib.__version__)
# PY