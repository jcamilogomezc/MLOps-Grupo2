FROM apache/airflow:slim-latest-python3.13

# Install additional python packages
USER root
RUN apt-get update && apt-get install -y \
    vim \
    && apt-get clean

# Switch back to Airflow user
USER airflow

# Install Python deps
COPY requirements/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Optionally copy DAGs / plugins
COPY dags/ /opt/airflow/dags/
