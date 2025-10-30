"""
Master DAG: Complete ML Pipeline
This DAG orchestrates the entire ML pipeline from data download to production deployment.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from dag_01_download_data import download_diabetes_data
from dag_02_split_data import split_dataset
from dag_03_store_raw_data import store_train_data, store_validation_data, store_test_data
from dag_04_clean_training_data import clean_training_data
from dag_05_train_models import train_models
from dag_06_select_best_model import select_best_model
from dag_07_publish_to_production import publish_model_to_production

from airflow.sensors.external_task import ExternalTaskSensor


# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Create the master DAG
with DAG(
    'dag_00_master_pipeline',
    default_args=default_args,
    description='Master pipeline orchestrating the complete ML workflow',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['diabetes', 'ml-pipeline', 'master'],
) as dag:

    download_task = PythonOperator(task_id='download_diabetes_dataset', python_callable=download_diabetes_data)    
    split_task = PythonOperator(task_id='split_train_val_test', python_callable=split_dataset)
    store_train_task = PythonOperator(task_id='store_train_data', python_callable=store_train_data)
    store_val_task = PythonOperator(task_id='store_validation_data', python_callable=store_validation_data)
    store_test_task = PythonOperator(task_id='store_test_data', python_callable=store_test_data)
    clean_task = PythonOperator(task_id='clean_training_dataset', python_callable=clean_training_data)
    train_task = PythonOperator(task_id='train_ml_models', python_callable=train_models)
    select_task = PythonOperator(task_id='select_best_model_validation', python_callable=select_best_model)
    publish_task = PythonOperator(task_id='publish_model_production', python_callable=publish_model_to_production)

    download_task >> split_task >> [store_train_task, store_val_task, store_test_task] >> clean_task >> train_task >> select_task >> publish_task
