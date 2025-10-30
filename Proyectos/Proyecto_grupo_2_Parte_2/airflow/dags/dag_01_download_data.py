"""
DAG 1: Download Diabetes Dataset
This DAG downloads the diabetes dataset from Google Drive and saves it locally.
"""

import logging
from datetime import datetime, timedelta
import os
import requests

logger = logging.getLogger("download_diabetes_dataset")

# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}


def download_diabetes_data(**context):
    """
    Download the Diabetes dataset from Google Drive.
    """
    logger.info("Start dag download_diabetes_data")
    # Directory of the raw data files
    data_root = '/opt/airflow/data/Diabetes'
    data_filepath = os.path.join(data_root, 'Diabetes.csv')

    # Create directory if it doesn't exist
    os.makedirs(data_root, exist_ok=True)

    # Download data if not already present
    if not os.path.isfile(data_filepath):
        print(f"Downloading dataset to {data_filepath}...")
        url = 'https://docs.google.com/uc?export=download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'
        r = requests.get(url, allow_redirects=True, stream=True)

        # Save the file
        with open(data_filepath, 'wb') as f:
            f.write(r.content)

        print(f"Dataset downloaded successfully to {data_filepath}")
    else:
        print(f"Dataset already exists at {data_filepath}")

    # Verify the file exists and has content
    file_size = os.path.getsize(data_filepath)
    print(f"Dataset file size: {file_size / 1024 / 1024:.2f} MB")

    # Push the file path to XCom for downstream tasks
    context['ti'].xcom_push(key='data_filepath', value=data_filepath)

    return data_filepath
