"""
DAG 02b: Create Parallel Data Batches
This DAG splits the dataset into batches of 15,000 records for parallel processing.
Total records: 101,767 → 7 batches (6x15,000 + 1x1,767)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import os
import numpy as np


# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def create_parallel_batches(**context):
    """
    Split the training dataset into batches of 15,000 records each for parallel processing.
    Returns the number of batches created for downstream task mapping.
    """
    data_dir = '/opt/airflow/data/Diabetes'
    train_file = os.path.join(data_dir, 'train_raw.csv')
    batch_dir = os.path.join(data_dir, 'batches')

    # Create batches directory
    os.makedirs(batch_dir, exist_ok=True)

    print(f"Loading training data from: {train_file}")
    df_train = pd.read_csv(train_file)

    total_records = len(df_train)
    batch_size = 15000
    num_batches = int(np.ceil(total_records / batch_size))

    print(f"Total training records: {total_records}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches to create: {num_batches}")

    # Shuffle the data to ensure random distribution across batches
    df_train_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    batch_info = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_records)

        df_batch = df_train_shuffled.iloc[start_idx:end_idx]
        batch_file = os.path.join(batch_dir, f'train_batch_{batch_idx}.csv')

        df_batch.to_csv(batch_file, index=False)

        batch_records = len(df_batch)
        print(f"Created batch {batch_idx}: {batch_records} records → {batch_file}")

        batch_info.append({
            'batch_id': batch_idx,
            'records': batch_records,
            'file': batch_file
        })

    # Push batch information to XCom for downstream tasks
    context['ti'].xcom_push(key='num_batches', value=num_batches)
    context['ti'].xcom_push(key='batch_info', value=batch_info)

    print(f"\n✓ Successfully created {num_batches} batches")
    print(f"Batch directory: {batch_dir}")

    return num_batches


def validate_batches(**context):
    """
    Validate that all batches were created correctly and sum to original dataset size.
    """
    batch_info = context['ti'].xcom_pull(task_ids='create_batches', key='batch_info')

    if not batch_info:
        raise ValueError("No batch information found in XCom")

    total_batch_records = sum(batch['records'] for batch in batch_info)

    # Load original train data to compare
    train_file = '/opt/airflow/data/Diabetes/train_raw.csv'
    df_train = pd.read_csv(train_file)
    original_records = len(df_train)

    print(f"Original training records: {original_records}")
    print(f"Total records in batches: {total_batch_records}")

    if total_batch_records != original_records:
        raise ValueError(
            f"Batch validation failed! "
            f"Original: {original_records}, Batches: {total_batch_records}"
        )

    print("\n✓ Batch validation successful")
    print(f"All {len(batch_info)} batches contain exactly {original_records} total records")

    # Print batch distribution
    print("\nBatch distribution:")
    for batch in batch_info:
        print(f"  Batch {batch['batch_id']}: {batch['records']} records")

    return True
