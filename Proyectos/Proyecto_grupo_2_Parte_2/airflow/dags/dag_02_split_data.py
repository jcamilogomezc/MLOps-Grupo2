"""
DAG 2: Split Data into Train, Validation, and Test Sets
This DAG splits the downloaded dataset into training (70%), validation (15%), and test (15%) sets.
"""

from datetime import datetime, timedelta
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def split_dataset(**context):
    """
    Split the dataset into train (70%), validation (15%), and test (15%) sets.
    """
    # Get the data filepath
    data_filepath = '/opt/airflow/data/Diabetes/Diabetes.csv'

    print(f"Reading data from: {data_filepath}")

    # Read the dataset
    df = pd.read_csv(data_filepath)
    print(f"Total dataset size: {len(df)} rows, {df.shape[1]} columns")

    # Split: 70% train, 30% temp (which will be split into 15% val + 15% test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df['readmitted']
    )

    # Split temp into 50% validation and 50% test (each 15% of original)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df['readmitted']
    )

    # Print split statistics
    print(f"Training set: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Save the split datasets
    data_dir = os.path.dirname(data_filepath)
    train_path = os.path.join(data_dir, 'train_raw.csv')
    val_path = os.path.join(data_dir, 'validation_raw.csv')
    test_path = os.path.join(data_dir, 'test_raw.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to: {train_path}")
    print(f"Validation data saved to: {val_path}")
    print(f"Test data saved to: {test_path}")

    # Push file paths to XCom
    ti = context['ti']
    ti.xcom_push(key='train_path', value=train_path)
    ti.xcom_push(key='val_path', value=val_path)
    ti.xcom_push(key='test_path', value=test_path)

    return {
        'train_path': train_path,
        'val_path': val_path,
        'test_path': test_path,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df)
    }

