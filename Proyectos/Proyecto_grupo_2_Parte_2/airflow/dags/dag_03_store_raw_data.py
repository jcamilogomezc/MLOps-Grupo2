"""
DAG 3: Store Raw Data in PostgreSQL
This DAG stores the raw train, validation, and test datasets in the postgres-raw-data database.
"""

from datetime import datetime, timedelta
from airflow.providers.postgres.hooks.postgres import PostgresHook
import os
import pandas as pd


# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def create_table_from_csv(table_name: str, csv_path: str, postgres_conn_id: str = 'raw_data'):
    """
    Create a PostgreSQL table and insert data from CSV using PostgresHook.
    """
    postgres_hook = PostgresHook(postgres_conn_id=postgres_conn_id)

    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Infer column types from pandas DataFrame
    type_mapping = {
        'int64': 'INTEGER',
        'float64': 'DOUBLE PRECISION',
        'object': 'TEXT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP'
    }

    # Create table schema
    columns_def = []
    for col, dtype in df.dtypes.items():
        pg_type = type_mapping.get(str(dtype), 'TEXT')
        columns_def.append(f'"{col.lower()}" {pg_type}')

    create_table_sql = f"""
    DROP TABLE IF EXISTS {table_name} CASCADE;
    CREATE TABLE {table_name} (
        {', '.join(columns_def)}
    );
    """

    print(f"Creating table {table_name}...")
    postgres_hook.run(create_table_sql)

    # Insert data in batches using COPY or INSERT
    print(f"Inserting data into {table_name}...")

    # Prepare insert statement
    columns = ', '.join([f'"{col}"' for col in df.columns])
    placeholders = ', '.join(['%s'] * len(df.columns))
    insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    # Convert dataframe to list of tuples
    records = [tuple(row) for row in df.values]

    # Insert in batches
    batch_size = 1000
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        postgres_hook.insert_rows(table_name, batch, target_fields=df.columns.tolist())

    print(f"Successfully inserted {len(df)} rows into {table_name}")

    return len(df)


def store_train_data(**context):
    """Store training dataset in PostgreSQL."""
    data_dir = '/opt/airflow/data/Diabetes'
    train_path = os.path.join(data_dir, 'train_raw.csv')

    num_rows = create_table_from_csv('train_raw', train_path)

    ti = context['ti']
    ti.xcom_push(key='train_table', value='train_raw')
    ti.xcom_push(key='train_rows', value=num_rows)

    return {'table': 'train_raw', 'rows': num_rows}


def store_validation_data(**context):
    """Store validation dataset in PostgreSQL."""
    data_dir = '/opt/airflow/data/Diabetes'
    val_path = os.path.join(data_dir, 'validation_raw.csv')

    num_rows = create_table_from_csv('validation_raw', val_path)

    ti = context['ti']
    ti.xcom_push(key='val_table', value='validation_raw')
    ti.xcom_push(key='val_rows', value=num_rows)

    return {'table': 'validation_raw', 'rows': num_rows}


def store_test_data(**context):
    """Store test dataset in PostgreSQL."""
    data_dir = '/opt/airflow/data/Diabetes'
    test_path = os.path.join(data_dir, 'test_raw.csv')

    num_rows = create_table_from_csv('test_raw', test_path)

    ti = context['ti']
    ti.xcom_push(key='test_table', value='test_raw')
    ti.xcom_push(key='test_rows', value=num_rows)

    return {'table': 'test_raw', 'rows': num_rows}
