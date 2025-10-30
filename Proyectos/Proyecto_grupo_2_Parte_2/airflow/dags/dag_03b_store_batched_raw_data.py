"""
DAG 3b: Store Raw Data in PostgreSQL in Batches
This DAG stores the training dataset in batches of 15,000 records in separate database tables.
- train_batch_0: records 0-14,999 (15,000 records)
- train_batch_1: records 15,000-29,999 (15,000 records)
- train_batch_2: records 30,000-44,999 (15,000 records)
- train_batch_3: records 45,000-59,999 (15,000 records)
- train_batch_4: records 60,000-74,999 (15,000 records)
- train_batch_5: records 75,000-89,999 (15,000 records)
- train_batch_6: records 90,000-101,766 (11,767 records)

Total: 7 batch tables for cumulative training
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import os
import pandas as pd
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


def create_batch_tables(**context):
    """
    Split training data into batches of 15,000 records and store each batch
    in a separate PostgreSQL table for cumulative training.
    """
    data_dir = '/opt/airflow/data/Diabetes'
    train_path = os.path.join(data_dir, 'train_raw.csv')

    print(f"Loading training data from: {train_path}")
    df_train = pd.read_csv(train_path)

    total_records = len(df_train)
    batch_size = 15000
    num_batches = int(np.ceil(total_records / batch_size))

    print(f"\nTraining Data Information:")
    print(f"  Total records: {total_records}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")

    # Shuffle data for random distribution
    df_train_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # PostgreSQL connection
    postgres_hook = PostgresHook(postgres_conn_id='raw_data')

    # Infer column types from pandas DataFrame
    type_mapping = {
        'int64': 'INTEGER',
        'float64': 'DOUBLE PRECISION',
        'object': 'TEXT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP'
    }

    # Create table schema from first batch
    columns_def = []
    for col, dtype in df_train_shuffled.dtypes.items():
        pg_type = type_mapping.get(str(dtype), 'TEXT')
        columns_def.append(f'"{col.lower()}" {pg_type}')

    columns_def_str = ', '.join(columns_def)

    batch_info = []

    # Create and populate batch tables
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_records)

        df_batch = df_train_shuffled.iloc[start_idx:end_idx]
        batch_records = len(df_batch)
        table_name = f'train_batch_{batch_idx}'

        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_idx}")
        print(f"{'='*60}")
        print(f"  Records: {start_idx} to {end_idx-1} ({batch_records} total)")
        print(f"  Table: {table_name}")

        # Create table
        create_table_sql = f"""
        DROP TABLE IF EXISTS {table_name} CASCADE;
        CREATE TABLE {table_name} (
            batch_id INTEGER,
            {columns_def_str}
        );
        """

        print(f"  Creating table {table_name}...")
        postgres_hook.run(create_table_sql)

        # Add batch_id column
        df_batch_with_id = df_batch.copy()
        df_batch_with_id.insert(0, 'batch_id', batch_idx)

        # Prepare insert statement
        columns = ', '.join([f'"{col}"' for col in df_batch_with_id.columns])
        placeholders = ', '.join(['%s'] * len(df_batch_with_id.columns))
        insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        # Convert dataframe to list of tuples
        records = [tuple(row) for row in df_batch_with_id.values]

        # Insert in smaller batches for efficiency
        insert_batch_size = 1000
        print(f"  Inserting {batch_records} records...")

        for i in range(0, len(records), insert_batch_size):
            insert_batch = records[i:i + insert_batch_size]
            postgres_hook.insert_rows(
                table_name,
                insert_batch,
                target_fields=df_batch_with_id.columns.tolist()
            )

        # Verify insertion
        count_sql = f"SELECT COUNT(*) FROM {table_name}"
        result = postgres_hook.get_first(count_sql)
        actual_count = result[0] if result else 0

        print(f"  ✓ Successfully stored {actual_count} records in {table_name}")

        batch_info.append({
            'batch_id': batch_idx,
            'table_name': table_name,
            'records': batch_records,
            'start_idx': start_idx,
            'end_idx': end_idx - 1
        })

    # Push batch information to XCom
    context['ti'].xcom_push(key='num_batches', value=num_batches)
    context['ti'].xcom_push(key='batch_info', value=batch_info)
    context['ti'].xcom_push(key='total_records', value=total_records)

    print(f"\n{'='*60}")
    print(f"✓ Batch Storage Complete")
    print(f"{'='*60}")
    print(f"  Total batches created: {num_batches}")
    print(f"  Total records stored: {sum(b['records'] for b in batch_info)}")
    print(f"  Database: raw_data (PostgreSQL)")

    return {
        'num_batches': num_batches,
        'total_records': total_records,
        'batch_info': batch_info
    }


def create_cumulative_views(**context):
    """
    Create PostgreSQL views for cumulative data access:
    - train_cumulative_0: batches 0 (15k records)
    - train_cumulative_1: batches 0-1 (30k records)
    - train_cumulative_2: batches 0-2 (45k records)
    - etc.

    This allows easy access to cumulative datasets for training.
    """
    batch_info = context['ti'].xcom_pull(task_ids='create_batch_tables', key='batch_info')
    num_batches = len(batch_info)

    postgres_hook = PostgresHook(postgres_conn_id='raw_data')

    print(f"\nCreating cumulative views for {num_batches} batches...\n")

    for cumulative_idx in range(num_batches):
        view_name = f'train_cumulative_{cumulative_idx}'

        # Get all batch tables from 0 to cumulative_idx
        batch_tables = [f'train_batch_{i}' for i in range(cumulative_idx + 1)]

        # Create UNION ALL query
        union_query = ' UNION ALL '.join([f'SELECT * FROM {table}' for table in batch_tables])

        create_view_sql = f"""
        DROP VIEW IF EXISTS {view_name} CASCADE;
        CREATE VIEW {view_name} AS
        {union_query};
        """

        print(f"Creating view: {view_name}")
        print(f"  Includes batches: 0 to {cumulative_idx}")

        postgres_hook.run(create_view_sql)

        # Count records in view
        count_sql = f"SELECT COUNT(*) FROM {view_name}"
        result = postgres_hook.get_first(count_sql)
        record_count = result[0] if result else 0

        expected_records = sum(batch_info[i]['records'] for i in range(cumulative_idx + 1))

        print(f"  ✓ View created with {record_count} records (expected: {expected_records})")

        if record_count != expected_records:
            raise ValueError(
                f"View {view_name} has incorrect record count! "
                f"Expected: {expected_records}, Got: {record_count}"
            )

    print(f"\n✓ Successfully created {num_batches} cumulative views")

    # Push view information to XCom
    cumulative_views = [
        {
            'view_name': f'train_cumulative_{i}',
            'batches_included': list(range(i + 1)),
            'num_batches': i + 1,
            'total_records': sum(batch_info[j]['records'] for j in range(i + 1))
        }
        for i in range(num_batches)
    ]

    context['ti'].xcom_push(key='cumulative_views', value=cumulative_views)

    return cumulative_views


def store_validation_test_data(**context):
    """
    Store validation and test datasets in PostgreSQL (single tables, not batched).
    """
    data_dir = '/opt/airflow/data/Diabetes'
    postgres_hook = PostgresHook(postgres_conn_id='raw_data')

    # Type mapping
    type_mapping = {
        'int64': 'INTEGER',
        'float64': 'DOUBLE PRECISION',
        'object': 'TEXT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP'
    }

    datasets = [
        ('validation_raw', os.path.join(data_dir, 'validation_raw.csv')),
        ('test_raw', os.path.join(data_dir, 'test_raw.csv'))
    ]

    results = {}

    for table_name, csv_path in datasets:
        print(f"\n{'='*60}")
        print(f"Storing {table_name}")
        print(f"{'='*60}")
        print(f"Loading from: {csv_path}")

        df = pd.read_csv(csv_path)
        num_records = len(df)

        print(f"Records: {num_records}")

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

        # Insert data
        print(f"Inserting {num_records} records...")
        records = [tuple(row) for row in df.values]

        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            postgres_hook.insert_rows(table_name, batch, target_fields=df.columns.tolist())

        # Verify
        count_sql = f"SELECT COUNT(*) FROM {table_name}"
        result = postgres_hook.get_first(count_sql)
        actual_count = result[0] if result else 0

        print(f"✓ Successfully stored {actual_count} records in {table_name}")

        results[table_name] = {'records': actual_count}

    context['ti'].xcom_push(key='validation_test_info', value=results)

    return results
