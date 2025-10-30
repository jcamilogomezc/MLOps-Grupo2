"""
DAG 4: Clean Training Data
This DAG cleans the training dataset by:
- Removing constant features
- Handling missing values (especially '?' values)
- Removing duplicate rows
- Feature engineering if needed
"""

from datetime import datetime, timedelta
from airflow.providers.postgres.hooks.postgres import PostgresHook
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


def clean_training_data(**context):
    """
    Clean the training dataset using PostgresHook.
    Reads from train_raw table and writes to train_clean table.
    """
    # Use PostgresHook to interact with the database
    postgres_hook = PostgresHook(postgres_conn_id='raw_data')

    print(f"Connecting to database to read raw training data...")

    # Read raw training data from PostgreSQL using hook
    print("Reading raw training data from train_raw table...")
    conn = postgres_hook.get_conn()
    train_df = pd.read_sql("SELECT * FROM train_raw", conn)
    conn.close()
    print(f"Raw training data shape: {train_df.shape}")

    # Save original size
    original_size = len(train_df)

    # Step 1: Remove constant features (features with only 1 unique value)
    print("\nStep 1: Removing constant features...")
    constant_cols = []
    for col in train_df.columns:
        if train_df[col].nunique() == 1:
            constant_cols.append(col)

    if constant_cols:
        print(f"  Removing {len(constant_cols)} constant features: {constant_cols}")
        train_df = train_df.drop(columns=constant_cols)
    else:
        print("  No constant features found")

    # Step 2: Handle missing values
    print("\nStep 2: Handling missing values...")

    # Replace '?' with NaN
    train_df = train_df.replace('?', np.nan)

    # Check missing values
    missing_pct = (train_df.isnull().sum() / len(train_df) * 100)
    high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)

    if len(high_missing) > 0:
        print(f"  Features with >50% missing values:")
        for col, pct in high_missing.items():
            print(f"    {col}: {pct:.2f}%")
        print(f"  Dropping these {len(high_missing)} features...")
        train_df = train_df.drop(columns=high_missing.index)

    # For remaining missing values, handle based on data type
    remaining_missing = train_df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]

    if len(remaining_missing) > 0:
        print(f"  Handling {len(remaining_missing)} features with moderate missing values...")
        for col in remaining_missing.index:
            if train_df[col].dtype == 'object':
                # For categorical: fill with mode or 'Unknown'
                mode_val = train_df[col].mode()
                if len(mode_val) > 0:
                    train_df[col] = train_df[col].fillna(mode_val[0])
                else:
                    train_df[col] = train_df[col].fillna('Unknown')
                print(f"    {col}: filled with mode/Unknown")
            else:
                # For numerical: fill with median
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                print(f"    {col}: filled with median")

    # Step 3: Remove duplicates
    print("\nStep 3: Removing duplicates...")
    duplicates = train_df.duplicated().sum()
    if duplicates > 0:
        print(f"  Found {duplicates} duplicate rows, removing...")
        train_df = train_df.drop_duplicates()
    else:
        print("  No duplicates found")

    # Step 4: Remove ID columns that don't provide predictive value
    print("\nStep 4: Removing ID columns...")
    id_columns = ['encounter_id', 'patient_nbr']
    id_cols_present = [col for col in id_columns if col in train_df.columns]
    if id_cols_present:
        print(f"  Removing ID columns: {id_cols_present}")
        train_df = train_df.drop(columns=id_cols_present)

    # Print final statistics
    print(f"\nCleaning summary:")
    print(f"  Original rows: {original_size}")
    print(f"  Final rows: {len(train_df)}")
    print(f"  Rows removed: {original_size - len(train_df)}")
    print(f"  Final shape: {train_df.shape}")
    print(f"  Remaining missing values: {train_df.isnull().sum().sum()}")

    # Save cleaned data to clean_data_db database (separate database)
    print(f"\nSaving cleaned training data to clean_data_db database...")

    # Use a different PostgresHook for the clean data database
    postgres_clean_hook = PostgresHook(postgres_conn_id='clean_data')

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
    for col, dtype in train_df.dtypes.items():
        pg_type = type_mapping.get(str(dtype), 'TEXT')
        columns_def.append(f'"{col}" {pg_type}')

    create_table_sql = f"""
    DROP TABLE IF EXISTS train_clean CASCADE;
    CREATE TABLE train_clean (
        {', '.join(columns_def)}
    );
    """

    print(f"Creating train_clean table in clean_data_db...")
    postgres_clean_hook.run(create_table_sql)

    # Insert data in batches
    print(f"Inserting cleaned data into train_clean table...")
    batch_size = 1000
    records = [tuple(row) for row in train_df.values]

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        postgres_clean_hook.insert_rows('train_clean', batch, target_fields=train_df.columns.tolist())

    print(f"Successfully stored {len(train_df)} rows in train_clean table (clean_data_db)")

    # Push to XCom
    ti = context['ti']
    ti.xcom_push(key='clean_train_table', value='train_clean')
    ti.xcom_push(key='clean_train_rows', value=len(train_df))

    return {
        'clean_train_table': 'train_clean',
        'clean_train_rows': len(train_df),
        'original_rows': original_size,
        'rows_removed': original_size - len(train_df)
    }

