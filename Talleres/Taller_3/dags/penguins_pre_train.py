from datetime import datetime
import csv
import palmerpenguins
import pandas as pd
import numpy as np
from airflow import DAG
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path

def load_penguins_data_to_raw():
    penguins = palmerpenguins.load_penguins()
    
    NA_SET = {"", "na", "n/a", "null", "none", "nan"}
    def to_float(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in NA_SET:
            return None
        # Soportar coma decimal "43,2"
        s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return None

    def to_text(x):
        if x is None:
            return None
        s = str(x).strip()
        return s if s and s.lower() not in NA_SET else None

    hook = MySqlHook(mysql_conn_id="mymysql")

    rows = []
    skipped = 0
    total = 0

    for index, row in penguins.iterrows():
        total += 1
        species  = to_text(row.get("species"))
        island   = to_text(row.get("island"))
        bill_len = to_float(row.get("bill_length_mm"))
        bill_dep = to_float(row.get("bill_depth_mm"))
        flip_len = to_float(row.get("flipper_length_mm"))
        body_g   = to_float(row.get("body_mass_g"))
        sex      = to_text(row.get("sex"))

        # Opcional: descartar filas vacías (sin ninguna métrica)
        if all(v is None for v in (bill_len, bill_dep, flip_len, body_g)):
            skipped += 1
            continue

        rows.append((species, island, bill_len, bill_dep, flip_len, body_g, sex))

    if rows:
        hook.insert_rows(
            table="penguins_raw",
            rows=rows,
            target_fields=[
                "species","island","bill_length_mm","bill_depth_mm",
                "flipper_length_mm","body_mass_g","sex"
            ],
        )

    print(f"[LOAD] Total filas leídas: {total} | Insertadas: {len(rows)} | Omitidas: {skipped}")

def preprocess():

    sql_hook = MySqlHook(mysql_conn_id="mymysql")

    def get_database_data():
        sql_query = "SELECT species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex FROM penguins_raw;"

        connection = sql_hook.get_conn()
        cursor = connection.cursor()
        cursor.execute(sql_query)

        records = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        cursor.close()
        connection.close()

        df = pd.DataFrame(records, columns=column_names)

        return df
    
    def clean(df):
        # Remove rows with missing species
        df_clean = df.dropna(subset=['species'])
        print(f"Removed {len(df) - len(df_clean)} rows with missing species")
        
        # Define column types
        numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        categorical_cols = ['island', 'sex']
        
        # Fill missing numeric values with median
        numeric_imputer = SimpleImputer(strategy='median')
        df_clean[numeric_cols] = numeric_imputer.fit_transform(df_clean[numeric_cols])
        
        # Fill missing categorical values with most frequent
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_clean[categorical_cols] = categorical_imputer.fit_transform(df_clean[categorical_cols])

        return df_clean
    
    def encode_features(df):
        species_encoder = LabelEncoder()
        df['species_numeric'] = species_encoder.fit_transform(df['species'])
        
        island_encoder = LabelEncoder()  
        df['island_numeric'] = island_encoder.fit_transform(df['island'])
        
        sex_encoder = LabelEncoder()
        df['sex_numeric'] = sex_encoder.fit_transform(df['sex'])
        
        return df

    def scale_features(df):
        scale_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                  'body_mass_g']
    
        # Scale features
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    def store_data(df):
        data_to_insert = []
        target_columns = [
            'bill_depth_mm',
            'flipper_length_mm',
            'body_mass_g',
            'island',
            'sex',
            'species_numeric',
            'island_numeric',
            'sex_numeric'
        ]
        for index, row in df.iterrows():
            data_to_insert.append((
                row.get('bill_length_mm'), 
                row.get('bill_depth_mm'), 
                row.get('flipper_length_mm'), 
                row.get('body_mass_g'),
                row.get('island'),
                row.get('sex'),
                row.get('species_numeric'),
                row.get('island_numeric'),
                row.get('sex_numeric')
            ))

        mysql_hook.insert_rows(
            table='penguins_clean',
            rows=data_to_insert,
            target_fields=target_columns
        )        

    df = get_database_data()
    df = clean(df)
    df = encode_features(df)
    df = scale_features(df)

def count_tables():
    hook = MySqlHook(mysql_conn_id="mymysql")
    raw  = hook.get_first("SELECT COUNT(*) FROM penguins_raw")[0]
    clean= hook.get_first("SELECT COUNT(*) FROM penguins_clean")[0]
    print(f"[CHECK] RAW={raw} | CLEAN={clean}")

def train():
    # Get clean data from MySQL database
    sql_hook = MySqlHook(mysql_conn_id="mymysql")
    sql_query = "SELECT * FROM penguins_clean;"


    connection = sql_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute(sql_query)

    records = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    cursor.close()
    connection.close()

    df = pd.DataFrame(records, columns=column_names)

    print("This is the data: ", df)

with DAG(
    dag_id="penguins_mysql_etl",
    start_date=datetime(2024,1,1),
    schedule_interval=None,
    catchup=False,
    tags=["mysql","etl","mlops_puj"]
) as dag:

    create_tables = MySqlOperator(
        task_id="create_tables",
        mysql_conn_id="mymysql",
        sql="""
        CREATE TABLE IF NOT EXISTS penguins_raw (
          id INT AUTO_INCREMENT PRIMARY KEY,
          species VARCHAR(64), island VARCHAR(64),
          bill_length_mm DECIMAL(10,4), bill_depth_mm DECIMAL(10,4),
          flipper_length_mm DECIMAL(10,4), body_mass_g DECIMAL(10,4),
          sex VARCHAR(16), inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS penguins_clean LIKE penguins_raw;
        """
    )

    truncate_raw = MySqlOperator(
        task_id="truncate_raw",
        mysql_conn_id="mymysql",
        sql="TRUNCATE TABLE penguins_raw;"
    )

    load_raw = PythonOperator(
        task_id="load_penguins_data_to_raw",
        python_callable=load_penguins_data_to_raw
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess
    )

    # preprocess = MySqlOperator(
    #     task_id="preprocess_clean",
    #     mysql_conn_id="mymysql",
    #     sql="""
    #     DELETE FROM penguins_clean;
    #     INSERT INTO penguins_clean (species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
    #     SELECT species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,
    #            CASE LOWER(TRIM(COALESCE(sex,'')))
    #                 WHEN 'male' THEN 'male'
    #                 WHEN 'female' THEN 'female'
    #                 ELSE NULL END AS sex_norm
    #     FROM penguins_raw
    #     WHERE bill_length_mm IS NOT NULL
    #       AND bill_depth_mm  IS NOT NULL
    #       AND flipper_length_mm IS NOT NULL
    #       AND body_mass_g    IS NOT NULL;
    #     """
    # )

    check_counts = PythonOperator(
        task_id="check_counts",
        python_callable=count_tables
    )

    train_models = PythonOperator(
        task_id="train",
        python_callable=train
    )

    create_tables >> truncate_raw >> load_raw >> preprocess >> check_counts >> train_models