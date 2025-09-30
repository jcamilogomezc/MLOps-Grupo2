from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from airflow import DAG
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
import joblib
import requests
import json
import os

# ---------- Config ----------
API_URI = os.getenv("AIRFLOW_CONN_API_URI", "http://10.43.100.89:8080/data?group_number=3")

# =========================
# 1) LOAD RAW TO MySQL
# =========================
def load_covertype_raw():
    r = requests.get(API_URI)
    d = json.loads(r.content.decode('utf-8'))
    covertype_data = d['data']  # Assuming the API returns data in this structure

    NA_SET = {"", "na", "n/a", "null", "none", "nan"}

    def to_int(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in NA_SET:
            return None
        try:
            return int(float(s))  # Convert to float first in case of decimal strings, then to int
        except (ValueError, TypeError):
            return None

    def to_text(x):
        if x is None:
            return None
        s = str(x).strip()
        return s if s and s.lower() not in NA_SET else None

    hook = MySqlHook(mysql_conn_id="mysql_trn")

    rows = []
    total = 0

    for row_data in covertype_data:
        total += 1
        
        # Extract all numeric fields
        elevation = to_int(row_data[0] if len(row_data) > 0 else None)
        aspect = to_int(row_data[1] if len(row_data) > 1 else None)
        slope = to_int(row_data[2] if len(row_data) > 2 else None)
        horizontal_distance_to_hydrology = to_int(row_data[3] if len(row_data) > 3 else None)
        vertical_distance_to_hydrology = to_int(row_data[4] if len(row_data) > 4 else None)
        horizontal_distance_to_roadways = to_int(row_data[5] if len(row_data) > 5 else None)
        hillshade_9am = to_int(row_data[6] if len(row_data) > 6 else None)
        hillshade_noon = to_int(row_data[7] if len(row_data) > 7 else None)
        hillshade_3pm = to_int(row_data[8] if len(row_data) > 8 else None)
        horizontal_distance_to_fire_points = to_int(row_data[9] if len(row_data) > 9 else None)
        
        # Extract categorical fields
        wilderness_area = to_text(row_data[10] if len(row_data) > 10 else None)
        soil_type = to_text(row_data[11] if len(row_data) > 11 else None)
        cover_type = to_int(row_data[12] if len(row_data) > 12 else None)

        rows.append((
            elevation, aspect, slope, horizontal_distance_to_hydrology,
            vertical_distance_to_hydrology, horizontal_distance_to_roadways,
            hillshade_9am, hillshade_noon, hillshade_3pm,
            horizontal_distance_to_fire_points, wilderness_area, soil_type, cover_type
        ))

    if rows:
        hook.insert_rows(
            table="covertype_raw",
            rows=rows,
            target_fields=[
                "elevation", "aspect", "slope", "horizontal_distance_to_hydrology",
                "vertical_distance_to_hydrology", "horizontal_distance_to_roadways",
                "hillshade_9am", "hillshade_noon", "hillshade_3pm",
                "horizontal_distance_to_fire_points", "wilderness_area", 
                "soil_type", "cover_type"
            ],
        )

    print(f"[LOAD] Total={total} | Insertadas={len(rows)}")

# =========================
# 2) CLEAN DATA
# =========================
def clean_covertype():
    sql_hook = MySqlHook(mysql_conn_id="mysql_trn")

    # Read RAW data
    with sql_hook.get_conn() as conn:
        df = pd.read_sql("""
            SELECT elevation, aspect, slope, horizontal_distance_to_hydrology,
                   vertical_distance_to_hydrology, horizontal_distance_to_roadways,
                   hillshade_9am, hillshade_noon, hillshade_3pm,
                   horizontal_distance_to_fire_points, wilderness_area, soil_type, cover_type
            FROM covertype_raw;
        """, conn)

    print(f"[PREPROCESS] Raw data loaded: {len(df)} rows")
    
    # Data Cleaning Steps
    
    # 1. Handle missing values in numeric columns
    numeric_cols = ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology',
                    'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways',
                    'hillshade_9am', 'hillshade_noon', 'hillshade_3pm',
                    'horizontal_distance_to_fire_points', 'cover_type']
    
    # Fill missing numeric values with median (robust to outliers)
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[PREPROCESS] Filled {df[col].isnull().sum()} missing values in {col} with median: {median_val}")
    
    # 2. Handle missing values in categorical columns
    categorical_cols = ['wilderness_area', 'soil_type']
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"[PREPROCESS] Filled {df[col].isnull().sum()} missing values in {col} with mode: {mode_val}")
    
    # 3. Remove extreme outliers using IQR method for key numeric features
    outlier_cols = ['elevation', 'slope', 'horizontal_distance_to_hydrology', 
                    'horizontal_distance_to_roadways', 'horizontal_distance_to_fire_points']
    
    initial_rows = len(df)
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # Using 3*IQR for more conservative outlier removal
        upper_bound = Q3 + 3 * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers_removed = outlier_mask.sum()
        
        if outliers_removed > 0:
            df = df[~outlier_mask].copy()
            print(f"[PREPROCESS] Removed {outliers_removed} outliers from {col}")
    
    print(f"[PREPROCESS] Outlier removal: {initial_rows} -> {len(df)} rows ({initial_rows - len(df)} removed)")
    
    # 4. Validate data ranges and fix inconsistencies
    # Ensure aspect is between 0-360 degrees
    df['aspect'] = df['aspect'].clip(0, 360)
    
    # Ensure slope is non-negative
    df['slope'] = df['slope'].clip(0, None)
    
    # Ensure hillshade values are between 0-255
    hillshade_cols = ['hillshade_9am', 'hillshade_noon', 'hillshade_3pm']
    for col in hillshade_cols:
        df[col] = df[col].clip(0, 255)
    
    # Ensure cover_type is within valid range (typically 1-7 for forest cover types)
    df['cover_type'] = df['cover_type'].clip(0, 7)
    
    # 5. Standardize categorical values
    # Clean wilderness_area names
    df['wilderness_area'] = df['wilderness_area'].str.strip().str.title()
    
    # Clean soil_type codes
    df['soil_type'] = df['soil_type'].str.strip().str.upper()
    
    # Final data quality check
    print(f"[PREPROCESS] Final data quality summary:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    print(f"  - Duplicate rows: {df.duplicated().sum()}")
    
    # Remove exact duplicates if any
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[PREPROCESS] Removed {df.duplicated().sum()} duplicate rows")
    
    # Select final columns for clean table (matching the schema)
    final_cols = ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology',
                  'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways',
                  'hillshade_9am', 'hillshade_noon', 'hillshade_3pm',
                  'horizontal_distance_to_fire_points', 'wilderness_area', 'soil_type', 'cover_type']
    
    df_clean = df[final_cols].copy()
    
    # Convert to tuples for insertion
    rows = list(df_clean.itertuples(index=False, name=None))
    
    # Insert cleaned data
    sql_hook.insert_rows(
        table='covertype_clean',
        rows=rows,
        target_fields=[
            'elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology',
            'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways',
            'hillshade_9am', 'hillshade_noon', 'hillshade_3pm',
            'horizontal_distance_to_fire_points', 'wilderness_area', 'soil_type', 'cover_type'
        ]
    )
    
    print(f"[PREPROCESS] Clean rows inserted: {len(rows)}")
    return f"Data cleaning completed. {len(rows)} clean records processed."

# =========================
# 3) Tranform & Train Models with MLflow
# =========================


# =========================
# DAG
# =========================
with DAG(
    dag_id="covertype_mysql_mlflow_train_models",
    start_date=datetime(2024,1,1),
    schedule=None,
    catchup=False,
    tags=["mysql","etl","mlops_puj","training"]
) as dag:

    # Esquema explícito para clean (incluye columnas *_numeric)
    create_tables = MySqlOperator(
        task_id="create_tables",
        mysql_conn_id="mysql_trn",
        sql="""
        CREATE TABLE IF NOT EXISTS covertype_raw (
            id INT AUTO_INCREMENT PRIMARY KEY,
            elevation INT,
            aspect INT,
            slope INT,
            horizontal_distance_to_hydrology INT,
            vertical_distance_to_hydrology INT,
            horizontal_distance_to_roadways INT,
            hillshade_9am INT,
            hillshade_noon INT,
            hillshade_3pm INT,
            horizontal_distance_to_fire_points INT,
            wilderness_area VARCHAR(32),
            soil_type VARCHAR(16),
            cover_type TINYINT,
            inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS covertype_clean (
            id INT AUTO_INCREMENT PRIMARY KEY,
            elevation DECIMAL(10,4),
            aspect DECIMAL(10,4),
            slope DECIMAL(10,4),
            horizontal_distance_to_hydrology DECIMAL(10,4),
            vertical_distance_to_hydrology DECIMAL(10,4),
            horizontal_distance_to_roadways DECIMAL(10,4),
            hillshade_9am DECIMAL(10,4),
            hillshade_noon DECIMAL(10,4),
            hillshade_3pm DECIMAL(10,4),
            horizontal_distance_to_fire_points DECIMAL(10,4),
            wilderness_area VARCHAR(32),
            soil_type VARCHAR(16),
            cover_type TINYINT,
            inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    load_raw = PythonOperator(
        task_id="load_covertype_data_to_raw",
        python_callable=load_covertype_raw
    )

    clean_data = PythonOperator(
        task_id="clean_covertype_data",
        python_callable=clean_covertype
    )

    create_tables >> load_raw >> clean_data