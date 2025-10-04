from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from airflow import DAG
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator, get_current_context, ShortCircuitOperator

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
import requests
import json
import os
import mlflow

import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

# ---------- Config ----------
API_URI = os.getenv("AIRFLOW_CONN_API_URI", "http://10.43.100.89:8080/data?group_number=2")
MLFLOW = os.getenv("AIRFLOW_CONN_MLFLOW", "http://mlflow:5000")
DEPLOY_TO_PRODUCTION = os.getenv("AIRFLOW_DEPLOY_TO_PRODUCTION", "FALSE")

# =========================
# 1) LOAD RAW TO MySQL
# =========================


def load_covertype_raw():
    r = requests.get(API_URI)
    d = json.loads(r.content.decode('utf-8'))

    batch_number = d['batch_number']

    # Save batch_number to XCom for downstream tasks
    context = get_current_context()
    context['ti'].xcom_push(key='batch_number', value=batch_number)

    covertype_data = d['data']  # Assuming the API returns data in this structure

    NA_SET = {"", "na", "n/a", "null", "none", "nan"}

    def to_int(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in NA_SET:
            return None
        try:
            # Convert to float first in case of decimal strings, then to int
            return int(float(s))
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
        horizontal_distance_to_hydrology = to_int(
            row_data[3] if len(row_data) > 3 else None)
        vertical_distance_to_hydrology = to_int(
            row_data[4] if len(row_data) > 4 else None)
        horizontal_distance_to_roadways = to_int(
            row_data[5] if len(row_data) > 5 else None)
        hillshade_9am = to_int(row_data[6] if len(row_data) > 6 else None)
        hillshade_noon = to_int(row_data[7] if len(row_data) > 7 else None)
        hillshade_3pm = to_int(row_data[8] if len(row_data) > 8 else None)
        horizontal_distance_to_fire_points = to_int(
            row_data[9] if len(row_data) > 9 else None)

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
# 2) SHORT-CIRCUIT CHECK
# =========================
def check_batch_number(**context):
    """
    Check if batch number is less than 10. 
    If batch_number >= 10, stop the DAG by returning False
    """
    # Get batch_number from XCom
    batch_number = context['ti'].xcom_pull(key='batch_number', task_ids='load_covertype_data_to_raw')
    
    print(f"[BATCH_CHECK] Current batch number: {batch_number}")
    
    # Return True to continue pipeline if batch_number < 10, False to stop
    if batch_number >= 10:
        print(f"[BATCH_CHECK] Batch number {batch_number} reached limit (10). Stopping DAG execution.")
        return False
    else:
        print(f"[BATCH_CHECK] Batch number {batch_number} is below limit. Continuing pipeline.")
        return True

# =========================
# 3) CLEAN DATA
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
            print(
                f"[PREPROCESS] Filled {df[col].isnull().sum()} missing values in {col} with median: {median_val}")

    # 2. Handle missing values in categorical columns
    categorical_cols = ['wilderness_area', 'soil_type']
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode(
            ).iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print(
                f"[PREPROCESS] Filled {df[col].isnull().sum()} missing values in {col} with mode: {mode_val}")

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
            print(
                f"[PREPROCESS] Removed {outliers_removed} outliers from {col}")

    print(
        f"[PREPROCESS] Outlier removal: {initial_rows} -> {len(df)} rows ({initial_rows - len(df)} removed)")

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
# 4) Tranform & Train Models with MLflow
# =========================
def train_covertype_models(ti):
    """
    Train multiple ML models on covertype clean data and log results to MLflow.
    Only registers the best performing model per batch to the model registry.
    """    
    # Read batch_number from context (Airflow XCom)
    context = get_current_context()
    batch_number = context['ti'].xcom_pull(key='batch_number', task_ids='load_covertype_data_to_raw')
    print(f"[INFO] batch_number from context: {batch_number}")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW)  # Docker service name
    mlflow.set_experiment("covertype_classification")
    
    # Load clean data from database
    hook = MySqlHook(mysql_conn_id="mysql_trn")
    
    with hook.get_conn() as conn:
        df = pd.read_sql("""
            SELECT elevation, aspect, slope, horizontal_distance_to_hydrology,
                   vertical_distance_to_hydrology, horizontal_distance_to_roadways,
                   hillshade_9am, hillshade_noon, hillshade_3pm,
                   horizontal_distance_to_fire_points, wilderness_area, 
                   soil_type, cover_type
            FROM covertype_clean;
        """, conn)
    
    print(f"[TRAIN] Loaded {len(df)} samples from covertype_clean")
    
    # Prepare features and target
    X = df.drop(columns=['cover_type'])
    y = df['cover_type']
    
    print(f"[TRAIN] Features shape: {X.shape}")
    print(f"[TRAIN] Target distribution:\n{y.value_counts().sort_index()}")
    
    # Train-test split
    try:
        # Try stratified split first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback to random split if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    
    # Create preprocessor
    numeric_features = ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology',
                       'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways',
                       'hillshade_9am', 'hillshade_noon', 'hillshade_3pm',
                       'horizontal_distance_to_fire_points']
    
    categorical_features = ['wilderness_area', 'soil_type']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Define models to train
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "svc": SVC(kernel='rbf', probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "adaboost": AdaBoostClassifier(n_estimators=100, random_state=42),
    }
    
    metrics = []
    best_name, best_f1, best_run_id = None, -1.0, None
    
    # Train and log each model (without registering)
    for name, base_model in models.items():
        print(f"[TRAIN] Training {name}...")
        
        # Create full pipeline
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", base_model)
        ])
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"bn{batch_number}_{name}") as run:
            # Get current run ID
            current_run_id = run.info.run_id
            
            # Train model
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            
            # Log basic parameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("batch_number", batch_number)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_classes", len(np.unique(y)))
            
            # Log model hyperparameters
            try:
                params = base_model.get_params()
                clean_params = {k: v for k, v in params.items() 
                              if isinstance(v, (int, float, str, bool, type(None)))}
                mlflow.log_params(clean_params)
            except Exception as e:
                print(f"[TRAIN] Warning: Could not log params for {name}: {e}")
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)
            
            # Log detailed classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            for label, metrics_dict in report.items():
                if isinstance(metrics_dict, dict):
                    for metric_name, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{label}_{metric_name}", value)
            
            # Log model to MLflow (without registering to model registry)
            try:
                mlflow.sklearn.log_model(
                    clf, 
                    artifact_path="model"
                    # Note: removed registered_model_name parameter
                )
                print(f"[TRAIN] Model {name} logged to MLflow successfully")
            except Exception as e:
                print(f"[TRAIN] Warning: Could not log model {name}: {e}")
            
            # Store metrics for comparison
            metrics.append({
                "model": name, 
                "accuracy": acc, 
                "f1_macro": f1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "run_id": current_run_id
            })
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
                best_run_id = current_run_id
            
            print(f"[TRAIN] {name} - Accuracy: {acc:.4f}, F1-macro: {f1:.4f}, Run ID: {current_run_id}")
    
    model_version_number = None

    # Register only the best model to the model registry
    if best_run_id is not None:
        try:
            model_version = mlflow.register_model(
                model_uri=f"runs:/{best_run_id}/model",
                name="CovertypeClassifier",
                tags={
                    "batch_number": str(batch_number),
                    "training_size": str(len(X_train)),
                    "best_model": best_name,
                    "f1_macro_score": str(best_f1),
                    "training_date": context['ds']
                }
            )
            model_version_number = model_version.version
            print(f"[REGISTRY] Best model '{best_name}' registered as version {model_version.version} (Run ID: {best_run_id})")
        except Exception as e:
            print(f"[REGISTRY] Warning: Could not register best model: {e}")
    else:
        print("[REGISTRY] Warning: No best model found to register")
    
    # Create metrics summary
    metrics_df = pd.DataFrame(metrics).sort_values("f1_macro", ascending=False)
    
    # Log summary metrics to MLflow
    with mlflow.start_run(run_name=f"bn{batch_number}_model_comparison"):
        mlflow.log_param("experiment_type", "model_comparison")
        mlflow.log_param("batch_number", batch_number)
        mlflow.log_param("best_model", best_name)
        mlflow.log_param("best_run_id", best_run_id)
        mlflow.log_metric("best_f1_macro", best_f1)
        
        # Create and log metrics table as artifact
        metrics_csv = "/tmp/covertype_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        mlflow.log_artifact(metrics_csv, artifact_path="results")

    # Print results
    print("\n[METRICS] Model Performance Summary:")
    print(metrics_df.to_string(index=False))
    print(f"\n[BEST] {best_name} (f1_macro={best_f1:.4f}) registered to model registry")
    print(f"[BEST] Run ID: {best_run_id}")

    ti.xcom_push(key="model_version", value=model_version_number)
    
    return {
        "best_model": best_name,
        "best_f1_score": best_f1,
        "best_run_id": best_run_id,
        "total_models_trained": len(models),
        "metrics": metrics_df.to_dict('records')
    }


def deploy_to_production(ti):
    if DEPLOY_TO_PRODUCTION == 'FALSE':
        return True
    
    mlflow.set_tracking_uri(MLFLOW)
    client = MlflowClient()

    try:
        client.transition_model_version_stage(
            name = "CovertypeClassifier",
            version = ti.xcom_pull(task_ids="train_covertype_models", key="model_version"),
            stage="Production",
            archive_existing_versions=True
        )
    except Exception as e:
        print(f"[TRANSITION_MODEL] Warning: Could not transition model: {e}")

# =========================
# DAG
# =========================
with DAG(
    dag_id="covertype_mysql_mlflow_train_models",
    start_date=datetime(2024, 1, 1),
    schedule=timedelta(seconds=305),
    catchup=False,
    max_active_runs=1,
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

    # Check batch number and stop DAG if >= 10
    check_batch = ShortCircuitOperator(
        task_id="check_batch_number",
        python_callable=check_batch_number
    )

    # truncate clean table before loading new data
    truncate_clean = MySqlOperator(
        task_id="truncate_covertype_clean",
        mysql_conn_id="mysql_trn",
        sql="TRUNCATE TABLE covertype_clean;"
    )

    clean_data = PythonOperator(
        task_id="clean_covertype_data",
        python_callable=clean_covertype
    )

    train_models = PythonOperator(
        task_id="train_covertype_models",
        python_callable=train_covertype_models
    )

    deploy_to_production = PythonOperator(
        task_id="deploy_to_production",
        python_callable=deploy_to_production
    )

    create_tables >> load_raw >> check_batch >> truncate_clean >> clean_data >> train_models >> deploy_to_production
