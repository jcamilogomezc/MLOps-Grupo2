"""
DAG 7: Select Best Model
This DAG evaluates all trained models on the validation dataset and selects the best one.
"""

from datetime import datetime, timedelta
from airflow.providers.postgres.hooks.postgres import PostgresHook
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn


# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def prepare_validation_data(val_df, target_col='readmitted'):
    """
    Prepare validation data by encoding categorical variables.
    """
    X = val_df.drop(columns=[target_col])
    y = val_df[target_col]

    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Encode categorical features
    X_encoded = X.copy()
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    return X_encoded, y_encoded


def select_best_model(**context):
    """
    Evaluate all trained models on validation data and select the best one.
    Uses PostgresHook to read from raw_data_db and clean_data_db.
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    mlflow.set_experiment("diabetes_readmission_prediction")

    # Use PostgresHook for raw and clean data databases
    raw_postgres_hook = PostgresHook(postgres_conn_id='raw_data')
    clean_postgres_hook = PostgresHook(postgres_conn_id='clean_data')

    print("Reading validation and training data...")

    # Read raw validation data
    raw_conn = raw_postgres_hook.get_conn()
    val_df_raw = pd.read_sql("SELECT * FROM validation_raw", raw_conn)
    raw_conn.close()

    # Read clean training data (to get feature names)
    clean_conn = clean_postgres_hook.get_conn()
    train_df_clean = pd.read_sql("SELECT * FROM train_clean", clean_conn)
    clean_conn.close()

    print(f"Raw validation data shape: {val_df_raw.shape}")
    print(f"Clean training data shape: {train_df_clean.shape}")

    # Clean validation data using same logic as training data
    # (Remove same columns that were removed from training)
    val_df = val_df_raw.copy()

    # Get feature columns from training data
    train_features = [col for col in train_df_clean.columns if col != 'readmitted']

    # Keep only common features in validation data
    val_features = [col for col in val_df.columns if col in train_features or col == 'readmitted']
    val_df = val_df[val_features]

    print(f"Cleaned validation data shape: {val_df.shape}")

    # Prepare validation data
    X_val, y_val = prepare_validation_data(val_df)

    # Scale features
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val)

    print(f"Validation data prepared: {X_val_scaled.shape}")

    # Get all runs from the experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("diabetes_readmission_prediction")

    if experiment is None:
        raise ValueError("Experiment 'diabetes_readmission_prediction' not found!")

    runs = client.search_runs(experiment.experiment_id)

    print(f"\nFound {len(runs)} model runs. Evaluating on validation data...")

    # Evaluate each model
    model_results = {}
    best_model_info = {'f1_score': 0, 'run_id': None, 'model_name': None}

    for run in runs:
        run_id = run.info.run_id
        model_name = run.data.params.get('model_type', 'unknown')

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} (run_id: {run_id[:8]}...)")
        print(f"{'='*60}")

        try:
            # Load model
            model_uri = f"runs:/{run_id}/{model_name}"
            model = mlflow.sklearn.load_model(model_uri)

            # Make predictions
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            f1 = f1_score(y_val, y_pred, average='weighted')

            try:
                roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                roc_auc = 0.0

            # Log validation metrics to MLflow
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("val_accuracy", accuracy)
                mlflow.log_metric("val_precision", precision)
                mlflow.log_metric("val_recall", recall)
                mlflow.log_metric("val_f1_score", f1)
                mlflow.log_metric("val_roc_auc", roc_auc)

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")

            model_results[model_name] = {
                'run_id': run_id,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }

            # Check if this is the best model
            if f1 > best_model_info['f1_score']:
                best_model_info = {
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'run_id': run_id,
                    'model_name': model_name
                }

        except Exception as e:
            print(f"  Error evaluating model: {e}")

    # Print best model
    print(f"\n{'='*60}")
    print(f"BEST MODEL SELECTED")
    print(f"{'='*60}")
    print(f"Model: {best_model_info['model_name']}")
    print(f"Run ID: {best_model_info['run_id']}")
    print(f"F1 Score: {best_model_info['f1_score']:.4f}")
    print(f"Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"{'='*60}")

    # Push to XCom
    ti = context['ti']
    ti.xcom_push(key='best_model_run_id', value=best_model_info['run_id'])
    ti.xcom_push(key='best_model_name', value=best_model_info['model_name'])
    ti.xcom_push(key='best_model_f1', value=best_model_info['f1_score'])

    return best_model_info
