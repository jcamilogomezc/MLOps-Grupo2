"""
DAG 06b: Select Best Model from Cumulative Training
This DAG evaluates all models trained on cumulative batches and selects the best one.

Total models evaluated: 7 cumulative datasets × 4 model types = 28 models

Selection criteria:
1. Primary: Highest F1 score on validation set
2. Secondary: ROC-AUC score as tiebreaker
3. Consider: Balance between performance and training data size
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import numpy as np
import os
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import warnings

warnings.filterwarnings('ignore')


# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def aggregate_cumulative_results(**context):
    """
    Aggregate all training results from cumulative batch training.
    Retrieves results from XCom pushed by train_cumulative_batch tasks.
    Dynamically determines which cumulative indices to check based on training configs.
    """
    print(f"\n{'='*70}")
    print("AGGREGATING CUMULATIVE TRAINING RESULTS")
    print(f"{'='*70}\n")

    all_results = []

    # Try to pull training configurations from generate_training_configs task
    training_configs = context['ti'].xcom_pull(
        task_ids='generate_training_configs',
        key='training_configs'
    )

    # Fallback to hardcoded configs if not found (for standalone execution)
    if not training_configs:
        print("⚠ Warning: Training configs not found, using fallback hardcoded configs")
        cumulative_configs = [
            {'cumulative_idx': 0, 'expected_records': 15000},
            {'cumulative_idx': 1, 'expected_records': 30000},
            {'cumulative_idx': 2, 'expected_records': 45000},
            {'cumulative_idx': 3, 'expected_records': 60000},
            {'cumulative_idx': 4, 'expected_records': 75000},
            {'cumulative_idx': 5, 'expected_records': 90000},
            {'cumulative_idx': 6, 'expected_records': 101767}
        ]
    else:
        cumulative_configs = training_configs
        print(f"✓ Found {len(cumulative_configs)} training configurations")

    # Pull results from each cumulative training task
    for config in cumulative_configs:
        cumulative_idx = config['cumulative_idx']
        expected_records = config['expected_records']

        # Try multiple task_id formats for backward compatibility
        task_ids_to_try = [
            f'train_cumulative_{cumulative_idx}',  # New format
            f'train_cumulative_{cumulative_idx}_{expected_records}records'  # Old format
        ]

        results = None
        for task_id in task_ids_to_try:
            # Try to pull from the same DAG (for dag_00b_master_cumulative_pipeline)
            results = context['ti'].xcom_pull(
                task_ids=task_id,
                key=f'cumulative_{cumulative_idx}_results'
            )

            if results:
                break

            # If not found, try from dag_05b_train_cumulative_batches
            results = context['ti'].xcom_pull(
                dag_id='dag_05b_train_cumulative_batches',
                task_ids=task_id,
                key=f'cumulative_{cumulative_idx}_results'
            )

            if results:
                break

        if results:
            all_results.extend(results)
            print(f"✓ Cumulative {cumulative_idx} ({expected_records} records): Retrieved {len(results)} model results")
        else:
            print(f"⚠ Cumulative {cumulative_idx} ({expected_records} records): No results found (may not have completed)")

    if not all_results:
        raise ValueError("No training results found! Ensure cumulative training tasks have completed successfully.")

    print(f"\n{'*'*70}")
    print(f"✓ Total models aggregated: {len(all_results)}")
    print(f"{'*'*70}\n")

    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(all_results)

    # Sort by validation F1 score
    df_results = df_results.sort_values('val_f1_score', ascending=False)

    print(f"\nTop 10 Models by Validation F1 Score:")
    print(df_results[['cumulative_idx', 'num_batches', 'total_records', 'model_name', 'val_f1_score', 'val_accuracy']].head(10).to_string(index=False))

    # Push aggregated results to XCom
    context['ti'].xcom_push(key='all_results', value=all_results)
    context['ti'].xcom_push(key='num_models', value=len(all_results))

    return df_results.to_dict('records')


def select_best_model(**context):
    """
    Select the best model based on validation metrics.
    Primary criterion: Highest F1 score
    Secondary criterion: ROC-AUC as tiebreaker
    """
    print(f"\n{'='*70}")
    print("SELECTING BEST MODEL")
    print(f"{'='*70}\n")

    # Retrieve aggregated results
    all_results = context['ti'].xcom_pull(task_ids='aggregate_results', key='all_results')

    if not all_results:
        raise ValueError("No aggregated results found!")

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)

    # Sort by F1 score (primary) and ROC-AUC (secondary)
    df_results = df_results.sort_values(['val_f1_score', 'val_roc_auc'], ascending=[False, False])

    # Select best model
    best_model = df_results.iloc[0]

    print(f"{'*'*70}")
    print("BEST MODEL SELECTED")
    print(f"{'*'*70}")
    print(f"  Model Type: {best_model['model_name']}")
    print(f"  Cumulative Dataset: {best_model['cumulative_idx']} (batches 0-{best_model['cumulative_idx']})")
    print(f"  Total Training Records: {best_model['total_records']}")
    print(f"  Number of Batches: {best_model['num_batches']}")
    print(f"\n  Validation Metrics:")
    print(f"    F1 Score:  {best_model['val_f1_score']:.4f}")
    print(f"    Accuracy:  {best_model['val_accuracy']:.4f}")
    print(f"    ROC-AUC:   {best_model['val_roc_auc']:.4f}")
    print(f"\n  Training Metrics:")
    print(f"    F1 Score:  {best_model['train_f1_score']:.4f}")
    print(f"    Accuracy:  {best_model['train_accuracy']:.4f}")
    print(f"\n  MLflow Run ID: {best_model['run_id']}")
    print(f"{'*'*70}\n")

    # Analysis: Compare performance across batch sizes
    print("\nPerformance Analysis Across Cumulative Datasets:\n")

    performance_by_cumulative = df_results.groupby('cumulative_idx').agg({
        'val_f1_score': 'max',
        'val_accuracy': 'max',
        'total_records': 'first'
    }).reset_index()

    performance_by_cumulative.columns = ['Cumulative_Idx', 'Best_F1', 'Best_Accuracy', 'Total_Records']

    print(performance_by_cumulative.to_string(index=False))

    # Analysis: Compare model types
    print("\n\nPerformance Analysis by Model Type:\n")

    performance_by_model = df_results.groupby('model_name').agg({
        'val_f1_score': ['max', 'mean', 'std'],
        'val_accuracy': ['max', 'mean', 'std']
    }).round(4)

    print(performance_by_model.to_string())

    # Push best model info to XCom
    best_model_dict = best_model.to_dict()
    context['ti'].xcom_push(key='best_model', value=best_model_dict)
    context['ti'].xcom_push(key='best_run_id', value=best_model['run_id'])

    return best_model_dict


def evaluate_on_full_validation(**context):
    """
    Evaluate the best model on the full validation dataset from PostgreSQL.
    This provides an unbiased evaluation on the complete validation set.
    """
    print(f"\n{'='*70}")
    print("EVALUATING BEST MODEL ON FULL VALIDATION SET")
    print(f"{'='*70}\n")

    # Get best model info
    best_model_info = context['ti'].xcom_pull(task_ids='select_best_model', key='best_model')

    if not best_model_info:
        raise ValueError("Best model information not found!")

    run_id = best_model_info['run_id']
    model_name = best_model_info['model_name']

    print(f"Loading model from MLflow...")
    print(f"  Run ID: {run_id}")
    print(f"  Model Type: {model_name}")

    # Load model from MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    model_uri = f"runs:/{run_id}/model_{model_name}"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

    # Load full validation dataset from PostgreSQL
    print(f"\nLoading full validation dataset from PostgreSQL...")
    postgres_hook = PostgresHook(postgres_conn_id='raw_data')

    select_sql = "SELECT * FROM validation_raw"
    connection = postgres_hook.get_conn()
    df_val = pd.read_sql(select_sql, connection)
    connection.close()

    print(f"Validation dataset loaded: {df_val.shape}")

    # Clean validation data (same as training)
    from dag_05b_train_cumulative_batches import clean_batch_data
    df_val_clean = clean_batch_data(df_val.copy())

    # Separate features and target
    if 'readmitted' not in df_val_clean.columns:
        raise ValueError("Target column 'readmitted' not found in validation dataset")

    X_val = df_val_clean.drop('readmitted', axis=1)
    y_val = df_val_clean['readmitted']

    print(f"\nValidation features shape: {X_val.shape}")
    print(f"Validation target distribution:\n{y_val.value_counts()}")

    # Load expected feature names from MLflow artifacts
    print("\nLoading expected feature names from MLflow...")

    try:
        # Download feature names artifact from MLflow
        client = MlflowClient()
        local_path = client.download_artifacts(run_id, 'feature_names.json')

        import json
        with open(local_path, 'r') as f:
            feature_data = json.load(f)
            expected_features = feature_data['feature_names']

        print(f"  ✓ Loaded {len(expected_features)} expected features from MLflow")

        # Align validation features to match expected features BEFORE encoding
        current_features = set(X_val.columns)
        expected_features_set = set(expected_features)

        # Features in validation but not in training
        extra_features = current_features - expected_features_set
        if extra_features:
            print(f"  Removing {len(extra_features)} extra features from validation: {sorted(extra_features)[:10]}...")
            X_val = X_val.drop(columns=list(extra_features))

        # Features in training but not in validation - add with appropriate fill values
        missing_features = expected_features_set - current_features
        if missing_features:
            print(f"  Adding {len(missing_features)} missing features to validation: {sorted(missing_features)[:10]}...")
            for feat in missing_features:
                # For missing features, use the most common approach based on expected data type
                # Since we don't know the type, we'll use 0 for numeric or 'Unknown' for categorical
                # The encoding step will handle these appropriately
                X_val[feat] = 'Unknown'  # Will be encoded to a numeric value

        # Reorder columns to match expected feature order (before encoding)
        X_val = X_val[expected_features]
        print(f"  ✓ Features aligned successfully: {X_val.shape[1]} features")

    except Exception as e:
        print(f"  ⚠ Could not load or align features: {e}")
        print(f"  This may cause prediction errors if features don't match!")
        print("  Proceeding with available features...")

    # Encode categorical variables
    for col in X_val.columns:
        if X_val[col].dtype == 'object':
            le = LabelEncoder()
            X_val[col] = le.fit_transform(X_val[col].astype(str))

    # Encode target
    target_encoder = LabelEncoder()
    y_val_encoded = target_encoder.fit_transform(y_val)

    # Scale features
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val)

    # Predict
    print(f"\nMaking predictions on full validation set...")
    y_val_pred = model.predict(X_val_scaled)

    # Calculate metrics
    val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
    val_precision = precision_score(y_val_encoded, y_val_pred, average='weighted', zero_division=0)
    val_recall = recall_score(y_val_encoded, y_val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val_encoded, y_val_pred, average='weighted', zero_division=0)

    try:
        if hasattr(model, 'predict_proba'):
            y_val_proba = model.predict_proba(X_val_scaled)
            val_roc_auc = roc_auc_score(y_val_encoded, y_val_proba, multi_class='ovr', average='weighted')
        else:
            val_roc_auc = 0.0
    except:
        val_roc_auc = 0.0

    print(f"\n{'*'*70}")
    print("FULL VALIDATION SET EVALUATION")
    print(f"{'*'*70}")
    print(f"  Model: {model_name}")
    print(f"  Validation Size: {len(X_val)} records")
    print(f"\n  Metrics:")
    print(f"    Accuracy:  {val_accuracy:.4f}")
    print(f"    Precision: {val_precision:.4f}")
    print(f"    Recall:    {val_recall:.4f}")
    print(f"    F1 Score:  {val_f1:.4f}")
    print(f"    ROC-AUC:   {val_roc_auc:.4f}")
    print(f"{'*'*70}\n")

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_val_encoded, y_val_pred, target_names=target_encoder.classes_))

    # Log evaluation metrics to MLflow
    client = MlflowClient()
    client.log_metric(run_id, 'full_val_accuracy', val_accuracy)
    client.log_metric(run_id, 'full_val_precision', val_precision)
    client.log_metric(run_id, 'full_val_recall', val_recall)
    client.log_metric(run_id, 'full_val_f1_score', val_f1)
    client.log_metric(run_id, 'full_val_roc_auc', val_roc_auc)

    # Push evaluation results to XCom
    evaluation_results = {
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1_score': val_f1,
        'roc_auc': val_roc_auc,
        'validation_size': len(X_val)
    }

    context['ti'].xcom_push(key='full_validation_results', value=evaluation_results)

    return evaluation_results
