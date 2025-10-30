"""
DAG 05b: Train Models with Cumulative Batches in Parallel
This DAG trains models on cumulative datasets using PostgreSQL views:
- Cumulative 0: 15,000 records (train_cumulative_0)
- Cumulative 1: 30,000 records (train_cumulative_1)
- Cumulative 2: 45,000 records (train_cumulative_2)
- Cumulative 3: 60,000 records (train_cumulative_3)
- Cumulative 4: 75,000 records (train_cumulative_4)
- Cumulative 5: 90,000 records (train_cumulative_5)
- Cumulative 6: 101,767 records (train_cumulative_6, all data)

Each cumulative dataset trains all 4 model types, and all cumulative trainings run in parallel.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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


def clean_batch_data(df):
    """
    Clean and preprocess batch data.
    Replicates cleaning logic from dag_04_clean_training_data.py
    """
    print(f"Initial shape: {df.shape}")

    # Replace '?' with NaN
    df = df.replace('?', np.nan)

    # Remove columns with >50% missing values
    missing_threshold = 0.5
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    if cols_to_drop:
        print(f"Dropping columns with >50% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Remove constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    # Drop ID columns and batch_id if they exist
    id_columns = ['encounter_id', 'patient_nbr', 'batch_id']
    existing_id_cols = [col for col in id_columns if col in df.columns]
    if existing_id_cols:
        print(f"Dropping ID columns: {existing_id_cols}")
        df = df.drop(columns=existing_id_cols)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            # Fill categorical with mode
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)
        else:
            # Fill numerical with median
            df[col].fillna(df[col].median(), inplace=True)

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_dups = initial_rows - len(df)
    if removed_dups > 0:
        print(f"Removed {removed_dups} duplicate rows")

    print(f"Final cleaned shape: {df.shape}")
    return df


def train_cumulative_batch(cumulative_idx: int, **context):
    """
    Train all model types on cumulative dataset from PostgreSQL view.

    Args:
        cumulative_idx: Index of cumulative view (0-6)
                       0 = 15k, 1 = 30k, 2 = 45k, etc.
    """
    print(f"\n{'='*70}")
    print(f"CUMULATIVE TRAINING - Dataset {cumulative_idx}")
    print(f"{'='*70}\n")

    # MLflow configuration
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    experiment_name = 'diabetes_cumulative_batch_training'
    mlflow.set_experiment(experiment_name)

    # Load cumulative data from PostgreSQL view
    view_name = f'train_cumulative_{cumulative_idx}'
    postgres_hook = PostgresHook(postgres_conn_id='raw_data')

    print(f"Loading data from PostgreSQL view: {view_name}")

    # Get record count
    count_sql = f"SELECT COUNT(*) FROM {view_name}"
    result = postgres_hook.get_first(count_sql)
    total_records = result[0] if result else 0

    print(f"Total records in {view_name}: {total_records}")

    # Load data into pandas DataFrame
    select_sql = f"SELECT * FROM {view_name}"
    connection = postgres_hook.get_conn()
    df_cumulative = pd.read_sql(select_sql, connection)
    connection.close()

    print(f"Loaded DataFrame shape: {df_cumulative.shape}")

    # Clean cumulative data
    print("\nCleaning cumulative data...")
    df_clean = clean_batch_data(df_cumulative.copy())

    # Separate features and target for training data
    if 'readmitted' not in df_clean.columns:
        raise ValueError(f"Target column 'readmitted' not found in {view_name}")

    X_train = df_clean.drop('readmitted', axis=1)
    y_train = df_clean['readmitted']

    print(f"\nTraining Features shape: {X_train.shape}")
    print(f"Training Target distribution:\n{y_train.value_counts()}")

    # Load validation data from validation_raw table
    print("\nLoading validation data from validation_raw table...")
    val_sql = "SELECT * FROM validation_raw"
    connection = postgres_hook.get_conn()
    df_val = pd.read_sql(val_sql, connection)
    connection.close()

    print(f"Validation data loaded: {df_val.shape}")

    # Clean validation data
    df_val_clean = clean_batch_data(df_val.copy())

    # Separate features and target for validation data
    if 'readmitted' not in df_val_clean.columns:
        raise ValueError("Target column 'readmitted' not found in validation_raw")

    X_val = df_val_clean.drop('readmitted', axis=1)
    y_val = df_val_clean['readmitted']

    print(f"\nValidation Features shape: {X_val.shape}")
    print(f"Validation Target distribution:\n{y_val.value_counts()}")

    print(f"\nTraining set: {X_train.shape[0]} records")
    print(f"Validation set: {X_val.shape[0]} records")

    # Align features between training and validation sets
    print("\nAligning features between training and validation sets...")

    # Get the common features
    train_features = set(X_train.columns)
    val_features = set(X_val.columns)

    # Features only in training
    train_only = train_features - val_features
    if train_only:
        print(f"  Features only in training ({len(train_only)}): {sorted(train_only)[:10]}...")
        # Remove these from training
        X_train = X_train.drop(columns=list(train_only))

    # Features only in validation
    val_only = val_features - train_features
    if val_only:
        print(f"  Features only in validation ({len(val_only)}): {sorted(val_only)[:10]}...")
        # Remove these from validation
        X_val = X_val.drop(columns=list(val_only))

    # Ensure columns are in the same order
    X_val = X_val[X_train.columns]

    print(f"  Aligned training features: {X_train.shape[1]}")
    print(f"  Aligned validation features: {X_val.shape[1]}")

    # Encode categorical variables
    label_encoders = {}
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()

            # Fit on training data
            X_train[col] = le.fit_transform(X_train[col].astype(str))

            # Transform validation data, handling unseen labels
            if col not in X_val:
                continue

            X_val_col = X_val[col].astype(str)
            # Map unseen labels to a special value that exists in training
            unseen_mask = ~X_val_col.isin(le.classes_)
            if unseen_mask.any():
                unseen_labels = X_val_col[unseen_mask].unique()
                print(f"  Column '{col}': Found {len(unseen_labels)} unseen labels in validation: {unseen_labels[:5]}")
                # Replace unseen labels with the most frequent class from training
                X_val_col[unseen_mask] = le.classes_[0]

            X_val[col] = le.transform(X_val_col)
            label_encoders[col] = le

    # Encode target variable
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train)

    # Handle unseen labels in validation target
    y_val_series = y_val.copy()
    unseen_mask = ~y_val_series.isin(target_encoder.classes_)
    if unseen_mask.any():
        unseen_labels = y_val_series[unseen_mask].unique()
        print(f"  Target 'readmitted': Found {len(unseen_labels)} unseen labels in validation: {unseen_labels}")
        # Replace unseen labels with the most frequent class from training
        y_val_series[unseen_mask] = target_encoder.classes_[0]

    y_val = target_encoder.transform(y_val_series)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Define models to train
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    cumulative_results = []

    # Train each model
    for model_name, model in models.items():
        print(f"\n{'#'*60}")
        print(f"Training {model_name} on Cumulative Dataset {cumulative_idx}")
        print(f"Dataset size: {total_records} records")
        print(f"{'#'*60}")

        with mlflow.start_run(run_name=f"cumulative_{cumulative_idx}_{model_name}"):
            # Log cumulative batch information
            mlflow.log_param('cumulative_idx', cumulative_idx)
            mlflow.log_param('num_batches_included', cumulative_idx + 1)
            mlflow.log_param('total_records', total_records)
            mlflow.log_param('model_type', model_name)
            mlflow.log_param('train_size', len(X_train))
            mlflow.log_param('val_size', len(X_val))
            mlflow.log_param('training_strategy', 'cumulative_batch')
            mlflow.log_param('data_source', view_name)
            mlflow.log_param('num_features', X_train_scaled.shape[1])

            # Log feature names for reproducibility
            feature_names = X_train.columns.tolist()
            mlflow.log_dict({'feature_names': feature_names}, 'feature_names.json')

            # Train model
            print(f"Training {model_name}...")
            model.fit(X_train_scaled, y_train)

            # Training predictions and metrics
            y_train_pred = model.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
            train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)

            # ROC AUC for training
            try:
                if hasattr(model, 'predict_proba'):
                    y_train_proba = model.predict_proba(X_train_scaled)
                    train_roc_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='weighted')
                else:
                    train_roc_auc = 0.0
            except:
                train_roc_auc = 0.0

            # Validation predictions and metrics
            y_val_pred = model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
            val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

            # ROC AUC for validation
            try:
                if hasattr(model, 'predict_proba'):
                    y_val_proba = model.predict_proba(X_val_scaled)
                    val_roc_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr', average='weighted')
                else:
                    val_roc_auc = 0.0
            except:
                val_roc_auc = 0.0

            # Log metrics
            mlflow.log_metric('train_accuracy', train_accuracy)
            mlflow.log_metric('train_precision', train_precision)
            mlflow.log_metric('train_recall', train_recall)
            mlflow.log_metric('train_f1_score', train_f1)
            mlflow.log_metric('train_roc_auc', train_roc_auc)

            mlflow.log_metric('val_accuracy', val_accuracy)
            mlflow.log_metric('val_precision', val_precision)
            mlflow.log_metric('val_recall', val_recall)
            mlflow.log_metric('val_f1_score', val_f1)
            mlflow.log_metric('val_roc_auc', val_roc_auc)

            # Log model
            mlflow.sklearn.log_model(model, f"model_{model_name}")

            # Get run ID
            run_id = mlflow.active_run().info.run_id

            print(f"\n{model_name} Results:")
            print(f"  Training   - Acc: {train_accuracy:.4f}, F1: {train_f1:.4f}, ROC-AUC: {train_roc_auc:.4f}")
            print(f"  Validation - Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}")
            print(f"  MLflow Run ID: {run_id}")

            cumulative_results.append({
                'cumulative_idx': cumulative_idx,
                'num_batches': cumulative_idx + 1,
                'total_records': total_records,
                'model_name': model_name,
                'run_id': run_id,
                'val_f1_score': val_f1,
                'val_accuracy': val_accuracy,
                'val_roc_auc': val_roc_auc,
                'train_f1_score': train_f1,
                'train_accuracy': train_accuracy,
            })

    # Push results to XCom
    context['ti'].xcom_push(key=f'cumulative_{cumulative_idx}_results', value=cumulative_results)

    print(f"\n{'='*70}")
    print(f"✓ Cumulative dataset {cumulative_idx} training completed")
    print(f"  Total records processed: {total_records}")
    print(f"  Models trained: {len(models)}")
    print(f"{'='*70}\n")

    return cumulative_results