"""
DAG 6: Train Models
This DAG trains multiple machine learning models on the cleaned training data
and logs them to MLflow.
"""

from datetime import datetime, timedelta
from airflow.providers.postgres.hooks.postgres import PostgresHook
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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


def prepare_data_for_training(df, target_col='readmitted'):
    """
    Prepare data for training by encoding categorical variables.
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Encode categorical features
    label_encoders = {}
    X_encoded = X.copy()

    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le

    return X_encoded, y_encoded, label_encoders, le_target


def train_models(**context):
    """
    Train multiple machine learning models and log them to MLflow.
    Reads data from postgres_clean_data database using PostgresHook.
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    mlflow.set_experiment("diabetes_readmission_prediction")

    # Use PostgresHook to interact with the clean data database
    postgres_hook = PostgresHook(postgres_conn_id='clean_data')

    print(f"Connecting to clean data database...")

    # Read cleaned training data from PostgreSQL using hook
    print("Reading cleaned training data from train_clean table...")
    conn = postgres_hook.get_conn()
    train_df = pd.read_sql("SELECT * FROM train_clean", conn)
    conn.close()
    print(f"Training data shape: {train_df.shape}")

    # Prepare data
    print("Preparing data for training...")
    X_train, y_train, label_encoders, le_target = prepare_data_for_training(train_df)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"Training data prepared: {X_train_scaled.shape}")
    print(f"Target distribution: {np.bincount(y_train)}")

    # Define models to train
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    # Train each model
    trained_models = {}
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_features", X_train_scaled.shape[1])
            mlflow.log_param("n_samples", X_train_scaled.shape[0])

            # Train model
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_train_scaled)
            y_pred_proba = model.predict_proba(X_train_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_train, y_pred)
            precision = precision_score(y_train, y_pred, average='weighted')
            recall = recall_score(y_train, y_pred, average='weighted')
            f1 = f1_score(y_train, y_pred, average='weighted')

            # For multi-class ROC AUC
            try:
                roc_auc = roc_auc_score(y_train, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                roc_auc = 0.0

            # Log metrics
            mlflow.log_metric("train_accuracy", accuracy)
            mlflow.log_metric("train_precision", precision)
            mlflow.log_metric("train_recall", recall)
            mlflow.log_metric("train_f1_score", f1)
            mlflow.log_metric("train_roc_auc", roc_auc)

            # Log model
            mlflow.sklearn.log_model(model, model_name)

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")

            trained_models[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1
            }

    # Push model info to XCom
    ti = context['ti']
    model_results = {name: {'accuracy': info['accuracy'], 'f1_score': info['f1_score']}
                     for name, info in trained_models.items()}
    ti.xcom_push(key='trained_models', value=model_results)

    print(f"\n{'='*60}")
    print(f"Training complete! {len(trained_models)} models trained.")
    print(f"{'='*60}")

    return model_results
