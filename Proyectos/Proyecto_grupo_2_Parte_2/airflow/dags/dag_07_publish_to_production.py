"""
DAG 8: Publish Model to Production
This DAG registers the best model in MLflow Model Registry and promotes it to production.
"""

from datetime import datetime, timedelta
import os
import mlflow
from mlflow.tracking import MlflowClient


# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def publish_model_to_production(**context):
    """
    Register the best model in MLflow Model Registry and promote it to production.
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))

    # Initialize MLflow client
    client = MlflowClient()

    # Get experiment
    experiment_name = 'diabetes_cumulative_batch_training'
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found!")

    # Get all runs and find the best one based on validation F1 score
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["metrics.val_f1_score DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found in the experiment!")

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_name_param = best_run.data.params.get('model_type', 'unknown')

    # Get metrics
    val_f1_score = best_run.data.metrics.get('val_f1_score', 0)
    val_accuracy = best_run.data.metrics.get('val_accuracy', 0)

    print(f"\n{'='*60}")
    print(f"PUBLISHING MODEL TO PRODUCTION")
    print(f"{'='*60}")
    print(f"Model Type: {model_name_param}")
    print(f"Run ID: {run_id}")
    print(f"Validation F1 Score: {val_f1_score:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"{'='*60}\n")

    # Register model in Model Registry
    model_uri = f"runs:/{run_id}/{model_name_param}"
    registered_model_name = "diabetes_readmission_model"

    print(f"Registering model '{registered_model_name}' in Model Registry...")

    try:
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
            tags={
                "model_type": model_name_param,
                "val_f1_score": str(val_f1_score),
                "val_accuracy": str(val_accuracy)
            }
        )

        version_number = model_version.version
        print(f"Model registered successfully as version {version_number}")

        # Transition model to Production stage
        print(f"Transitioning model version {version_number} to Production stage...")

        client.transition_model_version_stage(
            name=registered_model_name,
            version=version_number,
            stage="Production",
            archive_existing_versions=True  # Archive previous production versions
        )

        print(f"Model version {version_number} is now in Production!")

        # Add description to the model version
        client.update_model_version(
            name=registered_model_name,
            version=version_number,
            description=f"Best {model_name_param} model with val_f1={val_f1_score:.4f}, val_accuracy={val_accuracy:.4f}"
        )

        # Update registered model description
        try:
            client.update_registered_model(
                name=registered_model_name,
                description="Production model for predicting diabetes patient readmission"
            )
        except:
            pass  # Model description might already exist

        print(f"\n{'='*60}")
        print(f"PRODUCTION DEPLOYMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Model Name: {registered_model_name}")
        print(f"Version: {version_number}")
        print(f"Stage: Production")
        print(f"{'='*60}\n")

        # Push to XCom
        ti = context['ti']
        ti.xcom_push(key='production_model_name', value=registered_model_name)
        ti.xcom_push(key='production_model_version', value=version_number)
        ti.xcom_push(key='production_run_id', value=run_id)

        return {
            'model_name': registered_model_name,
            'version': version_number,
            'run_id': run_id,
            'model_type': model_name_param,
            'val_f1_score': val_f1_score,
            'val_accuracy': val_accuracy
        }

    except Exception as e:
        print(f"Error during model registration/promotion: {e}")
        raise
