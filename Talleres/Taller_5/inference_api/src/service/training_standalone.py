"""
Standalone training script for MLflow model registration
Run this directly without FastAPI
"""
import os
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "linear_regression_model")

def train_and_register_model():
    """Train model and register to MLflow"""
    print('🏋️  [TRAIN] Starting model training...')
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f'📍 [TRAIN] MLflow URI: {MLFLOW_TRACKING_URI}')
    
    # Check if model already exists in Production
    client = MlflowClient()
    try:
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if latest_versions:
            print(f'⏭️  [TRAIN] Model "{MODEL_NAME}" already in Production (v{latest_versions[0].version})')
            print('✅ [TRAIN] Skipping training - using existing model')
            return
    except Exception as e:
        print(f'📝 [TRAIN] No existing Production model found: {e}')
        print('🔄 [TRAIN] Proceeding with fresh training...')
    
    # Set experiment
    mlflow.set_experiment("linear_regression_prediction")
    print('📊 [TRAIN] Experiment set')
    
    # Generate synthetic data
    print('🔢 [TRAIN] Generating training data...')
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'📦 [TRAIN] Data split: {len(X_train)} train, {len(X_test)} test samples')
    
    # Train model
    print('🤖 [TRAIN] Training Linear Regression model...')
    model = LinearRegression()
    
    with mlflow.start_run(run_name=f"run_{MODEL_NAME}") as run:
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        
        # Log parameters
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("model_type", "LinearRegression")
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", mse ** 0.5)
        
        print(f'📈 [TRAIN] Model MSE: {mse:.4f}')
        
        # Log model
        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                input_example=X_train[:1],
                registered_model_name=MODEL_NAME
            )
            print(f"✅ [TRAIN] Model logged to MLflow successfully")
        except Exception as e:
            print(f"❌ [ERROR] Failed to log model: {e}")
            raise
        
        run_id = run.info.run_id
        print(f'🆔 [TRAIN] Run ID: {run_id}')
    
    # Get the latest version that was just registered
    print('🔍 [TRAIN] Fetching registered model version...')
    latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
    
    # Transition to Production
    print(f'🎯 [TRAIN] Transitioning model v{latest_version.version} to Production...')
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"✅ [SUCCESS] Model v{latest_version.version} is now in Production!")
    print(f"📊 [METRICS] Final MSE: {mse:.4f}")
    print("🎉 [TRAIN] Training pipeline completed successfully!")

if __name__ == "__main__":
    try:
        train_and_register_model()
        print("\n✅ Training script finished successfully")
        exit(0)
    except Exception as e:
        print(f"\n❌ Training script failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)