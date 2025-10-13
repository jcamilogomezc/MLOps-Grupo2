# Aquí está el código que permite entrenar el modelo y loggearlo en MLflow

from fastapi import FastAPI
from contextlib import asynccontextmanager

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://mlflow:5000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print('Start training')
        mlflow.set_experiment("linear_regression_prediction")
        X, y = make_regression(n_samples=100, n_features=3, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        name = "linear_regression_model"

        with mlflow.start_run(run_name=f"run_{name}") as run:         
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            
            # Log parameters
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_metric("mse", mse)

            # Log model with proper input example (first row of training data)
            try:
                mlflow.sklearn.log_model(
                    model, 
                    artifact_path="model",
                    input_example=X_train[:1]
                )
                print(f"[TRAIN] Model {name} logged to MLflow successfully")
            except Exception as e:
                print(f"[TRAIN] Warning: Could not log model {name}: {e}")
            
            # Get run_id for registration
            run_id = run.info.run_id

        # Register the model to Model Registry
        client = MlflowClient()
        try:
            model_uri = f"runs:/{run_id}/model"
            model_details = mlflow.register_model(model_uri=model_uri, name=name)
            print(f"[REGISTER] Model registered with version {model_details.version}")
            
            # Transition to Production
            client.transition_model_version_stage(
                name=name,
                version=model_details.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"[TRANSITION] Model version {model_details.version} transitioned to Production")
        except Exception as e:
            print(f"[REGISTER/TRANSITION] Warning: Could not register/transition model: {e}")

        print("Finish training")
    except Exception as e:
        print("Error training model: ", e)

    yield

    print("Finish application")