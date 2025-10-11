# Aquí está el código que permite entrenar el modelo y loggearlo en MLflow

from fastapi import FastAPI
from contextlib import asynccontextmanager

import mlflow
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# mlflow.set_tracking_uri("http://mlflow:5000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This is an basic example
    # This function create the model at the start of the application
    try:
        print('Start training')
        X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        mlflow.log_param("n_samples", 100)
        mlflow.log_param("n_features", 1)
        mlflow.log_metric("mse", mean_squared_error(y_test, model.predict(X_test)))

        mlflow.sklearn.log_model(model, name="linear_regression_model", input_example=X_train.flat[0])

        print("Finish training")
    except Exception as e:
        print("Error to training model: ", e)

    yield

    print("Finish application")