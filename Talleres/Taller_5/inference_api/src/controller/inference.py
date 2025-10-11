from src.app import app


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict():
    # The model must be loaded from MLflow and make the prediction with the training library
    return {"prediction": "ok"}
