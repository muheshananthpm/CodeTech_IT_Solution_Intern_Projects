from fastapi import FastAPI
from app.schema import IrisInput
from app.model import predict_species

app = FastAPI(title="Iris Flower Predictor API")

@app.get("/")
def home():
    return {"message": "Welcome to the Iris Predictor API"}

@app.post("/predict")
def predict(input_data: IrisInput):
    features = [
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]
    prediction = predict_species(features)
    return {"prediction": prediction}
