import pickle

# Load the model
with open("model/iris_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_species(features: list) -> str:
    classes = ['setosa', 'versicolor', 'virginica']
    prediction = model.predict([features])[0]
    return classes[prediction]
