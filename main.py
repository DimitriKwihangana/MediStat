from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import os
from src.model import create_model
from src.preprocessing import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

class PredictionInput(BaseModel):
    baseline_value: float
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    light_decelerations: float
    prolongued_decelerations: float
    abnormal_short_term_variability: float
    mean_value_of_short_term_variability: float
    percentage_of_time_with_abnormal_long_term_variability: float
    histogram_width: float
    histogram_min: float
    histogram_max: float
    histogram_number_of_peaks: float
    histogram_number_of_zeroes: float
    histogram_mode: float
    histogram_median: float
    histogram_tendency: float
# Paths to model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fetalhealth.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


# Utility to load the model and scaler
def load_trained_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    return model, scaler


@app.post("/predict/")
async def predict(features: PredictionInput):
    """
    Predict the fetal health class based on input features.
    Args:
        features (dict): Dictionary of features with their values.
    Returns:
        dict: Predicted class.
    """
    try:
        # Load the model and scaler
        model, scaler = load_trained_model()

        # Convert input features to array
        input_data = np.array([features.baseline_value,
                       features.accelerations,
                       features.fetal_movement,
                       features.uterine_contractions,
                       features.light_decelerations,
                       features.prolongued_decelerations,
                       features.abnormal_short_term_variability,
                       features.mean_value_of_short_term_variability,
                       features.percentage_of_time_with_abnormal_long_term_variability,
                       features.histogram_width,
                       features.histogram_min,
                       features.histogram_max,
                       features.histogram_number_of_peaks,
                       features.histogram_number_of_zeroes,
                       features.histogram_mode,
                       features.histogram_median,
                       features.histogram_tendency]).reshape(1, -1)


        # Scale the data
        input_scaled = scaler.transform(input_data)

        # Predict the class
        predictions = model.predict(input_scaled)
        class_index = np.argmax(predictions, axis=1)[0]

        # Class mapping
        class_labels = {0: "Normal", 1: "Suspect", 2: "Pathological"}

        return {"prediction": class_labels[class_index]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/retrain/")
async def retrain(file: UploadFile = File(...)):
    """
    Retrain the existing model using the uploaded CSV file.
    Args:
        file (UploadFile): CSV file containing new data for retraining.
    Returns:
        dict: Retraining status and updated accuracy.
    """
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(BASE_DIR, "temp.csv")
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Preprocess the new dataset
        trainX, testX, trainY, testY = preprocessing(file_path)

        # Load the current model and its architecture
        model, _ = load_trained_model()
        input_shape = trainX.shape[1]
        new_model, callbacks = create_model(input_shape)

        # Retrain the model with the new data
        new_model.fit(
            trainX, trainY,
            validation_data=(testX, testY),
            epochs=10,  # Adjust as needed
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate the retrained model
        predictions = np.argmax(new_model.predict(testX), axis=1)
        testY_true = np.argmax(testY, axis=1)
        accuracy = accuracy_score(testY_true, predictions)

        # Save the updated model and scaler
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)
        os.remove(file_path)  # Clean up the temporary file

        return {"status": "Model retrained successfully", "accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain error: {str(e)}")



@app.post("/fine_tune/")
async def fine_tune(file: UploadFile = File(...), epochs: int = 10):
    """
    Fine-tune the existing model using the uploaded CSV file.
    Args:
        file (UploadFile): CSV file containing new data for fine-tuning.
        epochs (int): Number of epochs for fine-tuning.
    Returns:
        dict: Fine-tuning status and updated accuracy.
    """
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(BASE_DIR, "temp.csv")
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Preprocess the new dataset
        trainX, testX, trainY, testY = preprocessing(file_path)

        # Load the current model
        model, _ = load_trained_model()

        # Fine-tune the model with the new data
        model.fit(
            trainX, trainY,
            validation_data=(testX, testY),
            epochs=epochs,  # Fine-tuning epochs
            verbose=1
        )

        # Evaluate the fine-tuned model
        predictions = np.argmax(model.predict(testX), axis=1)
        testY_true = np.argmax(testY, axis=1)
        accuracy = accuracy_score(testY_true, predictions)

        # Save the updated model and scaler
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        os.remove(file_path)  # Clean up the temporary file

        return {"status": "Model fine-tuned successfully", "accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fine-tuning error: {str(e)}")