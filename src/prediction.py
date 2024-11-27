import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_model(model_path):
    """Load the saved model."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def preprocess_input(input_data, scaler_path):
    """
    Preprocess input data:
    - Scale the features using a pre-fitted StandardScaler.
    - Return the scaled features.
    """
    # Load the saved scaler
    scaler = pickle.load(open(scaler_path, "rb"))
    
    # Scale the input data
    scaled_data = scaler.transform([input_data])  # Transform as a single sample
    return scaled_data

def predict(input_data, model_path, scaler_path):
    """
    Predict the class of the input data using the trained model.
    Args:
        input_data (list or np.ndarray): Feature values for prediction.
        model_path (str): Path to the saved model.
        scaler_path (str): Path to the saved scaler.
    Returns:
        str: Predicted class label.
    """
    # Preprocess the input data
    input_scaled = preprocess_input(input_data, scaler_path)
    
    # Load the trained model
    model = load_model(model_path)
    
    # Make predictions
    predictions = model.predict(input_scaled)
    class_index = predictions.argmax(axis=1)[0]  # Get the predicted class index

    # Class mapping
    class_labels = {0: "Normal", 1: "Suspect", 2: "Pathological"}
    return class_labels[class_index]

if __name__ == "__main__":
    # Construct paths relative to the `src` folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root directory
    models_folder = os.path.join(base_dir, "models")
    model_path = os.path.join(models_folder, "fetalhealth.pkl")
    scaler_path = os.path.join(models_folder, "scaler.pkl")

    # Prompt user to enter the features
    print("Enter the following features:")
    features = [
        'baseline value', 'accelerations', 'fetal_movement', 
        'uterine_contractions', 'light_decelerations',
        'prolongued_decelerations', 'abnormal_short_term_variability',
        'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability',
        'histogram_width', 'histogram_min', 'histogram_max',
        'histogram_number_of_peaks', 'histogram_number_of_zeroes',
        'histogram_mode', 'histogram_median', 'histogram_tendency'
    ]
    
    # Collect input data
    input_data = []
    for feature in features:
        value = float(input(f"{feature}: "))
        input_data.append(value)

    # Make prediction
    predicted_class = predict(input_data, model_path, scaler_path)

    # Print the input data and prediction
    print("\nInput Data:")
    for feature, value in zip(features, input_data):
        print(f"{feature}: {value}")

    print(f"\nPredicted Class: {predicted_class}")
