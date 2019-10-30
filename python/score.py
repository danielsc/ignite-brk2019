import pandas as pd
import joblib
import os
from azureml.core.model import Model

def init():
    global original_model
    for entry in os.walk('.'): 
        print (entry) 

    # Retrieve the path to the model file using the model name
    # Assume original model is named original_prediction_model
    original_model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'log_reg.pkl')
    original_model = joblib.load(original_model_path)

def run(raw_data):
    # Get predictions and explanations for each data point
    data = pd.read_json(raw_data)
    # Make prediction
    predictions = original_model.predict_proba(data)
    # You can return any data type as long as it is JSON-serializable
    return predictions.tolist()