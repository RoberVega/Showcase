import joblib
import mlflow
import pandas as pd
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient
from preprocess_data import preprocess
from preprocess_utils.preprocess import prepare_features


ct = joblib.load("data/data/ct.pkl")


def load_model_from_registry(model_name, model_version):
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    #client.transition_model_version_stage(
    #    name=model_name, 
    #    version=model_version, 
    #    stage="Production"
    # )
    # Load the model as a PyFunc model
    model = mlflow.xgboost.load_model(f"models:/{model_name}/{model_version}")

    return model

def prepare_patient(patient: pd.DataFrame):
    patient_prep = prepare_features(patient)
    patient_prep, _ = preprocess(patient_prep, ct, fit_ct=False)
    return patient_prep

def predict_probability(patient: pd.DataFrame):
    
    model = load_model_from_registry('heart-disease-xgbrf', 1)

    probability = model.predict_proba(patient)

    return probability



app = Flask('heart-disease-prediction')

@app.route('/', methods=['GET'])
def root_endpoint():
    return "Heart Disease Prediction API"

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    patient = pd.DataFrame(request.get_json(), index=[0])   
     
    patient = prepare_patient(patient)
    
    proba = predict_probability(patient)[:,1][0]

    result = {
        'Probability of diabetes': f'{proba}'
    }
    print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


