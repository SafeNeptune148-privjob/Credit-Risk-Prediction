from flask import Flask, request, jsonify 
import pandas as pd
import pickle as pkl
import mlflow 
from mlflow.tracking import MlflowClient

def prepare_features(record):
    
    df = pd.DataFrame([record])
    df.drop( ["SK_ID_CURR","SK_ID_BUREAU"], inplace = True, axis = 1)
    
    client = MlflowClient()
    model_version = client.get_model_version(name = "Final_model_XGB", version = "1")
    run_id = model_version.run_id
    local_path = mlflow.artifacts.download_artifacts(run_id = run_id, artifact_path = "cat_columns.pkl")
    
    with open(local_path, "rb") as f:
        cat_columns = pkl.load(f)
        
    yes_no = ["FLAG_OWN_CAR","FLAG_OWN_REALTY"] 
    
    for col in yes_no:
            df[col] = df[col].map({"Y":1,"N":0})
         
    df = pd.get_dummies(df, columns = cat_columns, drop_first = True).astype(int)
    
    local_path = mlflow.artifacts.download_artifacts(run_id = run_id, artifact_path = "top_features.pkl")
    
    with open(local_path, "rb") as f:
        top_features = pkl.load(f)

    x = df[top_features]
    
    return x

def make_predictions(x):
    
    client = MlflowClient()
    model_version = client.get_model_version(name = "Final_model_XGB", version = "1")
    run_id = model_version.run_id
    run  = client.get_run(run_id)
    best_threshold = run.data.metrics["best_threshold"]
    
    model = mlflow.sklearn.load_model("models:/Final_model_XGB/1")
    preds_proba = model.predict_proba(x)[:, 1]
    preds = (preds_proba > best_threshold).astype(int)
    return preds[0]

app = Flask('credit-risk-app')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    record = request.get_json()
    features = prepare_features(record)
    preds = make_predictions(features)
    if preds == 1:
        result = {
            "status":"default" 
        }
    else:
        result = {
            "status":"non-default"
        }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)