from airflow.decorators import dag, task
from airflow.models import Variable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import mlflow
from mlflow import MlflowClient
import os 
import mlflow.sklearn 
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl 
from sklearn.model_selection import train_test_split
from datetime import datetime

@dag(
    dag_id="credit-risk-project",
    description="Project: Credist Risk model",
    start_date=datetime(2025, 07, 21),
    schedule_interval="@daily",
    tags=["mlops","zoomcamo","Project"],
    catchup=False
)

def pipeline():
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("home-credit-default-risk")
    
    @task 
    def read_and_preprocess_data():
        path = Variable.get("data_path")
        df = pd.read_csv(path)
        df = df[df["AMT_INCOME_TOTAL"] < df["AMT_INCOME_TOTAL"].quantile(0.95)]
        df["BUREAU"] = df["SK_ID_BUREAU"].notna().astype(int)
        df.drop( ["SK_ID_CURR","SK_ID_BUREAU"], inplace = True, axis = 1) 
        return df
        
    @task 
    def fill_null_values(df):
        missing_one = df.isnull().sum()
        missing_one = missing_one[missing_one == 1]
        common_missing = set(df[df[missing_one.index[0]].isnull()].index)
        for col in missing_one.index[1:]:
            common_missing &= set(df[df[col].isnull()].index)
            
        df.drop(list(common_missing)[0], inplace = True)
        
        df["AMT_ANNUITY"] = df["AMT_ANNUITY"].fillna(df["AMT_ANNUITY"].median())
        
        for target in df["TARGET"].unique():
            mask = (df["TARGET"] == target) & (df["OCCUPATION_TYPE"].isna())
            n_missing = mask.sum()
            probs = df.loc[
                (df["TARGET"] == target) & (df["OCCUPATION_TYPE"].notna()),"OCCUPATION_TYPE"
            ].value_counts(normalize = True)
            imputed_values = np.random.choice(probs.index, size = n_missing, p = probs.values)
            df.loc[mask, "OCCUPATION_TYPE"] = imputed_values
        
        df["EXT_SOURCE_2"] = df["EXT_SOURCE_2"].fillna(df["EXT_SOURCE_2"].median())
        
        df["CREDIT_ACTIVE"] = df["CREDIT_ACTIVE"].fillna("NO_BUREAU")
        
        df["CREDIT_DAY_OVERDUE_BIN"] = pd.cut(
            df["CREDIT_DAY_OVERDUE"],
            bins = [-1,0,7,30,365,float("inf")],
            labels = [0,1,2,3,4]
        )

        df["CREDIT_DAY_OVERDUE_BIN"] = df["CREDIT_DAY_OVERDUE_BIN"].cat.add_categories(["No_Bureau"])
        df["CREDIT_DAY_OVERDUE_BIN"] = df["CREDIT_DAY_OVERDUE_BIN"].fillna("No_Bureau")
        
        df.drop(["CREDIT_DAY_OVERDUE"], axis = 1, inplace = True)
        
        df["DAYS_CREDIT"] = df["DAYS_CREDIT"].fillna(9999)
        
        df["LOG_AMT_CREDIT_SUM"] = np.log1p(df["AMT_CREDIT_SUM"]) 
        
        lower = df["LOG_AMT_CREDIT_SUM"].quantile(0.01)
        upper = df["LOG_AMT_CREDIT_SUM"].quantile(0.99)
        df["LOG_AMT_CREDIT_SUM_CLIPPED"] = df["LOG_AMT_CREDIT_SUM"].clip(lower, upper)
        
        df["LOG_AMT_CREDIT_SUM_CLIPPED"] = df["LOG_AMT_CREDIT_SUM_CLIPPED"].fillna(-1)
        
        df.drop(["AMT_CREDIT_SUM","LOG_AMT_CREDIT_SUM"], axis = 1, inplace = True)
        
        df["CNT_FAM_MEMBERS"] = df["CNT_FAM_MEMBERS"].fillna(df["CNT_FAM_MEMBERS"].median())
        
        return df
            
    @task 
    def dummy_variables(df):
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
        
        return df 
        
    
    @task 
    def split_data(df):
        
        client = MlflowClient()
        model_version = client.get_model_version(name = "Final_model_XGB", version = "1")
        run_id = model_version.run_id
        local_path = mlflow.artifacts.download_artifacts(run_id = run_id, artifact_path = "top_features.pkl")
        with open(local_path, "rb") as f:
            top_features = pkl.load(f)

        x = df[top_features]
        
        return x
        
    @task 
    def make_predictions(x):
        
        client = MlflowClient()
        model_version = client.get_model_version(name = "Final_model_XGB", version = "1")
        run_id = model_version.run_id
        run  = client.get_run(run_id)
        best_threshold = run.data.metrics["best_threshold"]
        
        model = mlflow.sklearn.load_model("models:/Final_model_XGB/1")
        y_prob = model.predict_proba(x)[:, 1]
        y_pred_opt = (y_prob > best_threshold).astype(int)
        return y_pred_opt 
    
    @task
    def save_predictions(preds):
        output_path = "D:/mokon/Documents/MlopsZoomcamp_project_data/predictions.csv"
        pd.DataFrame({"prediction": preds}).to_csv(output_path, index=False)
    

    data = read_and_preprocess_data()
    df_null = fill_null_values(data)
    dumm = dummy_variables(df_null)
    x = split_data(dumm)
    predictions = make_predictions(x)
    save_predictions(predictions)    
     

dag_instance = pipeline()