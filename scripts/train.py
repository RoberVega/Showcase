import os
import pathlib
import pickle
import click
import pandas as pd
import numpy as np
import scipy
import sklearn
import mlflow
import xgboost as xgb
from xgboost import XGBRFClassifier
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import precision_score, recall_score
from prefect import flow, task

mlflow.set_tracking_uri("sqlite:///mlflow.db")
HPO_EXPERIMENT_NAME = "heart-disease-hpo"
EXPERIMENT_NAME = "heart-disease-best-models"
RF_INT_PARAMS = ['n_estimators','max_depth','n_jobs']
RF_FLOAT_PARAMS = ['learning_rate','reg_alpha','reg_lambda']

#@task
def load_data(data_path: str):
    with open(data_path, "rb") as f_in:
        return pickle.load(f_in)


#@task
def train_and_log_model(data_path, params):
    X_train, y_train = load_data(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_data(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_data(os.path.join(data_path, "test.pkl"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    mlflow.xgboost.autolog()
    
    with mlflow.start_run():
        for param in RF_INT_PARAMS:
            params[param] = int(params[param])
        for param in RF_FLOAT_PARAMS:
            params[param] = float(params[param])
        params['eval_metric'] = recall_score

        rf = XGBRFClassifier(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rec = recall_score(y_true=y_val, y_pred=rf.predict(X_val))
        mlflow.log_metric("val_rec", val_rec)
        test_rec = recall_score(y_true=y_test, y_pred=rf.predict(X_test))
        mlflow.log_metric("test_rec", test_rec)

        val_prec = precision_score(y_true=y_val, y_pred=rf.predict(X_val))
        mlflow.log_metric("val_prec", val_prec)
        test_prec = precision_score(y_true=y_test, y_pred=rf.predict(X_test))
        mlflow.log_metric("test_prec", test_prec)

        pathlib.Path("models").mkdir(exist_ok=True)

        mlflow.xgboost.log_model(rf, artifact_path="models_mlflow")


#@flow
@click.command()
@click.option(
    "--data_path",
    default="data/data",
    help="Location where the processed heart disease data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models to save"
)
def run_train_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.recall DESC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)


if __name__ == "__main__":
    run_train_model()