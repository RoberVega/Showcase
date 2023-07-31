import os
import pickle
import click
import mlflow
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRFClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("heart-disease-hpo")


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="data/data",
    help="Location where the processed Heart disease data was saved"
)
@click.option(
    "--num_trials",
    default=10,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(trial):
        with mlflow.start_run():

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 100, 1),
                'max_depth': trial.suggest_int('max_depth', 1, 50, 1),
                'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'eval_metric': roc_auc_score,
                'n_jobs': -1
            }

            mlflow.log_params(params)
            print(params)


            rf = XGBRFClassifier(**params)
            rf.fit(X_train, y_train)

            
            y_pred = rf.predict(X_val)

            rec = recall_score(y_true=y_val, y_pred=y_pred)
            prec = precision_score(y_true=y_val, y_pred=y_pred)
            mlflow.log_metric("recall",rec)
            mlflow.log_metric("precision",prec)

            return rec
    print('Started the hyperparameter optimization study\n\n')

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)
    print('\n\nFinished the hyperparameter optimization study')


if __name__ == '__main__':
    run_optimization()