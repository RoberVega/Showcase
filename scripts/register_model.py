import click
import mlflow
from typing import Optional
from prefect import flow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient



EXPERIMENT_NAME = "heart-disease-best-models"


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)


@click.command()
@click.option(
    "--run_id",
    #default=5,
    type=int,
    help="Id of the model to register"
)
@flow(name='Register a model')
def run_register_model(run_id: Optional[str]):

    client = MlflowClient()
    
    if run_id == None:

        # Select the model with the highest test_recall
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
            order_by=["metrics.test_rec DESC"]
            )[0]

        # Register the best model
        model_uri = f"runs:/{best_run.info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name='heart-disease-xgbrf')

    else:
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name='heart-disease-xgbrf')


if __name__ == '__main__':
    run_register_model()