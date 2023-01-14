import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


class ModelRegistry:
    def __init__(self):
        self.client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

    def register_best_model(self, model_name, experiment_id):

        model_runs = self.client.search_runs(
            experiment_ids=experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
            order_by=["metrics.MAPE ASC"],
        )

        lowest_run_value = 0
        lowest_run_id = 0
        for run in model_runs:
            print(
                f"run id: {run.info.run_id}, accuracy: {run.data.metrics['MAPE']:.4f}"
            )
            lowest_run_value = run.data.metrics["MAPE"]
            lowest_run_id = run.info.run_id

        print(lowest_run_value)
        print(lowest_run_id)

        artifact_path = "model"
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=lowest_run_id, artifact_path=artifact_path
        )

        model_name = model_name
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(model_details)

        return model_details.status

    def get_latest_registered_model_uri(self, registered_model_name):

        model_versions = self.client.search_model_versions(
            "name='{}'".format(registered_model_name)
        )
        version = max([model_version.version for model_version in model_versions])

        registered_model = "models:/{}/{}".format(
            registered_model_name, version
        )
        print(registered_model)

        return registered_model
