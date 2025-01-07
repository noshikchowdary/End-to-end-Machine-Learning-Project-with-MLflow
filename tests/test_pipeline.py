import os
import shutil
import tempfile
import mlflow
import pytest
from src.mlProject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline

def test_training_pipeline_runs_and_logs():
    # Use a temporary directory for MLflow tracking
    temp_dir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"file://{temp_dir}")
    experiment_name = "Default"
    mlflow.set_experiment(experiment_name)

    # Run the training pipeline
    pipeline = ModelTrainerTrainingPipeline()
    pipeline.main()

    # Check that at least one run was logged
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[client.get_experiment_by_name(experiment_name).experiment_id])
    assert len(runs) > 0, "No MLflow runs were logged."

    # Clean up temp directory
    shutil.rmtree(temp_dir) 