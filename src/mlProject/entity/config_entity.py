from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    train_data_path: str
    test_data_path: str
    target_column: str
    root_dir: str
    model_name: str
    random_state: int = 42

@dataclass
class ModelEvaluationConfig:
    test_data_path: str
    model_path: str
    target_column: str
    mlflow_uri: str
    metric_file_name: str
    all_params: dict 