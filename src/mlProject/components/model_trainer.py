import pandas as pd
import os
from mlProject import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
import mlflow
import mlflow.sklearn

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        models = {
            "RandomForest": RandomForestClassifier(random_state=self.config.random_state),
        }
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=self.config.random_state)

        results = {}
        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                model.fit(train_x, train_y)
                preds = model.predict(test_x)
                acc = accuracy_score(test_y, preds)
                prec = precision_score(test_y, preds, average='weighted')
                rec = recall_score(test_y, preds, average='weighted')
                f1 = f1_score(test_y, preds, average='weighted')
                # ROC-AUC only for binary
                roc_auc = None
                if len(set(test_y)) == 2:
                    roc_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
                    mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_params(model.get_params())
                # Confusion matrix
                cm = confusion_matrix(test_y, preds)
                cm_path = os.path.join(self.config.root_dir, f"{name}_confusion_matrix.csv")
                pd.DataFrame(cm).to_csv(cm_path, index=False)
                mlflow.log_artifact(cm_path)
                # Save model
                model_path = os.path.join(self.config.root_dir, f"{name}_model.joblib")
                joblib.dump(model, model_path)
                mlflow.sklearn.log_model(model, "model")
                results[name] = acc
        # Compare models
        best_model = max(results, key=results.get)
        print(f"Best model: {best_model} with accuracy {results[best_model]:.4f}")

