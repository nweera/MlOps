import pandas as pd
import yaml
import joblib
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

model_type = config["model"]["model_type"]

# Start MLflow run
mlflow.start_run()

# Log parameters
mlflow.log_param("model_type", model_type)
mlflow.log_param("scaler", config["preprocessing"]["method"])
mlflow.log_param("test_size", config["data"]["test_size"])

# Model selection
if model_type == "logistic_regression":
    model = LogisticRegression(
        C=config["model"]["C"],
        max_iter=config["model"]["max_iter"],
        multi_class="auto"
    )

elif model_type == "svm":
    model = SVC(C=config["model"]["C"])

elif model_type == "random_forest":
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["training"]["random_state"]
    )

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

# Save model
joblib.dump(model, "model.pkl")

# Save metrics to file
with open("metrics.txt", "w") as f:
    f.write(f"accuracy: {accuracy}\n")
    f.write(f"precision_macro: {precision}\n")
    f.write(f"recall_macro: {recall}\n")
    f.write(f"f1_macro: {f1}\n")
    f.write(f"confusion_matrix:\n{cm}\n")

# Log metrics to MLflow
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision_macro", precision)
mlflow.log_metric("recall_macro", recall)
mlflow.log_metric("f1_macro", f1)

# Log artifacts
mlflow.log_artifact("model.pkl")
mlflow.log_artifact("metrics.txt")

mlflow.end_run()

print("Training completed:")
print(f"Accuracy:  {accuracy}")
print(f"Precision: {precision}")
print(f"Recall:    {recall}")
print(f"F1-score:  {f1}")
