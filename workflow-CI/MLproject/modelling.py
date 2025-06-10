import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import argparse

def train_and_log(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Logging
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # Save model locally
    os.makedirs("artifacts", exist_ok=True)
    mlflow.sklearn.save_model(model, "artifacts/model")

def main(data_path):
    mlflow.set_experiment("CI Heart Disease")

    # Ini akan work baik di `mlflow run` maupun `python modelling.py`
    if mlflow.active_run() is None:
        with mlflow.start_run(nested=True):
            train_and_log(args.data_path)
    else:
        train_and_log(data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="heart_preprocessed.csv")
    args = parser.parse_args()
    main(args.data_path)
