import pandas as pd
import mlflow
import mlflow.sklearn
from dagshub import dagshub_logger, init
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

init(
    repo_owner="rizkiwnfproject",
    repo_name="Heart-Failure-Project",
    mlflow=True,
)

mlflow.set_tracking_uri(
    "https://dagshub.com/rizkiwnfproject/Heart-Failure-Project.mlflow"
)
mlflow.set_experiment("Heart Disease DagsHub")
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

df = pd.read_csv("heart_preprocessed.csv")
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
    }
    
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5,  n_jobs=-1,
    verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    best_params = grid.best_params_
    mlflow.log_params(best_params)
    mlflow.log_metric("manual_accuracy", acc)
    mlflow.log_metric("manual_precision", precision)
    mlflow.log_metric("manual_recall", recall)
    mlflow.log_metric("manual_f1", f1)
    mlflow.sklearn.log_model(best_model, "model")

    print("Logged to DagsHub with extra metrics.")
