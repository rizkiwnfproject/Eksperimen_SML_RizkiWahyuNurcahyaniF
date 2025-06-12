import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, log_loss

df = pd.read_csv("heart_preprocessed.csv")
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Heart Disease Tuning")

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
    y_proba = best_model.predict_proba(X_test)[:, 1]

    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, best_model.predict_proba(X_test))

    best_params = grid.best_params_
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("log_loss", logloss)
    mlflow.sklearn.log_model(best_model, "model")

    print("Best Parameters:", best_params)
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | roc_auc: {roc_auc:.4f} | log_loss: {logloss:.4f}")
