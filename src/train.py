import os
import json
import time
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from mlflow.tracking import MlflowClient

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Timestamp for this run
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

# Store results for JSON export
results_summary = {}

# Train & log each model
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"\n✅ {model_name} Results:")
        print(f"  Accuracy : {acc:.2f}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall   : {recall:.2f}")
        print(f"  F1-score : {f1:.2f}")

        # Save results
        results_summary[model_name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        # Log parameters
        mlflow.log_param("model_type", model_name)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save model with timestamp
        model_path = f"models/{model_name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        print(f"📁 Saved {model_name} to {model_path}")

        # Confusion Matrix Plot with timestamp
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=iris.target_names,
                    yticklabels=iris.target_names)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plot_path = f"results/{model_name}_confusion_matrix_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        print(f"📊 Confusion matrix saved and logged for {model_name}")

# === Save all metrics to JSON (with timestamp) ===
metrics_path = f"results/metrics_{timestamp}.json"
with open(metrics_path, "w") as f:
    json.dump(results_summary, f, indent=4)
print(f"\n📑 Metrics summary saved to {metrics_path}")

mlflow.log_artifact(metrics_path)

# === Select the best model (based on F1-score) ===
best_model = max(results_summary.items(), key=lambda x: x[1]["f1_score"])
best_model_name, best_model_metrics = best_model

print(f"\n🌟 Best Model: {best_model_name}")
print(f"   F1-score: {best_model_metrics['f1_score']:.2f}")

# Save best model info locally (with timestamp)
best_model_path = f"results/best_model_{timestamp}.txt"
with open(best_model_path, "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Metrics: {best_model_metrics}\n")

print(f"📑 Best model details saved to {best_model_path}")

# Log best model file into MLflow
mlflow.log_artifact(best_model_path)

print("\n✅ Training, evaluation, logging, and model registration steps completed!")
