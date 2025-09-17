🔹 Problem Statement

The goal of this project is to train, evaluate, and monitor multiple ML models using an MLOps pipeline.
We implement tracking with MLflow, log metrics and artifacts, and register the best model.

The problem is a classification task using the Iris dataset, where the objective is to predict the species of a flower based on its features (sepal length, sepal width, petal length, petal width).

🔹 Dataset Description

Dataset: Iris Dataset (from scikit-learn).

Samples: 150 flowers (3 classes, 50 each).

Features: 4 numeric features (sepal length, sepal width, petal length, petal width).

Classes:

Setosa

Versicolor

Virginica

🔹 Model Selection & Comparison

We trained and compared 3 models:

Logistic Regression

Decision Tree

Random Forest

Metrics logged:

Accuracy

Precision

Recall

F1-score

Results are saved in:

/results/metrics_<timestamp>.json

/results/best_model_<timestamp>.txt

🔹 MLflow Tracking & Logging

All runs are tracked in MLflow.

Parameters (e.g., model type).

Metrics (accuracy, precision, recall, F1-score).

Artifacts (confusion matrices, metrics.json, best_model.txt).

Launch MLflow UI with:

mlflow ui


Open in browser: http://127.0.0.1:5000

📸 Screenshot of MLflow UI should be placed here (showing multiple runs and metrics).

🔹 Model Registration

The best model is selected based on F1-score and saved into:

/results/best_model_<timestamp>.txt

This file is also logged as an artifact in MLflow.

If connected to an MLflow server with registry enabled, the best model can be registered as:

from mlflow import register_model

mlflow.register_model(
    model_uri="runs:/<RUN_ID>/model",
    name="IrisClassifier"
)