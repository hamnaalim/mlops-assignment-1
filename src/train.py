import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models to compare
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

# Loop over models
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"\n✅ {model_name} Results:")
        print(f"  Accuracy : {acc:.2f}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall   : {recall:.2f}")
        print(f"  F1-score : {f1:.2f}")

        # Log parameters
        mlflow.log_param("model_type", model_name)
        if model_name == "LogisticRegression":
            mlflow.log_param("max_iter", 200)
        elif model_name == "DecisionTree":
            mlflow.log_param("max_depth", 3)
        elif model_name == "RandomForest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 5)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save model
        mlflow.sklearn.log_model(model, "model")
