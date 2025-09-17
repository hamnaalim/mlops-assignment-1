import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to compare
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42)
}

# Loop over models
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ {model_name} trained. Accuracy: {acc:.2f}")

        # Log params, metrics, and model
        mlflow.log_param("model_type", model_name)
        if model_name == "LogisticRegression":
            mlflow.log_param("max_iter", 200)
        else:
            mlflow.log_param("max_depth", 3)

        mlflow.log_metric("accuracy", acc)

        # Save model
        mlflow.sklearn.log_model(model, "model")
