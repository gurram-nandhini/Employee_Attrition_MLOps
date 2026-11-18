import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# 1. Load Dataset
df = pd.read_csv("employee.csv")

print("Dataset loaded successfully!")
print(df.head())


# 2. Separate Features and Target
X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"Yes": 1, "No": 0})


# 3. Identify Columns
categorical_features = ["Department", "JobRole"]

numerical_features = [
    "Salary",
    "YearsAtCompany",
    "Satisfaction"
]


# 4. Preprocessing
preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numerical_features
    ),

    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])


# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. Define Two Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# 7. Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Employee_Attrition_Experiment")


best_model = None
best_accuracy = -1


# 8. Train and Evaluate Models
for name, classifier in models.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    with mlflow.start_run(run_name=name):

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Evaluation
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        print("\nModel:", name)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # Log parameters
        mlflow.log_param("model_name", name)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save model in MLflow
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="Employee_Attrition_Model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        # Select best model
        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model


# 9. Save Best Model
joblib.dump(best_model, "model.pkl")


# 10. Final Result
print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")

print("Best Model Accuracy:", best_accuracy)

print("Best Model Saved as model.pkl")

print("MLflow Tracking Completed!")