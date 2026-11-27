from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "vertebral_column.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
DB_PATH = BASE_DIR / "mlflow.db"

EXPERIMENT_NAME = "Vertebral_Column_Classification"
REGISTERED_MODEL_NAME = "VertebralColumnClassifier"

FEATURE_COLUMNS = [
    "pelvic_incidence",
    "pelvic_tilt",
    "lumbar_lordosis_angle",
    "sacral_slope",
    "pelvic_radius",
    "grade_of_spondylolisthesis",
]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["diagnosis"]

    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Store MLflow tracking data and Model Registry in local SQLite database
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Compare two hyperparameter configurations
    experiments = [
        {"run_name": "rf_baseline_50_trees", "n_estimators": 50, "max_depth": 5},
        {"run_name": "rf_tuned_120_trees", "n_estimators": 120, "max_depth": None},
    ]

    best_f1 = -1.0
    best_model = None

    for cfg in experiments:
        with mlflow.start_run(run_name=cfg["run_name"]):
            model = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=cfg["n_estimators"],
                        max_depth=cfg["max_depth"],
                        random_state=random_state,
                    ),
                ),
            ])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")

            # 1. Log hyperparameters
            mlflow.log_params({
                "model_type": "RandomForestClassifier",
                "scaler": "StandardScaler",
                "n_estimators": cfg["n_estimators"],
                "max_depth": str(cfg["max_depth"]),
                "test_size": test_size,
                "random_state": random_state,
            })

            # 2. Log evaluation metrics
            mlflow.log_metrics({
                "accuracy": round(acc, 4),
                "f1_macro": round(f1, 4),
                "precision_macro": round(precision, 4),
                "recall_macro": round(recall, 4),
            })

            # 3. Save model file and log artifacts + register in Model Registry
            joblib.dump(model, MODEL_PATH)
            mlflow.log_artifact(str(MODEL_PATH), artifact_path="joblib_model")

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
                input_example=X_train.iloc[:3],
                registered_model_name=REGISTERED_MODEL_NAME,
            )

            print(
                f"[{cfg['run_name']}] accuracy={acc:.4f}, f1_macro={f1:.4f} "
                f"-> registered as '{REGISTERED_MODEL_NAME}'"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_model = model

    # Keep the best-performing model saved as model.joblib for FastAPI
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model saved to {MODEL_PATH.name} (f1_macro={best_f1:.4f})")


if __name__ == "__main__":
    main()
