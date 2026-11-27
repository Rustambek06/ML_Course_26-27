# SIS - 3: Full ML Pipeline with FastAPI, Streamlit, and MLflow

This project extends **Practical Task 6** (Vertebral Column orthopedic classification API) into a complete ML pipeline by adding:
1. **MLflow integration** — experiment tracking (hyperparameters, metrics, model artifacts) and model registration in the **MLflow Model Registry**.
2. **Streamlit frontend** — a clean web interface (`app.py`) where a user can enter or select spine/pelvis measurements and request predictions from the FastAPI service.

---

## System Architecture

```text
train.py ──► Logs params, metrics & artifacts to MLflow (mlflow.db)
         ──► Registers model versions in MLflow Model Registry
         ──► Exports best pipeline to model.joblib

app.py (Streamlit UI, port 8501)
   │
   │  POST /predict (JSON)
   ▼
main.py (FastAPI Backend, port 8000) ──► Loads model.joblib ──► Returns prediction + probabilities
```

---

## Project Structure

```text
SIS - 3/
├── train.py               # Trains models, logs to MLflow, registers model versions
├── main.py                # FastAPI backend (/ and /predict)
├── app.py                 # Streamlit user interface
├── model.joblib           # Saved best RandomForest pipeline
├── vertebral_column.csv   # UCI Vertebral Column dataset (310 patient records)
├── requirements.txt       # Python dependencies
├── Dockerfile             # FastAPI container configuration
├── Dockerfile.streamlit   # Streamlit container configuration
├── docker-compose.yml     # Runs FastAPI + Streamlit together
└── README.md              # Documentation and run steps
```

---

## How to Run Locally

Open terminals inside the `SIS - 3` folder:

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models and register in MLflow
```bash
python train.py
```
This script runs two Random Forest configurations (`rf_baseline_50_trees` and `rf_tuned_120_trees`) under the **`Vertebral_Column_Classification`** experiment:
- Logs hyperparameters (`n_estimators`, `max_depth`, `scaler`, `test_size`, `random_state`)
- Logs evaluation metrics (`accuracy`, `f1_macro`, `precision_macro`, `recall_macro`)
- Logs `model.joblib` and scikit-learn model artifacts
- Registers both runs in the **MLflow Model Registry** as **`VertebralColumnClassifier`** (Versions 1 and 2)

### 3. View MLflow UI (Experiment Tracking & Model Registry)
```bash
mlflow ui
```
Open `http://127.0.0.1:5000` in your browser to inspect:
- **Experiments** -> `Vertebral_Column_Classification`
- **Models** -> `VertebralColumnClassifier` (Version 1 & Version 2)

### 4. Start the FastAPI backend
```bash
uvicorn main:app --reload
```
- Root check: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

### 5. Start the Streamlit frontend (in a new terminal)
```bash
streamlit run app.py
```
Open `http://localhost:8501`, pick a preset patient sample or enter custom values, and click **Predict Diagnosis**.

---

## Run with Docker Compose (Optional)

To build and launch both the FastAPI backend (`:8000`) and Streamlit frontend (`:8501`) in containers:

```bash
docker compose up --build
```
