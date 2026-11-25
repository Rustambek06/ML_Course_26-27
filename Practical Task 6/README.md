# Practical Task 6: Deploy ML Model with FastAPI and Docker

In this task, I trained a classification model on the **UCI Vertebral Column Dataset** and deployed it as a REST API using **FastAPI** and **Docker**.

## Dataset Overview

Instead of standard toy datasets, this project uses the biomechanical **Vertebral Column** dataset from the UCI Machine Learning Repository (collected at the Centre Médico-Chirurgical de Réadaptation des Massues, Lyon, France).

It contains **310 orthopedic patient records** described by 6 biomechanical measurements of the pelvis and lumbar spine:
- `pelvic_incidence`
- `pelvic_tilt`
- `lumbar_lordosis_angle`
- `sacral_slope`
- `pelvic_radius`
- `grade_of_spondylolisthesis`

**Target classes (`diagnosis`):**
- `Normal` (100 patients)
- `Disk Hernia` (60 patients)
- `Spondylolisthesis` (150 patients)

The model in `train.py` uses a scikit-learn `Pipeline` (`StandardScaler` + `RandomForestClassifier`) and reaches **~85.5% accuracy** on the held-out test set.

## Project Structure

```text
Practical Task 6/
├── main.py                # FastAPI application (/ and /predict endpoints)
├── train.py               # Trains the classifier and saves model.joblib
├── model.joblib           # Saved trained model artifact
├── vertebral_column.csv   # UCI Vertebral Column dataset (310 rows)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build instructions
└── README.md              # Project description and usage guide
```

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (optional — `model.joblib` is already included)

```bash
python train.py
```

### 3. Start the FastAPI server with Uvicorn

```bash
uvicorn main:app --reload
```

Once running, open:
- Root check: `http://127.0.0.1:8000/`
- Swagger UI (`/docs`): `http://127.0.0.1:8000/docs`

## Example API Requests

### `GET /`

Response:
```json
{
  "message": "ML API is running"
}
```

### `POST /predict`

Request body:
```json
{
  "pelvic_incidence": 63.03,
  "pelvic_tilt": 22.55,
  "lumbar_lordosis_angle": 39.61,
  "sacral_slope": 40.48,
  "pelvic_radius": 98.67,
  "grade_of_spondylolisthesis": -0.25
}
```

Example `curl` command:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"pelvic_incidence\": 63.03, \"pelvic_tilt\": 22.55, \"lumbar_lordosis_angle\": 39.61, \"sacral_slope\": 40.48, \"pelvic_radius\": 98.67, \"grade_of_spondylolisthesis\": -0.25}"
```

Response:
```json
{
  "prediction": "Disk Hernia",
  "probabilities": {
    "Disk Hernia": 0.7667,
    "Normal": 0.2167,
    "Spondylolisthesis": 0.0167
  }
}
```

## Run with Docker

### 1. Build the Docker image

```bash
docker build -t vertebral-ml-api .
```

### 2. Run the container

```bash
docker run -d -p 8000:8000 --name vertebral-api-container vertebral-ml-api
```

### 3. Test the containerized API

- Health check: `http://localhost:8000/`
- Interactive Swagger docs: `http://localhost:8000/docs`
- Prediction endpoint: `POST http://localhost:8000/predict`

To stop and remove the container when finished:
```bash
docker stop vertebral-api-container
docker rm vertebral-api-container
```
