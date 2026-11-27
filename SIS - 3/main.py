from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = FastAPI(
    title="Vertebral Column Diagnosis API",
    description="FastAPI service for classifying orthopedic spine measurements (SIS-3).",
    version="2.0.0",
)

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None


class PatientFeatures(BaseModel):
    pelvic_incidence: float = Field(..., description="Pelvic incidence angle")
    pelvic_tilt: float = Field(..., description="Pelvic tilt angle")
    lumbar_lordosis_angle: float = Field(..., description="Lumbar lordosis angle")
    sacral_slope: float = Field(..., description="Sacral slope angle")
    pelvic_radius: float = Field(..., description="Pelvic radius")
    grade_of_spondylolisthesis: float = Field(..., description="Grade of spondylolisthesis")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pelvic_incidence": 63.03,
                "pelvic_tilt": 22.55,
                "lumbar_lordosis_angle": 39.61,
                "sacral_slope": 40.48,
                "pelvic_radius": 98.67,
                "grade_of_spondylolisthesis": -0.25,
            }
        }
    }


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: PatientFeatures):
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="Model file not found. Run train.py first.",
            )
        model = joblib.load(MODEL_PATH)

    input_df = pd.DataFrame([features.model_dump()])
    pred_class = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    prob_dict = {
        str(cls): round(float(p), 4)
        for cls, p in zip(model.classes_, probs)
    }

    return {
        "prediction": str(pred_class),
        "probabilities": prob_dict,
    }
