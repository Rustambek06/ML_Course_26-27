import os
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Vertebral Column Diagnosis",
    layout="centered",
)

st.title("Vertebral Column Diagnosis")
st.write(
    "Enter the six biomechanical spine and pelvis measurements below to classify "
    "the patient condition via the FastAPI model endpoint."
)

SAMPLES = {
    "Sample 1 - Disk Hernia": {
        "pelvic_incidence": 63.03,
        "pelvic_tilt": 22.55,
        "lumbar_lordosis_angle": 39.61,
        "sacral_slope": 40.48,
        "pelvic_radius": 98.67,
        "grade_of_spondylolisthesis": -0.25,
    },
    "Sample 2 - Normal": {
        "pelvic_incidence": 38.51,
        "pelvic_tilt": 16.96,
        "lumbar_lordosis_angle": 35.11,
        "sacral_slope": 21.54,
        "pelvic_radius": 127.63,
        "grade_of_spondylolisthesis": 7.99,
    },
    "Sample 3 - Spondylolisthesis": {
        "pelvic_incidence": 74.33,
        "pelvic_tilt": 19.24,
        "lumbar_lordosis_angle": 77.12,
        "sacral_slope": 55.09,
        "pelvic_radius": 115.86,
        "grade_of_spondylolisthesis": 52.38,
    },
}

selected_sample = st.selectbox("Preset patient example:", list(SAMPLES.keys()))
defaults = SAMPLES[selected_sample]

col1, col2 = st.columns(2)

with col1:
    pelvic_incidence = st.number_input(
        "Pelvic Incidence", value=defaults["pelvic_incidence"], step=0.5, format="%.2f"
    )
    pelvic_tilt = st.number_input(
        "Pelvic Tilt", value=defaults["pelvic_tilt"], step=0.5, format="%.2f"
    )
    lumbar_lordosis_angle = st.number_input(
        "Lumbar Lordosis Angle", value=defaults["lumbar_lordosis_angle"], step=0.5, format="%.2f"
    )

with col2:
    sacral_slope = st.number_input(
        "Sacral Slope", value=defaults["sacral_slope"], step=0.5, format="%.2f"
    )
    pelvic_radius = st.number_input(
        "Pelvic Radius", value=defaults["pelvic_radius"], step=0.5, format="%.2f"
    )
    grade_of_spondylolisthesis = st.number_input(
        "Grade of Spondylolisthesis",
        value=defaults["grade_of_spondylolisthesis"],
        step=0.5,
        format="%.2f",
    )

if st.button("Predict Diagnosis", type="primary"):
    payload = {
        "pelvic_incidence": pelvic_incidence,
        "pelvic_tilt": pelvic_tilt,
        "lumbar_lordosis_angle": lumbar_lordosis_angle,
        "sacral_slope": sacral_slope,
        "pelvic_radius": pelvic_radius,
        "grade_of_spondylolisthesis": grade_of_spondylolisthesis,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"]
        probs = result["probabilities"]

        if prediction == "Normal":
            st.success(f"Predicted Condition: **{prediction}**")
        else:
            st.warning(f"Predicted Condition: **{prediction}**")

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame(
            {"Class": list(probs.keys()), "Probability": list(probs.values())}
        ).set_index("Class")
        st.bar_chart(prob_df)
        st.dataframe(prob_df)

    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to the FastAPI server at {API_URL}. "
            "Please start it first with: uvicorn main:app --reload"
        )
    except Exception as e:
        st.error(f"Prediction request failed: {e}")
