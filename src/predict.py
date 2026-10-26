"""
JobShield Prediction Module

Loads trained artefacts and makes predictions
using the same preprocessing as training.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import sparse

from src.preprocessing import (
    prepare_features,
    TEXT_COLS
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def load_artifacts():
    """
    Load the training-fitted preprocessor,
    trained DNN and validation-selected threshold.
    """

    preprocessing_path = (
        MODEL_DIR / "preprocessor.joblib"
    )

    model_path = (
        MODEL_DIR / "jobshield_dnn.keras"
    )

    threshold_path = (
        MODEL_DIR / "decision_threshold.json"
    )

    required_files = [
        preprocessing_path,
        model_path,
        threshold_path
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required artefact not found: {file_path}"
            )

    preprocessor = joblib.load(preprocessing_path)

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    with open(
        threshold_path,
        "r",
        encoding="utf-8"
    ) as file:
        configuration = json.load(file)

    threshold = float(
        configuration["decision_threshold"]
    )

    if not 0 < threshold < 1:
        raise ValueError(
            "The saved classification threshold is invalid."
        )

    expected_features = len(
        preprocessor.get_feature_names_out()
    )

    model_features = int(
        model.input_shape[-1]
    )

    if expected_features != model_features:
        raise ValueError(
            "Preprocessing and DNN feature counts "
            "do not match."
        )

    return {
        "preprocessor": preprocessor,
        "model": model,
        "threshold": threshold,
        "configuration": configuration
    }


def predict_job(job_data, artifacts):
    """
    Predict whether a job advertisement is
    potentially fraudulent.

    Returns a model score and classification.
    The score is not a calibrated probability.
    """

    if not isinstance(job_data, dict):
        raise TypeError(
            "Job information must be a dictionary."
        )

    has_text = any(
        str(job_data.get(column) or "").strip()
        for column in TEXT_COLS
    )

    if not has_text:
        raise ValueError(
            "Enter a job title or some job posting text."
        )

    # Convert one advertisement into a DataFrame
    input_df = pd.DataFrame([job_data])

    # Apply the same fixed cleaning rules
    cleaned_data = prepare_features(input_df)

    # Use the SAVED fitted transformer
    features = artifacts["preprocessor"].transform(
        cleaned_data
    )

    if sparse.issparse(features):
        features = features.toarray()

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # Generate the DNN sigmoid score
    prediction = artifacts["model"].predict(
        features,
        verbose=0
    )

    score = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(
            "The model returned an invalid score."
        )

    threshold = artifacts["threshold"]

    flagged = score >= threshold

    # Illustrative screening bands.
    # These are NOT calibrated risk categories.
    if flagged:
        risk_level = "HIGH"
    elif score >= threshold * 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if flagged:
        classification = "POTENTIALLY FRAUDULENT"
    else:
        classification = "NOT FLAGGED"

    return {
        "classification": classification,
        "flagged": bool(flagged),
        "score": score,
        "threshold": threshold,
        "risk_level": risk_level
    }
