"""
Automated Unit and Integration Tests for Student Performance Prediction.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.predict import StudentPerformancePredictor
from src.train import load_and_prepare_data, train_models


def test_data_loading():
    """Verify dataset loading and target encoding."""
    csv_path = "data/student_data.csv"
    assert os.path.exists(csv_path), "Data file should exist"

    df, X, y, features = load_and_prepare_data(csv_path)
    assert len(df) > 0
    assert X.shape[1] == 3
    assert set(features) == {"attendance_rate", "study_hours", "past_academic_score"}
    assert set(y.unique()).issubset({0, 1})


def test_model_files_exist():
    """Verify that model files and metadata were created."""
    assert os.path.exists("models/best_student_model.pkl")
    assert os.path.exists("models/random_forest_model.pkl")
    assert os.path.exists("models/decision_tree_model.pkl")
    assert os.path.exists("models/model_metadata.json")


def test_probability_distribution():
    """Verify that pass and fail probabilities sum to 1.0."""
    predictor = StudentPerformancePredictor()
    res = predictor.predict_single(attendance_rate=80.0, study_hours=14.0, past_academic_score=72.0)

    p_pass = res["probabilities"]["pass_probability"]
    p_fail = res["probabilities"]["fail_probability"]

    assert 0.0 <= p_pass <= 1.0
    assert 0.0 <= p_fail <= 1.0
    assert np.isclose(p_pass + p_fail, 1.0, atol=1e-3)


def test_monotonic_consistency():
    """A student with high attendance and study should have a higher pass probability than a failing profile."""
    predictor = StudentPerformancePredictor()

    high_student = predictor.predict_single(attendance_rate=95.0, study_hours=25.0, past_academic_score=90.0)
    low_student = predictor.predict_single(attendance_rate=45.0, study_hours=2.0, past_academic_score=35.0)

    p_high = high_student["probabilities"]["pass_probability"]
    p_low = low_student["probabilities"]["pass_probability"]

    assert p_high > p_low
    assert high_student["prediction"] == "Pass"
    assert low_student["prediction"] == "Fail"


def test_batch_prediction():
    """Verify batch prediction DataFrame structure."""
    predictor = StudentPerformancePredictor()
    test_df = pd.DataFrame(
        [
            {"student_id": "TEST_1", "attendance_rate": 90.0, "study_hours": 20.0, "past_academic_score": 85.0},
            {"student_id": "TEST_2", "attendance_rate": 50.0, "study_hours": 3.0, "past_academic_score": 40.0},
        ]
    )

    pred_df = predictor.predict_batch(test_df)
    assert "pass_probability" in pred_df.columns
    assert "fail_probability" in pred_df.columns
    assert "predicted_result" in pred_df.columns
    assert "risk_level" in pred_df.columns
    assert len(pred_df) == 2
