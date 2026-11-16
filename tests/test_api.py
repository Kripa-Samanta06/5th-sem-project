"""
Unit Tests for FastAPI route handlers without network binding.
"""

from api import root, predict_student, compare_models_endpoint, get_model_metrics, StudentInput


def test_root_handler():
    res = root()
    assert res["status"] == "online"
    assert "best" in res["supported_models"]


def test_predict_single_handler():
    payload = StudentInput(
        attendance_rate=88.0,
        study_hours=16.0,
        past_academic_score=78.0,
        model_type="best",
        threshold=0.50,
    )
    res = predict_student(payload)
    assert res["prediction"] == "Pass"
    assert res["probabilities"]["pass_probability"] > 0.70
    assert "risk_assessment" in res


def test_predict_failing_handler():
    payload = StudentInput(
        attendance_rate=45.0,
        study_hours=2.0,
        past_academic_score=35.0,
    )
    res = predict_student(payload)
    assert res["prediction"] == "Fail"
    assert res["probabilities"]["fail_probability"] > 0.70


def test_compare_models_handler():
    payload = StudentInput(
        attendance_rate=75.0,
        study_hours=12.0,
        past_academic_score=65.0,
    )
    res = compare_models_endpoint(payload)
    assert "decision_tree" in res
    assert "random_forest" in res
    assert res["decision_tree"]["pass_probability"] >= 0.0


def test_get_metrics_handler():
    metrics = get_model_metrics()
    assert "best_model" in metrics
    assert "metrics" in metrics
