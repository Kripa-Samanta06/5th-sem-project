"""
Prediction and Probability Scoring Engine for Student Performance.
Loads trained scikit-learn models (RandomForestClassifier, DecisionTreeClassifier)
and outputs pass/fail probability scores, risk levels, and actionable interventions.
"""

import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Optional


class StudentPerformancePredictor:
    """Predictor class supporting Decision Tree and Random Forest models."""

    MODEL_FILES = {
        "best": "best_student_model.pkl",
        "random_forest": "random_forest_model.pkl",
        "decision_tree": "decision_tree_model.pkl",
    }

    def __init__(self, models_dir: str = "models", default_model: str = "best"):
        self.models_dir = models_dir
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")
        self.metadata = {}

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

        self.feature_names = self.metadata.get(
            "features", ["attendance_rate", "study_hours", "past_academic_score"]
        )

        self.loaded_models: Dict[str, Any] = {}
        self.default_model_type = default_model
        # Load default model
        self.get_model(default_model)

    def get_model(self, model_type: str = "best"):
        """Load and cache requested model type."""
        model_type = model_type.lower()
        if model_type not in self.MODEL_FILES:
            raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(self.MODEL_FILES.keys())}")

        if model_type not in self.loaded_models:
            filename = self.MODEL_FILES[model_type]
            model_path = os.path.join(self.models_dir, filename)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file '{model_path}' not found. Please train models first.")
            try:
                self.loaded_models[model_type] = joblib.load(model_path)
            except Exception:
                with open(model_path, "rb") as f:
                    self.loaded_models[model_type] = pickle.load(f)

        return self.loaded_models[model_type]

    def _assess_risk(self, pass_prob: float) -> Dict[str, str]:
        """Categorize academic risk tier based on pass probability score."""
        if pass_prob >= 0.75:
            return {
                "level": "Low Risk",
                "badge": "🟢 SECURE PASS",
                "action": "Student is on track. Maintain current study rhythm and active engagement.",
            }
        elif pass_prob >= 0.50:
            return {
                "level": "Moderate Risk",
                "badge": "🟡 BORDERLINE PASS",
                "action": "Satisfactory performance but sensitive to variance. Monitor attendance closely.",
            }
        elif pass_prob >= 0.35:
            return {
                "level": "Elevated Risk",
                "badge": "🟠 AT-RISK / BORDERLINE FAIL",
                "action": "Proactive intervention recommended. Increase structured study hours and review past weak areas.",
            }
        else:
            return {
                "level": "Critical Risk",
                "badge": "🔴 HIGH AT-RISK OF FAILURE",
                "action": "Urgent academic counseling, remedial tutoring, and strict attendance tracking required.",
            }

    def _generate_recommendations(self, attendance: float, study_hours: float, past_score: float) -> List[str]:
        """Generate targeted pedagogical advice based on specific risk indicators."""
        advice = []
        if attendance < 75.0:
            advice.append(
                f"Improve attendance from {attendance:.1f}% to at least 75.0%+. Regular lecture participation is key to exam success."
            )
        if study_hours < 10.0:
            advice.append(
                f"Increase weekly study hours from {study_hours:.1f}h to at least 12.0h - 15.0h/week."
            )
        elif study_hours < 14.0:
            advice.append(
                f"Adding 2-3 focused self-study hours per week would elevate pass confidence."
            )
        if past_score < 60.0:
            advice.append(
                f"Prior academic mark ({past_score:.1f}/100) indicates knowledge gaps. Enroll in remedial revision sessions."
            )

        if not advice:
            advice.append("Exemplary academic indicators across all metrics. Maintain current trajectory!")

        return advice

    def predict_single(
        self,
        attendance_rate: float,
        study_hours: float,
        past_academic_score: float,
        model_type: str = "best",
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Predict pass/fail probability for an individual student.

        Parameters:
            attendance_rate: Class attendance percentage (0 - 100)
            study_hours: Hours of self-study per week (>= 0)
            past_academic_score: Prior exam record / GPA equivalent (0 - 100)
            model_type: 'best', 'random_forest', or 'decision_tree'
            threshold: Probability threshold for Pass classification

        Returns:
            Dictionary with probability scores, outcome, risk info, and recommendations.
        """
        model = self.get_model(model_type)

        features_df = pd.DataFrame(
            [
                {
                    "attendance_rate": float(attendance_rate),
                    "study_hours": float(study_hours),
                    "past_academic_score": float(past_academic_score),
                }
            ]
        )[self.feature_names]

        # Calculate class probabilities
        probs = model.predict_proba(features_df)[0]
        prob_fail = float(probs[0])
        prob_pass = float(probs[1])

        prediction = "Pass" if prob_pass >= threshold else "Fail"
        risk_info = self._assess_risk(prob_pass)
        recommendations = self._generate_recommendations(
            attendance_rate, study_hours, past_academic_score
        )

        return {
            "model_type": model_type,
            "inputs": {
                "attendance_rate": float(attendance_rate),
                "study_hours": float(study_hours),
                "past_academic_score": float(past_academic_score),
            },
            "probabilities": {
                "pass_probability": round(prob_pass, 4),
                "fail_probability": round(prob_fail, 4),
                "pass_percentage": f"{prob_pass * 100:.1f}%",
                "fail_percentage": f"{prob_fail * 100:.1f}%",
            },
            "prediction": prediction,
            "decision_threshold": threshold,
            "risk_assessment": risk_info,
            "recommendations": recommendations,
        }

    def compare_models(
        self,
        attendance_rate: float,
        study_hours: float,
        past_academic_score: float,
    ) -> Dict[str, Any]:
        """Compare predictions and probability scores between Decision Tree and Random Forest."""
        dt_res = self.predict_single(attendance_rate, study_hours, past_academic_score, model_type="decision_tree")
        rf_res = self.predict_single(attendance_rate, study_hours, past_academic_score, model_type="random_forest")

        return {
            "inputs": dt_res["inputs"],
            "decision_tree": {
                "prediction": dt_res["prediction"],
                "pass_probability": dt_res["probabilities"]["pass_probability"],
                "fail_probability": dt_res["probabilities"]["fail_probability"],
                "pass_percentage": dt_res["probabilities"]["pass_percentage"],
                "fail_percentage": dt_res["probabilities"]["fail_percentage"],
            },
            "random_forest": {
                "prediction": rf_res["prediction"],
                "pass_probability": rf_res["probabilities"]["pass_probability"],
                "fail_probability": rf_res["probabilities"]["fail_probability"],
                "pass_percentage": rf_res["probabilities"]["pass_percentage"],
                "fail_percentage": rf_res["probabilities"]["fail_percentage"],
            },
            "risk_assessment": rf_res["risk_assessment"],
            "recommendations": rf_res["recommendations"],
        }

    def predict_batch(
        self,
        data_or_path: Union[str, pd.DataFrame],
        model_type: str = "best",
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """Batch predict student pass/fail probabilities."""
        model = self.get_model(model_type)

        if isinstance(data_or_path, str):
            df = pd.read_csv(data_or_path)
        else:
            df = data_or_path.copy()

        for col in self.feature_names:
            if col not in df.columns:
                raise ValueError(f"Missing required feature column in batch data: '{col}'")

        X = df[self.feature_names]
        probs = model.predict_proba(X)

        df["pass_probability"] = np.round(probs[:, 1], 4)
        df["fail_probability"] = np.round(probs[:, 0], 4)
        df["predicted_result"] = np.where(df["pass_probability"] >= threshold, "Pass", "Fail")
        df["risk_level"] = [self._assess_risk(p)["level"] for p in df["pass_probability"]]

        return df


def format_prediction_card(res: Dict[str, Any]) -> str:
    """Format single model prediction as a clean terminal card."""
    inp = res["inputs"]
    probs = res["probabilities"]
    risk = res["risk_assessment"]
    m_name = res.get("model_type", "model").replace("_", " ").title()

    divider = "=" * 65
    card = f"""
{divider}
🎓 STUDENT PERFORMANCE PREDICTION & EDUCATION ANALYTICS
{divider}
🤖 Active Model         : {m_name}
📊 Student Profile:
   • Attendance Rate     : {inp['attendance_rate']:.1f}%
   • Weekly Study Hours  : {inp['study_hours']:.1f} hrs/week
   • Past Academic Score : {inp['past_academic_score']:.1f} / 100

📈 Pass / Fail Probability Scores:
   • Pass Probability    : {probs['pass_percentage']}  (Score: {probs['pass_probability']:.4f})
   • Fail Probability    : {probs['fail_percentage']}  (Score: {probs['fail_probability']:.4f})

🎯 Outcome Prediction   : {res['prediction'].upper()}
⚠️ Academic Risk Level  : {risk['badge']} ({risk['level']})

💡 Educational Guidance & Recommendations:
"""
    for rec in res["recommendations"]:
        card += f"   - {rec}\n"
    card += f"   Action: {risk['action']}\n"
    card += f"{divider}\n"
    return card


def format_comparison_card(comp: Dict[str, Any]) -> str:
    """Format side-by-side model comparison card."""
    inp = comp["inputs"]
    dt = comp["decision_tree"]
    rf = comp["random_forest"]

    divider = "=" * 65
    card = f"""
{divider}
🔬 MODEL COMPARISON: DECISION TREE vs RANDOM FOREST
{divider}
📊 Student Profile:
   • Attendance: {inp['attendance_rate']:.1f}% | Study Hours: {inp['study_hours']:.1f} h/wk | Past Score: {inp['past_academic_score']:.1f}/100

Model Prediction & Probability Breakdown:
-----------------------------------------------------------------
{'Model':<24} | {'Outcome':<8} | {'Pass Prob':<12} | {'Fail Prob':<12}
-----------------------------------------------------------------
{'Decision Tree':<24} | {dt['prediction']:<8} | {dt['pass_percentage']:<12} | {dt['fail_percentage']:<12}
{'Random Forest (Ensemble)':<24} | {rf['prediction']:<8} | {rf['pass_percentage']:<12} | {rf['fail_percentage']:<12}
-----------------------------------------------------------------
Risk Status: {comp['risk_assessment']['badge']} ({comp['risk_assessment']['level']})
{divider}\n"""
    return card
