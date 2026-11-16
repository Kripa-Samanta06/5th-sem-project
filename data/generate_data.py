"""
Dataset Generator for Student Performance Prediction (Education Analytics)
Generates a realistic, well-calibrated dataset capturing the empirical impact of
attendance, study hours, and past academic records on passing or failing.
"""

import os
import numpy as np
import pandas as pd


def generate_student_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic student cohort data.
    Features:
      - attendance_rate: class attendance percentage (40% - 100%)
      - study_hours: self-study hours per week (1 - 35 hrs)
      - past_academic_score: past exam score / GPA percentile (30 - 100)
    Target:
      - passed: 1 (Pass) or 0 (Fail)
    """
    np.random.seed(random_state)

    student_ids = [f"STU_{i:04d}" for i in range(1, n_samples + 1)]

    # 1. Attendance percentage (mean=74%, std=13%, bounded 40%-100%)
    attendance = np.clip(np.random.normal(loc=74.0, scale=13.0, size=n_samples), 40.0, 100.0)

    # 2. Weekly study hours (gamma distribution: mean ~12 hrs, range 1-35)
    study_hours = np.clip(np.random.gamma(shape=3.0, scale=4.0, size=n_samples), 1.0, 35.0)

    # 3. Past academic score (prior exam records, mean ~64, bounded 30-100)
    past_score = np.clip(
        0.35 * attendance + 0.75 * study_hours + np.random.normal(loc=28.0, scale=10.0, size=n_samples),
        30.0,
        100.0,
    )

    # Academic Performance Composite Index (Log-odds of passing)
    # Educational benchmark: 75% attendance, 12h study, 60 past score is near 50/50 boundary
    log_odds = (
        0.080 * (attendance - 72.0)
        + 0.160 * (study_hours - 11.5)
        + 0.075 * (past_score - 60.0)
        + np.random.normal(loc=0.0, scale=0.65, size=n_samples)
    )

    # Pass probability via sigmoid function: P(Pass) = 1 / (1 + e^(-z))
    prob_pass = 1.0 / (1.0 + np.exp(-log_odds))

    # Binary outcome based on true probability
    passed = (np.random.uniform(0, 1, size=n_samples) < prob_pass).astype(int)

    df = pd.DataFrame(
        {
            "student_id": student_ids,
            "attendance_rate": np.round(attendance, 1),
            "study_hours": np.round(study_hours, 1),
            "past_academic_score": np.round(past_score, 1),
            "true_probability": np.round(prob_pass, 4),
            "result": np.where(passed == 1, "Pass", "Fail"),
            "passed": passed,
        }
    )

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_student_dataset(n_samples=2000, random_state=42)
    output_path = os.path.join("data", "student_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records at: {output_path}")
    print("\nSummary Statistics:")
    print(df[["attendance_rate", "study_hours", "past_academic_score", "passed"]].describe().to_string())
    print("\nPass vs Fail Distribution:")
    print(df["result"].value_counts(normalize=True).round(3))
