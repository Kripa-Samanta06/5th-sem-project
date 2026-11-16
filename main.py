"""
Student Performance Prediction (Education Analytics) - Main Entry Point.
Provides CLI interface for single student predictions, batch scoring,
model comparisons, and model training.
"""

import os
import sys
import argparse
import pandas as pd
from src.train import train_models
from src.predict import StudentPerformancePredictor, format_prediction_card, format_comparison_card


def run_demo(predictor: StudentPerformancePredictor):
    """Run demonstration with diverse student archetypes."""
    print("\n" + "=" * 65)
    print("🚀 RUNNING EDUCATION ANALYTICS STUDENT PREDICTION DEMO")
    print("=" * 65)

    archetypes = [
        {
            "name": "High Achiever (High Attendance, Dedicated Study)",
            "attendance": 92.0,
            "study_hours": 20.0,
            "past_score": 88.0,
        },
        {
            "name": "Diligent Student with Past Struggles",
            "attendance": 85.0,
            "study_hours": 16.0,
            "past_score": 58.0,
        },
        {
            "name": "Borderline / Volatile Student",
            "attendance": 70.0,
            "study_hours": 10.0,
            "past_score": 60.0,
        },
        {
            "name": "Severe At-Risk Student (Chronic Absenteeism)",
            "attendance": 48.0,
            "study_hours": 3.0,
            "past_score": 38.0,
        },
    ]

    for student in archetypes:
        print(f"\n📋 Archetype: {student['name']}")
        comparison = predictor.compare_models(
            attendance_rate=student["attendance"],
            study_hours=student["study_hours"],
            past_academic_score=student["past_score"],
        )
        print(format_comparison_card(comparison))


def run_interactive(predictor: StudentPerformancePredictor):
    """Interactive command-line student evaluation tool."""
    print("\n" + "=" * 65)
    print("🎓 INTERACTIVE STUDENT PERFORMANCE EVALUATOR")
    print("=" * 65)

    try:
        raw_att = input("Enter Attendance Percentage (0 - 100)  [e.g. 78.5]: ").strip()
        attendance = float(raw_att) if raw_att else 78.5

        raw_study = input("Enter Weekly Study Hours (e.g. 12.0)     [e.g. 12.0]: ").strip()
        study_hours = float(raw_study) if raw_study else 12.0

        raw_score = input("Enter Past Academic Score (0 - 100)      [e.g. 65.0]: ").strip()
        past_score = float(raw_score) if raw_score else 65.0

        raw_thresh = input("Enter Pass Threshold Probability (0-1) [Default 0.50]: ").strip()
        threshold = float(raw_thresh) if raw_thresh else 0.50

        print("\n[1] Evaluating with Best Model (Random Forest Ensemble)...")
        res = predictor.predict_single(
            attendance_rate=attendance,
            study_hours=study_hours,
            past_academic_score=past_score,
            threshold=threshold,
        )
        print(format_prediction_card(res))

        print("\n[2] Side-by-Side Model Comparison:")
        comp = predictor.compare_models(
            attendance_rate=attendance,
            study_hours=study_hours,
            past_academic_score=past_score,
        )
        print(format_comparison_card(comp))

    except ValueError as e:
        print(f"❌ Invalid numerical input: {e}")
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")


def run_batch_evaluation(predictor: StudentPerformancePredictor, input_path: str, output_path: str):
    """Run batch prediction on an entire CSV file."""
    print(f"\nProcessing batch student records from: {input_path}")
    df_out = predictor.predict_batch(input_path)
    df_out.to_csv(output_path, index=False)
    print(f"✅ Successfully generated predictions for {len(df_out)} students.")
    print(f"📁 Output saved to: {output_path}")

    # Summary of predictions
    print("\nBatch Summary:")
    print(df_out["predicted_result"].value_counts(normalize=True).round(3))
    print("\nRisk Level Breakdown:")
    print(df_out["risk_level"].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Student Performance Prediction System")
    parser.add_argument("--mode", choices=["demo", "interactive", "train", "predict", "batch"], default="demo",
                        help="Execution mode (demo, interactive, train, predict, batch)")
    parser.add_argument("--attendance", type=float, help="Attendance percentage (0-100)")
    parser.add_argument("--study_hours", type=float, help="Weekly study hours")
    parser.add_argument("--past_score", type=float, help="Past academic score (0-100)")
    parser.add_argument("--model", choices=["best", "random_forest", "decision_tree"], default="best",
                        help="Classifier model to use")
    parser.add_argument("--batch_in", type=str, default="data/student_data.csv", help="Input CSV path for batch mode")
    parser.add_argument("--batch_out", type=str, default="data/predictions_output.csv", help="Output CSV path for batch mode")

    args = parser.parse_args()

    # Ensure models exist
    if not os.path.exists("models/best_student_model.pkl") or args.mode == "train":
        print("Training models...")
        train_models()
        if args.mode == "train":
            return

    predictor = StudentPerformancePredictor()

    if args.mode == "demo":
        run_demo(predictor)

    elif args.mode == "interactive":
        run_interactive(predictor)

    elif args.mode == "predict":
        if args.attendance is None or args.study_hours is None or args.past_score is None:
            print("❌ For --mode predict, please provide --attendance, --study_hours, and --past_score.")
            sys.exit(1)
        res = predictor.predict_single(
            attendance_rate=args.attendance,
            study_hours=args.study_hours,
            past_academic_score=args.past_score,
            model_type=args.model,
        )
        print(format_prediction_card(res))

    elif args.mode == "batch":
        run_batch_evaluation(predictor, args.batch_in, args.batch_out)


if __name__ == "__main__":
    main()
