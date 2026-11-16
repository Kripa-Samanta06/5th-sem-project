"""
Model Inspector Utility.
Safely inspects serialized model parameters, tree structures, and feature importances.
"""

import os
import json
import pickle
import joblib


def inspect(model_path: str = "models/best_student_model.pkl", metadata_path: str = "models/model_metadata.json"):
    print("=" * 65)
    print(f"🔍 MODEL INSPECTOR: {model_path}")
    print("=" * 65)

    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    # Load model
    try:
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    print(f"📦 Model Class        : {type(model).__name__}")
    print(f"📁 Module             : {type(model).__module__}")
    if hasattr(model, "n_estimators"):
        print(f"🌲 Number of Trees    : {model.n_estimators}")
    if hasattr(model, "max_depth"):
        print(f"📏 Max Depth          : {model.max_depth}")
    if hasattr(model, "classes_"):
        print(f"🏷️  Classes            : {list(model.classes_)} (0 = Fail, 1 = Pass)")
    if hasattr(model, "n_features_in_"):
        print(f"🔢 Input Features Count: {model.n_features_in_}")

    if hasattr(model, "feature_importances_"):
        features = ["attendance_rate", "study_hours", "past_academic_score"]
        print("\n📊 Feature Importances:")
        for feat, imp in zip(features, model.feature_importances_):
            print(f"   • {feat:<22}: {imp * 100:.2f}%")

    if os.path.exists(metadata_path):
        print("\n📈 Benchmark Metrics (from metadata):")
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            best_key = "random_forest" if "Random" in meta.get("best_model", "") else "decision_tree"
            metrics = meta.get("metrics", {}).get(best_key, {})
            for k in ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]:
                if k in metrics:
                    print(f"   • {k:<20}: {metrics[k]:.4f}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "models/best_student_model.pkl"
    inspect(path)
