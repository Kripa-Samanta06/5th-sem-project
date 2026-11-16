"""
Model Training and Evaluation Pipeline for Student Performance Prediction.
Implements DecisionTreeClassifier and RandomForestClassifier using Scikit-Learn.
"""

import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def load_and_prepare_data(csv_path: str, feature_cols: list = None):
    """Load dataset and extract feature matrix X and target y."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if feature_cols is None:
        # Default feature set adhering strictly to prompt requirements
        feature_cols = ["attendance_rate", "study_hours", "past_academic_score"]

    X = df[feature_cols]
    y = df["passed"] if "passed" in df.columns else (df["result"] == "Pass").astype(int)

    return df, X, y, feature_cols


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Evaluate classifier performance with cross-validation and test metrics."""
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    # Fit model on training data
    model.fit(X_train, y_train)

    # Predictions and probability scores
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Pass)

    # Compute core performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "model_name": model_name,
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["Fail", "Pass"], output_dict=True),
    }

    return model, results


def train_models(data_path="data/student_data.csv", output_dir="models"):
    """Train DecisionTree and RandomForest classifiers, evaluate, and save artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    # Selected features
    features = ["attendance_rate", "study_hours", "past_academic_score"]
    df, X, y, feature_names = load_and_prepare_data(data_path, features)

    print(f"Loaded {len(df)} student records.")
    print(f"Features used: {feature_names}")
    print(f"Class distribution: Fail={sum(y==0)}, Pass={sum(y==1)}\n")

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples\n")

    # 1. Decision Tree Classifier
    dt_clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=10,
        random_state=42,
    )
    dt_model, dt_results = evaluate_model(dt_clf, X_train, X_test, y_train, y_test, "Decision Tree Classifier")

    # 2. Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        criterion="gini",
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    rf_model, rf_results = evaluate_model(rf_clf, X_train, X_test, y_train, y_test, "Random Forest Classifier")

    # Print comparison summary table
    print("=" * 70)
    print(f"{'Metric':<25} | {'Decision Tree':<18} | {'Random Forest':<18}")
    print("-" * 70)
    print(f"{'CV Accuracy (5-fold)':<25} | {dt_results['cv_accuracy_mean']:.4f} (+/- {dt_results['cv_accuracy_std']:.3f}) | {rf_results['cv_accuracy_mean']:.4f} (+/- {rf_results['cv_accuracy_std']:.3f})")
    print(f"{'Test Accuracy':<25} | {dt_results['test_accuracy']:.4f}{'':<12} | {rf_results['test_accuracy']:.4f}")
    print(f"{'Precision (Pass)':<25} | {dt_results['test_precision']:.4f}{'':<12} | {rf_results['test_precision']:.4f}")
    print(f"{'Recall (Pass)':<25} | {dt_results['test_recall']:.4f}{'':<12} | {rf_results['test_recall']:.4f}")
    print(f"{'F1-Score':<25} | {dt_results['test_f1']:.4f}{'':<12} | {rf_results['test_f1']:.4f}")
    print(f"{'ROC-AUC Score':<25} | {dt_results['test_roc_auc']:.4f}{'':<12} | {rf_results['test_roc_auc']:.4f}")
    print("=" * 70)

    # Feature Importance for Random Forest
    rf_importances = dict(zip(feature_names, [round(float(v), 4) for v in rf_model.feature_importances_]))
    dt_importances = dict(zip(feature_names, [round(float(v), 4) for v in dt_model.feature_importances_]))
    print("\nFeature Importances (Random Forest):")
    for feat, imp in sorted(rf_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feat:<22}: {imp * 100:.1f}%")

    # Decision Tree Rule Export
    tree_rules = export_text(dt_model, feature_names=feature_names)
    with open(os.path.join(output_dir, "decision_tree_rules.txt"), "w") as f:
        f.write(tree_rules)

    # Select best model based on ROC-AUC / F1
    best_model_name = "Random Forest" if rf_results["test_roc_auc"] >= dt_results["test_roc_auc"] else "Decision Tree"
    best_model = rf_model if best_model_name == "Random Forest" else dt_model
    best_results = rf_results if best_model_name == "Random Forest" else dt_results

    # Save Models using standard protocol 4 pickle for universal compatibility with VS Code Pickle Viewer and joblib
    dt_path = os.path.join(output_dir, "decision_tree_model.pkl")
    rf_path = os.path.join(output_dir, "random_forest_model.pkl")
    best_path = os.path.join(output_dir, "best_student_model.pkl")

    with open(dt_path, "wb") as f:
        pickle.dump(dt_model, f, protocol=4)
    with open(rf_path, "wb") as f:
        pickle.dump(rf_model, f, protocol=4)
    with open(best_path, "wb") as f:
        pickle.dump(best_model, f, protocol=4)

    # Save Metadata and Evaluation metrics
    metadata = {
        "best_model": best_model_name,
        "features": feature_names,
        "classes": ["Fail", "Pass"],
        "class_mapping": {"0": "Fail", "1": "Pass"},
        "metrics": {
            "decision_tree": dt_results,
            "random_forest": rf_results,
        },
        "feature_importances": {
            "random_forest": rf_importances,
            "decision_tree": dt_importances,
        },
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nModels successfully saved to '{output_dir}/':")
    print(f"  - Decision Tree: {dt_path}")
    print(f"  - Random Forest: {rf_path}")
    print(f"  - Best Model:    {best_path} ({best_model_name})")
    print(f"  - Metadata:      {metadata_path}")

    return {
        "dt_model": dt_model,
        "rf_model": rf_model,
        "best_model": best_model,
        "metadata": metadata,
    }


if __name__ == "__main__":
    train_models()
