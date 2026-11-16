# 🎓 Model Card & Architecture Summary

This document provides a human-readable inspection of the trained machine learning models in `models/`.
Because `.pkl` files are compiled binary objects meant to be loaded by Python (`pickle.load()` or `joblib.load()`), this card details the full internal parameters, feature weights, decision thresholds, and performance metrics.

---

## 📌 Active Production Model: Random Forest Ensemble

* **File:** [`models/best_student_model.pkl`](best_student_model.pkl) (and duplicate [`models/random_forest_model.pkl`](random_forest_model.pkl))
* **Algorithm:** `sklearn.ensemble.RandomForestClassifier`
* **Target Variable:** `passed` (1 = Pass, 0 = Fail)
* **Number of Estimators ($B$):** `150` Decision Trees
* **Maximum Tree Depth:** `8`
* **Minimum Samples Split:** `10`
* **Minimum Samples Leaf:** `5`
* **Splitting Criterion:** Gini Impurity

### 📊 Feature Importances
| Feature Name | Feature Role | Model Weight |
| :--- | :--- | :---: |
| `past_academic_score` | Prior historical exam score (0-100) | **39.87%** |
| `study_hours` | Weekly dedicated self-study hours | **32.27%** |
| `attendance_rate` | Lecture attendance percentage (0-100%) | **27.86%** |

### 📈 Test Set Validation Metrics (400 Unseen Students)
* **Test Accuracy:** `75.00%`
* **5-Fold Cross-Validation Accuracy:** `74.44% (+/- 2.1%)`
* **Precision (Pass Class):** `75.62%`
* **Recall (Pass Class):** `81.70%`
* **F1-Score:** `78.54%`
* **ROC-AUC Score:** `0.8154`

---

## 🌲 Interpretable Model: Decision Tree Classifier

* **File:** [`models/decision_tree_model.pkl`](decision_tree_model.pkl)
* **Algorithm:** `sklearn.tree.DecisionTreeClassifier`
* **Max Depth:** `5`
* **Min Samples Split:** `15`
* **Min Samples Leaf:** `10`
* **Human-Readable Rule Export:** See [`models/decision_tree_rules.txt`](decision_tree_rules.txt)

---

## 💻 How to Programmatically Inspect the Model

Do not open `.pkl` files directly as text. Instead, run the inspector:

```bash
python inspect_model.py
```

Or load in Python:
```python
import joblib

model = joblib.load("models/best_student_model.pkl")
print("Model type:", type(model))
print("Classes:   ", model.classes_)
```
