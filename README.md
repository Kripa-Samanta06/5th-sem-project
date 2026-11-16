# 🎓 Student Performance Prediction (Education Analytics)

A production-grade machine learning system designed to predict whether a student will **Pass** or **Fail** based on their **attendance rate**, **weekly study hours**, and **past academic records**, outputting calibrated **pass/fail probability scores**, academic risk classifications, and pedagogical intervention guidance.

---

## 📌 Project Overview & Objectives

* **Core Objective:** Predict pass/fail outcomes and compute well-calibrated probability scores to identify at-risk students before final examinations.
* **Input Features:**
  1. `attendance_rate`: Class attendance percentage (0.0% – 100.0%)
  2. `study_hours`: Dedicated weekly self-study hours
  3. `past_academic_score`: Past academic records / previous exam scores (0.0 – 100.0)
* **Technologies:** Python 3.11+, Pandas, Scikit-Learn (`DecisionTreeClassifier`, `RandomForestClassifier`), Joblib, FastAPI, Pytest.
* **Primary Deliverables:**
  - Trained models (`models/best_student_model.pkl`, `models/random_forest_model.pkl`, `models/decision_tree_model.pkl`).
  - Prediction engine calculating exact $P(\text{Pass})$ and $P(\text{Fail})$ probability scores.
  - Academic Risk Tier categorization (Low, Moderate, Elevated, Critical).
  - Multi-interface execution: Interactive CLI, Batch CSV scoring, REST API, and automated unit test suite.

---

## 📂 Project Structure

```text
├── data/
│   ├── generate_data.py          # Synthetic dataset generator with pedagogical distributions
│   ├── student_data.csv          # Educational dataset (2,000 student records)
│   └── student_predictions.csv   # Batch prediction output with probability scores
├── models/
│   ├── best_student_model.pkl    # Serialized top-performing classifier
│   ├── random_forest_model.pkl   # Serialized Random Forest model
│   ├── decision_tree_model.pkl   # Serialized Decision Tree model
│   ├── decision_tree_rules.txt   # Human-readable rule tree export
│   └── model_metadata.json       # Hyperparameters, evaluation metrics, feature importances
├── src/
│   ├── __init__.py
│   ├── train.py                  # Training pipeline, stratified cross-validation & evaluation
│   └── predict.py                # Inference engine, probability scoring & risk categorizer
├── tests/
│   ├── test_model.py             # Model verification, probability bounds, monotonic checks
│   └── test_api.py               # REST API route handler tests
├── api.py                        # FastAPI REST microservice
├── main.py                       # Unified CLI interface (demo, interactive, batch, predict)
├── requirements.txt              # Project dependencies
└── README.md                     # Comprehensive documentation
```

---

## ⚙️ Mathematical & Algorithmic Formulation

### 1. Probability Estimation via Ensemble Averaging
For an input vector $\mathbf{x} = [\text{attendance}, \text{study\_hours}, \text{past\_score}]$, the Random Forest ensemble of $B = 150$ decision trees estimates the posterior probability of passing as:

$$P(\text{Pass} \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} P_b(\text{Pass} \mid \mathbf{x})$$

where $P_b(\text{Pass} \mid \mathbf{x})$ is the fraction of positive training samples in the leaf node reached by sample $\mathbf{x}$ in tree $b$.

### 2. Decision Tree Splitting (Gini Impurity)
The Decision Tree minimizes node impurity $I_G(t)$ at each split:

$$I_G(t) = 1 - \sum_{k=0}^{1} p(k \mid t)^2 = 1 - \left(p_{\text{fail}}^2 + p_{\text{pass}}^2\right)$$

### 3. Risk Stratification Criteria
| Probability Range $P(\text{Pass})$ | Risk Tier | Indicator | Recommended Pedagogical Action |
| :--- | :--- | :--- | :--- |
| **$0.75 \le P \le 1.00$** | **Low Risk** | 🟢 Secure Pass | Maintain current study consistency and course engagement. |
| **$0.50 \le P < 0.75$** | **Moderate Risk** | 🟡 Borderline Pass | Monitor attendance closely; fragile against minor drops in study hours. |
| **$0.35 \le P < 0.50$** | **Elevated Risk** | 🟠 At-Risk / Borderline Fail | Academic intervention recommended; establish structured weekly study goals. |
| **$0.00 \le P < 0.35$** | **Critical Risk** | 🔴 High At-Risk | Immediate remedial tutoring and attendance tracking required. |

---

## 📊 Model Evaluation & Benchmark Results

The models were evaluated using an 80/20 stratified train/test split on 2,000 student records:

| Evaluation Metric | Decision Tree Classifier | Random Forest Classifier (Selected Best) |
| :--- | :---: | :---: |
| **5-Fold Cross-Validation Accuracy** | $71.88\% \ (\pm 3.0\%)$ | **$74.44\% \ (\pm 2.1\%)$** |
| **Test Set Accuracy** | $74.00\%$ | **$75.00\%$** |
| **Precision (Pass Class)** | $75.42\%$ | **$75.62\%$** |
| **Recall (Pass Class)** | $79.46\%$ | **$81.70\%$** |
| **F1-Score** | $77.39\%$ | **$78.54\%$** |
| **ROC-AUC Score** | $0.7904$ | **$0.8154$** |

### Feature Importance Breakdown (Random Forest)
1. **Past Academic Score**: `39.9%` (Strongest baseline predictor of foundational comprehension)
2. **Weekly Study Hours**: `32.3%` (Primary active driver for academic improvement)
3. **Attendance Rate**: `27.9%` (Foundational prerequisite for classroom concept acquisition)

---

## 🚀 How to Run

### 1. Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Demonstration Mode
Run archetype demonstrations comparing High Achiever, Borderline, and At-Risk students:
```bash
python main.py --mode demo
```

### 3. Interactive Student Prediction
Launch the terminal evaluation prompt:
```bash
python main.py --mode interactive
```
*Prompt asks for Attendance Rate, Weekly Study Hours, and Past Academic Score, outputting the full score card.*

### 4. CLI Direct Prediction
Predict directly via command-line arguments:
```bash
python main.py --mode predict --attendance 82.5 --study_hours 15.0 --past_score 72.0
```

### 5. Batch CSV Scoring
Score an entire cohort from a CSV file and output results with probability scores:
```bash
python main.py --mode batch --batch_in data/student_data.csv --batch_out data/student_predictions.csv
```

### 6. Retraining & Evaluation
Retrain both models and regenerate model artifacts:
```bash
python src/train.py
```

### 7. Run Test Suite
Run the automated unit and integration tests:
```bash
python -m pytest tests/ -v
```

---

## 🌐 REST API Usage (FastAPI)

Start the REST API server:
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

### Predict Single Student: `POST /predict`
**Request:**
```json
{
  "attendance_rate": 88.0,
  "study_hours": 16.0,
  "past_academic_score": 78.0,
  "model_type": "best",
  "threshold": 0.50
}
```

**Response:**
```json
{
  "model_type": "best",
  "inputs": {
    "attendance_rate": 88.0,
    "study_hours": 16.0,
    "past_academic_score": 78.0
  },
  "probabilities": {
    "pass_probability": 0.9412,
    "fail_probability": 0.0588,
    "pass_percentage": "94.1%",
    "fail_percentage": "5.9%"
  },
  "prediction": "Pass",
  "decision_threshold": 0.5,
  "risk_assessment": {
    "level": "Low Risk",
    "badge": "🟢 SECURE PASS",
    "action": "Student is on track. Maintain current study rhythm and active engagement."
  },
  "recommendations": [
    "Exemplary academic indicators across all metrics. Maintain current trajectory!"
  ]
}
```
