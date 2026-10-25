# LRRS
# Loan Recommendation and Recovery System (LRRS)

## Overview

The Loan Recommendation and Recovery System (LRRS) is a machine learning-based decision-support application designed to assess loan default risk, explain predictions, recommend alternative loan configurations, and support post-loan monitoring and recovery decisions.

The system combines machine learning, explainable AI, and rule-based monitoring in an interactive Streamlit dashboard.

## Objectives

- Predict loan default risk using machine learning.
- Improve probability reliability through calibration.
- Explain predictions using SHAP-based explainable AI.
- Evaluate alternative loan amounts and repayment terms.
- Support post-loan monitoring and recovery decisions.

## Key Features

- **Loan Default Prediction:** Uses XGBoost to estimate the probability of loan default.
- **Probability Calibration:** Uses Platt scaling to improve the reliability of predicted probabilities.
- **Explainable AI:** Uses native XGBoost TreeSHAP contributions to identify factors influencing predictions.
- **Loan Optimization:** Evaluates alternative loan configurations based on predicted risk.
- **Post-Loan Monitoring:** Uses rule-based conditions to support early warnings and recovery workflows.
- **Interactive Dashboard:** Presents risk predictions, explanations, and recommendations through Streamlit.

## Technologies Used

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- TreeSHAP

See `requirements.txt` for the project's dependencies.

## System Workflow

1. Enter borrower and loan application details.
2. Preprocess and validate the input data.
3. Predict default risk using XGBoost.
4. Calibrate the predicted probability using Platt scaling.
5. Explain the prediction using TreeSHAP.
6. Evaluate alternative loan configurations.
7. Display predictions, explanations, recommendations, and monitoring information through the dashboard.

## Dataset

The project uses historical loan datasets, including LendingClub data, with a separate replication experiment using Prosper loan data.

Application-time feature selection is designed to avoid data leakage from information recorded after loan origination. In particular, `last_fico_range_low` and `last_fico_range_high` are excluded from application-time prediction features.

## Model Evaluation

The project includes temporal evaluation experiments using ROC-AUC, PR-AUC, and Brier score.

| Experiment | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| LendingClub baseline | 0.7164 | 0.3927 | — |
| Prosper replication | 0.7394 | 0.3712 | 0.1365 |

These results apply to their respective experimental datasets and evaluation settings. They do not guarantee performance on future loan portfolios.

## Limitations

- Performance depends on the quality and representativeness of historical data.
- Predicted probabilities do not guarantee actual loan outcomes.
- Loan optimization remains experimental and requires further improvement and validation.
- Post-loan monitoring is rule-based rather than a separately trained longitudinal prediction model.
- The system is a decision-support prototype and should not be used as the sole basis for real lending decisions without appropriate validation and governance.

## Future Work

- Evaluate the system using newer and institution-specific loan datasets.
- Improve loan optimization by incorporating affordability and lending-policy constraints.
- Investigate fairness and bias in credit risk predictions.
- Explore privacy-preserving collaboration between financial institutions.
- Enhance monitoring using validated repayment-history data.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/raghavkrishna01/LRRS.git
cd LRRS
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Install dependencies

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser if it does not open automatically.

## Responsible Use

LRRS is intended for educational, research, and decision-support purposes. Do not upload confidential borrower information, credentials, API keys, or datasets you do not have permission to share. Real-world deployment requires further validation, security reviews, fairness assessment, and compliance with applicable lending regulations.

## License

No license has been specified. Add a license before permitting others to reuse, modify, or distribute the project.

## Author

Raghav Krishna
