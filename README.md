# Multi-Task Default & Interest Rate Prediction Model 🚀

## Overview
Welcome to a state-of-the-art machine learning pipeline for financial risk assessment! This project predicts both loan default (classification) and loan interest rate (regression) using advanced feature engineering and XGBoost modeling. Designed for real-world credit scoring, it emphasizes model interpretability, reproducibility, and actionable insights.

---

## ✨ Features
- **Automated Data Cleaning & Validation**: Ensure data quality from the start.
- **Smart Feature Engineering**: Select the most predictive features using XGBoost and model-based techniques.
- **Multi-Task Learning**: Predict default risk and interest rates in a single, unified workflow.
- **Hyperparameter Optimization**: Find the best XGBoost settings with randomized search.
- **Comprehensive Evaluation**: Get AUC, F1, Precision, Recall, confusion matrix, ROC curve, and more.
- **Model Interpretability**: SHAP analysis and feature importance plots for full transparency.
- **Reproducible Results**: All models, metrics, and plots are saved for easy review and sharing.

---

## 🧠 Deep Learning Multi-Task Baseline
This project also features a robust deep learning multi-task model built with TensorFlow/Keras. The model jointly predicts:
- **Loan Default (Classification):**
  - Default Accuracy: **0.9148**
  - Default AUC: **0.9104**
- **Loan Interest Rate (Regression):**
  - Interest MAE: **0.9647**
  - Interest MSE: **1.7464**

The deep learning model leverages shared representations for both tasks, improving generalization and efficiency. Training curves, ROC curves, and scatter plots for regression predictions are automatically generated and saved in `models/dl_multitask_baseline/`.

---

## 🗂️ Project Structure
```
├── data/                # Raw, cleaned, and feature-engineered datasets
├── docs/                # Documentation and changelogs
├── model-results/       # Model outputs, plots, and reports
├── models/              # Saved models and artifacts
├── notebooks/           # Jupyter notebooks for exploration and analysis
├── outputs/             # Additional outputs and importance scores
├── reports/             # Analytical reports and feature documentation
├── src/                 # Source code (data, features, models, evaluation)
├── tests/               # Unit and integration tests
├── requirements.txt     # Python dependencies
└── README.md            # Project overview (this file)
```

---

## 🚦 Getting Started
1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd multi-task-default-interest-model
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Prepare your data:**
   - Place your raw and processed data in the `data/` directory as shown above.
4. **Run the pipeline:**
   - Execute the main script for XGBoost baseline:
     ```sh
     python src/models/XGBoost.py
     ```
   - Outputs (plots, metrics, models) will be saved in the `models/` and `model-results/` directories.

---

## 🛠️ Key Scripts
- `src/models/XGBoost.py`: Main pipeline for XGBoost-based multi-task modeling.
- `src/features/feature_engineer_xgb_top30.py`: Feature selection and engineering for XGBoost.
- `src/data/`: Data cleaning and validation utilities.

---

## 🔍 Model Interpretability
- SHAP summary and waterfall plots for transparency.
- Feature importance is visualized and logged for both tasks.

---

## 🤝 Contributing
We welcome contributions from the community! If you have ideas, spot bugs, or want to add features, please open an issue or submit a pull request.

---

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 📬 Contact
Questions or want to collaborate? Reach out via GitHub Issues or email the maintainer. Let's build better credit risk models together!
