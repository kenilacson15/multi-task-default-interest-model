# Multi-Task Default & Interest Rate Prediction Model 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Baseline-lightgrey.svg)](https://xgboost.readthedocs.io/)

---

## 🌟 Introduction

**Multi-Task Default & Interest Rate Prediction Model** is a modern, production-ready machine learning pipeline for financial risk assessment. The project’s core is a deep learning multi-task model that simultaneously predicts loan default (classification) and loan interest rate (regression) from engineered features. XGBoost baselines are included for benchmarking and interpretability.

This repository is designed for real-world credit scoring, with a focus on transparency, reproducibility, and extensibility. Whether you’re a data scientist, ML engineer, or financial analyst, this project provides a robust foundation for advanced credit risk modeling.

---

## ✨ Features

- **Unified Multi-Task Deep Learning:** Jointly predicts default risk and interest rates in a single, efficient architecture.
- **XGBoost Baselines:** Strong tree-based models for benchmarking and interpretability.
- **End-to-End Data Pipeline:** Automated data cleaning, validation, and advanced feature engineering.
- **Comprehensive Evaluation:** AUC, F1, Precision, Recall, confusion matrix, ROC curve, MAE, MSE, and more.
- **Model Interpretability:** SHAP analysis, feature importance plots, and clear reporting.
- **Reproducibility:** All models, metrics, and plots are versioned and saved for review.
- **Modern Engineering Practices:** Modular code, early stopping, checkpointing, and robust logging.

---

## 💡 Use Cases

- **Credit Scoring:** Predict the likelihood of loan default and estimate fair interest rates for applicants.
- **Risk Management:** Identify high-risk borrowers and optimize lending strategies.
- **Benchmarking:** Compare deep learning and XGBoost approaches for multi-task financial modeling.
- **Feature Engineering Research:** Explore the impact of advanced feature engineering on model performance.

---

## 🚀 Model Performance Highlights

### Deep Learning Multi-Task Model (Latest Validation)

| Metric           | Value    |
|------------------|----------|
| Default Accuracy | **0.886**|
| Default AUC      | **0.904**|
| Interest MAE     | **1.68** |
| Interest MSE     | **4.04** |

*Latest validation: 2025-05-29. Demonstrates strong performance on both classification and regression tasks.*

### XGBoost Baseline (Latest Validation)

| Metric           | Value    |
|------------------|----------|
| Default Accuracy | 0.912    |
| Default AUC      | 0.946    |
| Interest MAE     | 10.70    |
| Interest MSE     | 123.27   |

*Latest validation: 2025-05-30. See `models/xgb_validation/` for full details and plots.*

---

## 🧠 Deep Learning Model Architecture

- **Shared Representation:** Dense layers with batch normalization and dropout extract common patterns from input features.
- **Task-Specific Heads:**  
  - **Default Head:** Outputs default probability (sigmoid activation).  
  - **Interest Rate Head:** Predicts interest rate (linear activation).
- **Multi-Task Loss:** Combines binary cross-entropy (default) and MSE (interest rate) for joint optimization.
- **Regularization:** Dropout and batch normalization prevent overfitting and stabilize training.
- **Training Best Practices:** Early stopping, checkpointing, and automatic generation of training/validation plots.

This architecture leverages shared information between tasks, often outperforming separate models. The code is modular and easy to extend for additional tasks or features.

---

## 📦 Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd multi-task-default-interest-model
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## ⚡ Usage

1. **Prepare your data:**
   - Place your raw and processed data in the `data/` directory as shown in the project structure.

2. **Run the deep learning multi-task model:**
   ```sh
   python src/models/DL_Multitask_Baseline.py
   ```

3. **Run the XGBoost baseline:**
   ```sh
   python src/models/XGBoost.py
   ```

4. **Outputs:**
   - All models, metrics, and plots will be saved in the `models/` and `model-results/` directories.

---

## 📝 Example

After running the deep learning pipeline, you’ll find:

- `models/dl_multitask_baseline/best_model.h5` — Trained Keras model
- `models/dl_multitask_baseline/dl_validation_metrics.json` — Validation metrics
- `models/dl_multitask_baseline/roc_curve_default.png` — ROC curve for default prediction
- `models/dl_multitask_baseline/int_rate_scatter.png` — Scatter plot for interest rate prediction
- `models/dl_multitask_baseline/training_curves.png` — Training/validation curves

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

## 🔍 Model Interpretability

- SHAP summary and waterfall plots for transparency.
- Feature importance is visualized and logged for both tasks.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas, spot bugs, or want to add features, please open an issue or submit a pull request. See the `docs/` folder for more information.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

Questions or want to collaborate?  
Open an issue or email the maintainer.  
Let’s build better credit risk models together!
