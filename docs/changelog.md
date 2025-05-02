# 📅 Changelog - May 2, 2025

## 🗑️ Removed
### Legacy Components
- Old feature engineering script and data:
  - `src/features/feature_engineering_old.py`
  - `data/feature-engineer/processed_features_old.csv`
  - `reports/eda_old.ipynb`, `reports/data_quality_old.pdf`  
  *Reason:* Addressed quality issues (e.g., near-constant features like `person_home_ownership_OTHER`, improper binning) and streamlined analysis.

---

## 🛠️ Created
### 1. **New Feature Engineering Pipeline**
- **Key Additions:**
  - **Stability Score**: `person_emp_length * cb_person_cred_hist_length`  
    *Rationale:* Combines job stability and credit history for financial risk assessment.
  - **Interaction Features**: `age_income_interaction`, `risk_group`  
    *Rationale:* Captures demographic risk clusters and combined risk signals.
  - **Log Transformations**: Applied to skewed features (`person_income`, `loan_amnt`)  
    *Rationale:* Reduces skewness for better model convergence.
  - **Binning Strategy**:  
    - Income: `[low, medium, high]` (quantile-based)  
    - Loan Percent Income: `[low (<0.1), medium (0.1–0.3), high (>0.3)]`  
    *Rationale:* Turns continuous ratios into interpretable risk tiers.

### 2. **XGBoost Baseline Model**
- Replaced logistic regression with XGBoost for loan default prediction:
  ```python
  params = {
      'objective': 'binary:logistic',
      'eval_metric': 'auc',
      'tree_method': 'hist',
      'scale_pos_weight': 3.62,  # From class imbalance analysis
      'device': 'cpu'
  }
  ```
- **Performance Gains:**
  | Metric       | Previous (Logistic) | New (XGBoost) | Δ Change |
  |--------------|---------------------|----------------|----------|
  | **AUC**      | 0.8887              | **0.9407**     | +5.85%   |
  | **F1-Score** | 0.68                | **0.7976**     | +17.3%   |
  | **Recall**   | 0.78                | **0.8016**     | +2.76%   |

- **Technical Improvements:**
  - Early stopping with 5-fold stratified CV
  - Class imbalance handling via `scale_pos_weight`
  - Feature importance tracking for interpretability

---

## 🧹 Cleaned Up
### EDA & Data Reports
- Deleted outdated diagnostic files:
  - `reports/data_quality_report_old.ipynb`
  - `data/feature-engineer/legacy/`

- Kept essential documentation:
  - `docs/feature_documentation.md` (new)
  - `docs/model_card.md` (new)
  - `src/models/XGBoost.py` (new baseline implementation)

---

## 📈 Impact Summary
### Key Improvements
1. **Model Quality**  
   - AUC improved from **0.8887 → 0.9407** (+5.85% absolute gain).  
   - F1-score improved from **0.68 → 0.7976** (+17.3% absolute gain).  
   - Recall improved from **0.78 → 0.8016** (critical for catching defaults).

2. **Feature Engineering**  
   - Removed near-constant features (e.g., `risk_group_GradeF_Default`) that caused noise.  
   - Added domain-specific interactions (e.g., `stability_score`, `risk_group`).  
   - Improved numerical transformations (log1p, capping at 99th percentile).

3. **Code Maintainability**  
   - Reduced technical debt by removing legacy code.  
   - Structured feature pipeline for future multi-task extensions.

---

## 🚀 Next Steps
1. **Hyperparameter Tuning**  
   - Grid search over `max_depth`, `learning_rate`, and `subsample`.

2. **Multi-Task Learning**  
   - Extend XGBoost to predict both `loan_status` and `loan_int_rate` jointly.

3. **SHAP Interpretability**  
   - Add SHAP values for auditability and regulatory compliance.

4. **Drift Detection**  
   - Implement PSI/KL divergence for production monitoring.

