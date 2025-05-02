# Feature Engineering Documentation  
**For: Multi-Task Default-Interest Model**

---

## 📌 Overview  
This document details the **feature engineering pipeline** developed for the credit risk modeling project. The process transforms raw data into a robust feature set suitable for multi-task learning (loan default prediction + interest rate estimation).  

The pipeline includes:  
- Numerical transformations for skewed distributions  
- Domain-driven interaction features  
- Categorical encoding strategies  
- Automated removal of near-constant features  
- Full preprocessing for model readiness  

---

## 🎯 Objectives  
1. **Handle skewness** in income, loan amount, and stability metrics  
2. **Create interaction features** that capture financial risk patterns  
3. **Standardize categorical encodings** for consistency  
4. **Remove low-value features** (e.g., >99% constant columns)  
5. **Produce model-ready data** with proper scaling and encoding  

---

## 🧱 Feature Engineering Plan  

### 1. **Numerical Feature Transformations**  
| Feature | Transformation | Rationale |
|--------|----------------|-----------|
| `person_age` | Binning + Binary Flag<br>`[20, 25, 30, 35, 40, ∞]` → `['20-25', '25-30', ..., '40+']`<br>`person_age_young_flag = <25` | Captures life stage risk patterns |
| `person_income` | Log1p + Capping at 99th percentile<br>Quantile binning (3 bins) | Reduces skewness, creates income tiers |
| `person_emp_length` | Binning<br>`[-∞, 1, 5, 10, ∞]` → `['0-1', '1-5', '5-10', '>10']` | Reflects job stability levels |
| `loan_amnt` | Log1p + Capping at 99th percentile<br>Quartile binning | Standardizes loan sizes |
| `loan_percent_income` | Binning<br>`[-∞, 0.1, 0.3, ∞]` → `['low', 'medium', 'high']` | Debt-to-income ratio proxy |
| `cb_person_cred_hist_length` | Binning<br>`[-∞, 2, 5, 10, ∞]` → `['<2', '2-5', '5-10', '>10']` | Credit history duration analysis |

### 2. **Categorical Encoding**  
| Feature | Method | Notes |
|--------|--------|-------|
| `person_home_ownership` | One-Hot Encoding | Categories: RENT, MORTGAGE, OWN, OTHER |
| `loan_intent` | One-Hot Encoding | Captures loan purpose risk patterns |
| `loan_grade` | Ordinal Encoding | A=0, B=1, ..., G=6 |
| `cb_person_default_on_file` | Binary Mapping | Y=1, N=0 |

### 3. **Interaction & Domain Features**  
| Feature | Construction | Purpose |
|--------|--------------|---------|
| `stability_score` | `person_emp_length * cb_person_cred_hist_length` | Combined job + credit stability metric |
| `risk_group` | `Grade{loan_grade}_{Default/NoDefault}` | Loan grade + default history combination |
| `age_income_interaction` | `{age_group}_{income_bracket}` | Demographic risk clusters |

---

## 📋 Generated Features  
**Total Features After Engineering:** 65  
**Final Features After Cleaning:** 60 (removed 5 near-constant features)

### 🔢 Numerical Features  
- `person_age`  
- `person_income_log` (log1p of capped income)  
- `loan_amnt_log` (log1p of capped loan amount)  
- `loan_percent_income`  
- `cb_person_cred_hist_length`  
- `stability_score` (engineered feature)  
- `person_income_capped`  
- `loan_amnt_capped`  

### 📐 Scaled Features  
All numerical features above were standardized using `StandardScaler()` for model compatibility.

### 🔤 Categorical Features (One-Hot Encoded)  
```python
[
    'person_home_ownership',
    'loan_intent',
    'person_age_group',
    'person_income_bracket',
    'person_emp_length_group',
    'loan_amnt_quartile',
    'loan_percent_income_tier',
    'cb_person_cred_hist_length_group',
    'risk_group',
    'age_income_interaction'
]
```

### 🧮 Binary/Ordinal Features  
- `person_age_young_flag` (binary: 1 if <25)  
- `loan_grade_encoded` (ordinal: A-G → 0-6)  
- `cb_person_default_on_file_encoded` (Y/N → 1/0)  

---

## 🛠️ Implementation Highlights  

### 1. **Skewness Mitigation**  
- Applied `np.log1p()` to highly skewed features (`person_income`, `loan_amnt`)  
- Capped outliers at 99th percentile to reduce extreme value impact  

### 2. **Missing Value Handling**  
- For numerical:  
  ```python
  # Example for employment length
  features_eng['person_emp_length'].fillna(-1, inplace=True)
  ```
- For categorical:  
  ```python
  features_eng[col].fillna('Missing', inplace=True)
  ```

### 3. **ColumnTransformer Pipeline**  
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_to_scale),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_ohe)
    ],
    remainder='passthrough'
)
```

### 4. **Near-Constant Feature Removal**  
Automatically removes features where a single value appears in >99% of rows:  
```python
# Example output from logs:
Removing 5 near-constant features:
 - person_home_ownership_OTHER: 99.69% constant (0.0)
 - risk_group_GradeF_Default: 99.62% constant (0.0)
 - risk_group_GradeF_NoDefault: 99.59% constant (0.0)
 - risk_group_GradeG_Default: 99.88% constant (0.0)
 - risk_group_GradeG_NoDefault: 99.91% constant (0.0)
```

---

## 📁 Output Files  
### 1. **Processed Data**  
- `processed_features.csv`: Final feature matrix (60 cleaned features)  
- `targets.csv`: Original targets (`loan_status`, `loan_int_rate`)  

### 2. **Preprocessor Object** *(commented out but available)*  
- `preprocessor.joblib`: Scaler/encoder pipeline for deployment consistency  

---

## 📁 Directory Structure  
```
data/
└── feature-engineer/
    ├── processed_features.csv  # Engineered features
    ├── targets.csv            # Original targets
    └── preprocessor.joblib   # [Optional] Pipeline for inference
```

---

### Top 10 Features by Gain:
| Rank | Feature Name | Importance Score |
|------|--------------|------------------|
| 1    | person_home_ownership_RENT | 87.58 |
| 2    | loan_grade_encoded | 59.61 |
| 3    | loan_percent_income | 37.03 |
| 4    | loan_intent_DEBTCONSOLIDATION | 29.84 |
| 5    | loan_intent_MEDICAL | 22.98 |
| 6    | loan_intent_HOMEIMPROVEMENT | 16.82 |
| 7    | person_home_ownership_OWN | 13.82 |
| 8    | loan_intent_VENTURE | 12.64 |
| 9    | person_emp_length_group_0-1 | 11.89 |
| 10   | person_income_log | 11.58 |

---

## 📝 Key Engineering Decisions  

### 1. **Why Log1p Instead of Log?**  
- Handles zero values gracefully  
- Better for right-skewed distributions (income, loan amount)  

### 2. **Why Use Binning?**  
- Turns continuous values into risk tiers (e.g., income brackets)  
- Improves interpretability and handles non-linear relationships  

### 3. **Interaction Features Justification**  
- `stability_score` combines job stability + credit history  
- `risk_group` captures combined grade/default history signals  
- `age_income_interaction` identifies high-risk demographic clusters  

---

## ✅ Data Quality Improvements  

| Issue | Fix | Impact |
|------|-----|--------|
| Skewness | Log1p + capping | Better gradient flow in models |
| Class Imbalance | `scale_pos_weight` in XGBoost | Improved recall for rare defaults |
| Near-Constant Features | Automated removal | Reduced noise (removed 5 low-value features) |

---

## 🧪 Next Steps  
1. **Modeling Enhancements**  
   - Add SHAP values for interpretability  
   - Implement multi-task learning for joint prediction  
   - Add drift detection (PSI, KL divergence)  

2. **Pipeline Improvements**  
   - Save `ColumnTransformer` for production use  
   - Add Great Expectations for data validation  
   - Track feature lineage in DVC  

3. **Documentation**  
   - Add feature importance plots  
   - Create feature descriptions in `feature_documentation.md`  
   - Update model card with XGBoost metrics  

---

## 🧰 Tools Used  
| Tool | Purpose |
|------|---------|
| `pandas` | Data manipulation |
| `sklearn.compose.ColumnTransformer` | Unified preprocessing pipeline |
| `sklearn.preprocessing.OneHotEncoder` | Categorical encoding |
| `sklearn.preprocessing.StandardScaler` | Feature standardization |
| `numpy` | Numerical operations |

