import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# --- Configuration ---
TEST_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\tests\data\test.csv"
OUTPUT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\tests\data\feature-engineer-data"
TARGET_COLUMNS = ['loan_status', 'loan_int_rate']

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading test data from: {TEST_PATH}")
df_test = pd.read_csv(TEST_PATH)

# --- Separate Features and Targets ---
targets = df_test[TARGET_COLUMNS].copy()
features = df_test.drop(columns=TARGET_COLUMNS)

# Store original numerical columns for potential interactions before transformation
original_emp_length = features['person_emp_length'].copy()
original_cred_hist = features['cb_person_cred_hist_length'].copy()
original_income = features['person_income'].copy()
original_loan_amnt = features['loan_amnt'].copy()

# --- Feature Engineering Pipeline (same as training) ---
features_eng = features.copy()

# 1. Numerical Feature Transformations
age_bins = [20, 25, 30, 35, 40, np.inf]
age_labels = ['20-25', '25-30', '30-35', '35-40', '40+']
features_eng['person_age_group'] = pd.cut(features_eng['person_age'], bins=age_bins, labels=age_labels, right=False)
features_eng['person_age_young_flag'] = (features_eng['person_age'] < 25).astype(int)

income_cap = original_income.quantile(0.99)
person_income_capped = np.clip(original_income, a_min=None, a_max=income_cap)
features_eng['person_income_capped'] = person_income_capped
features_eng['person_income_log'] = np.log1p(person_income_capped)
features_eng['person_income_bracket'] = pd.qcut(person_income_capped, q=3, labels=['low', 'medium', 'high'], duplicates='drop')

emp_bins = [-np.inf, 1, 5, 10, np.inf]
emp_labels = ['0-1', '1-5', '5-10', '>10']
features_eng['person_emp_length'] = features_eng['person_emp_length'].fillna(-1)
features_eng['person_emp_length_group'] = pd.cut(features_eng['person_emp_length'], bins=emp_bins, labels=emp_labels, right=True)

loan_amnt_cap = original_loan_amnt.quantile(0.99)
loan_amnt_capped = np.clip(original_loan_amnt, a_min=None, a_max=loan_amnt_cap)
features_eng['loan_amnt_capped'] = loan_amnt_capped
features_eng['loan_amnt_log'] = np.log1p(loan_amnt_capped)
features_eng['loan_amnt_quartile'] = pd.qcut(loan_amnt_capped, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

lpi_bins = [-np.inf, 0.1, 0.3, np.inf]
lpi_labels = ['low', 'medium', 'high']
features_eng['loan_percent_income_tier'] = pd.cut(features_eng['loan_percent_income'], bins=lpi_bins, labels=lpi_labels, right=False)

cred_hist_bins = [-np.inf, 2, 5, 10, np.inf]
cred_hist_labels = ['<2', '2-5', '5-10', '>10']
features_eng['cb_person_cred_hist_length_group'] = pd.cut(features_eng['cb_person_cred_hist_length'], bins=cred_hist_bins, labels=cred_hist_labels, right=False)

# 2. Categorical Feature Encoding
home_ownership_cats = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
features_eng['person_home_ownership'] = pd.Categorical(features_eng['person_home_ownership'], categories=home_ownership_cats, ordered=False)

grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
ordinal_encoder_grade = OrdinalEncoder(categories=[grade_order])
features_eng['loan_grade_encoded'] = ordinal_encoder_grade.fit_transform(features_eng[['loan_grade']])

features_eng['cb_person_default_on_file_encoded'] = features_eng['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
features_eng['cb_person_default_on_file_encoded'] = features_eng['cb_person_default_on_file_encoded'].fillna(-1)

# 3. Interaction & Domain-Specific Features
features_eng['stability_score'] = original_emp_length.fillna(0) * original_cred_hist.fillna(0)
features_eng['risk_group'] = (
    'Grade' + features_eng['loan_grade'].astype(str) + '_' +
    features_eng['cb_person_default_on_file'].map({'Y': 'Default', 'N': 'NoDefault'}).fillna('UnknownDefault')
)
features_eng['age_income_interaction'] = features_eng['person_age_group'].astype(str) + '_' + features_eng['person_income_bracket'].astype(str)

# 4. Final Processing (Encoding Remaining Categoricals & Scaling)
categorical_features_ohe = [
    'person_home_ownership', 'loan_intent', 'person_age_group', 'person_income_bracket',
    'person_emp_length_group', 'loan_amnt_quartile', 'loan_percent_income_tier',
    'cb_person_cred_hist_length_group', 'risk_group', 'age_income_interaction'
]
categorical_features_ohe = [col for col in categorical_features_ohe if col in features_eng.columns]
for col in categorical_features_ohe:
    features_eng[col] = features_eng[col].astype(str).fillna('Missing')

numerical_features_to_scale = [
    'person_age', 'person_income_log', 'loan_amnt_log',
    'loan_percent_income', 'cb_person_cred_hist_length',
    'stability_score', 'person_income_capped', 'loan_amnt_capped'
]
binary_ordinal_features = [
    'person_age_young_flag', 'loan_grade_encoded', 'cb_person_default_on_file_encoded'
]

# Load pre-fitted preprocessor from training (assumed saved as joblib)
preprocessor_path = os.path.join(os.path.dirname(__file__), '../../data/feature-engineer/preprocessor.joblib')
if os.path.exists(preprocessor_path):
    preprocessor = joblib.load(preprocessor_path)
    features_processed = preprocessor.transform(features_eng[numerical_features_to_scale + categorical_features_ohe + binary_ordinal_features])
    num_feature_names = numerical_features_to_scale
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features_ohe)
    remainder_feature_names = binary_ordinal_features
    final_feature_names = num_feature_names + list(ohe_feature_names) + remainder_feature_names
    features_final = pd.DataFrame(features_processed, columns=final_feature_names, index=features_eng.index)
else:
    print("WARNING: Preprocessor not found. Fitting new one (not recommended for test/val).")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_to_scale),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_ohe)
        ],
        remainder='passthrough'
    )
    features_processed = preprocessor.fit_transform(features_eng[numerical_features_to_scale + categorical_features_ohe + binary_ordinal_features])
    num_feature_names = numerical_features_to_scale
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features_ohe)
    remainder_feature_names = binary_ordinal_features
    final_feature_names = num_feature_names + list(ohe_feature_names) + remainder_feature_names
    features_final = pd.DataFrame(features_processed, columns=final_feature_names, index=features_eng.index)

# Remove Near-Constant Features (optional, use same columns as training for best results)
# Save Processed Data
features_output_path = os.path.join(OUTPUT_DIR, "processed_features_test.csv")
targets_output_path = os.path.join(OUTPUT_DIR, "targets_test.csv")

print(f"Saving processed features to: {features_output_path}")
features_final.to_csv(features_output_path, index=False)

print(f"Saving target variables to: {targets_output_path}")
targets.to_csv(targets_output_path, index=False)

print("\nTest feature engineering completed!")
