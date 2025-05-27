"""
Feature engineering script for XGBoost: uses only the top 30 features by gain.
Input: processed_features.csv (from your main feature engineering pipeline)
Output: processed_features_xgb_top30.csv (optimized for XGBoost)
"""
import pandas as pd
import os

# Top 30 features by XGBoost gain
TOP_XGB_FEATURES = [
    'loan_grade_encoded',
    'person_home_ownership_RENT',
    'loan_percent_income',
    'person_home_ownership_OWN',
    'risk_group_GradeA_NoDefault',
    'loan_intent_VENTURE',
    'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL',
    'loan_intent_DEBTCONSOLIDATION',
    'person_income_log',
    'loan_amnt_quartile_Q1',
    'person_emp_length_group_0-1',
    'person_home_ownership_MORTGAGE',
    'person_age',
    'risk_group_GradeB_NoDefault',
    'loan_intent_EDUCATION',
    'person_age_group_35-40',
    'age_income_interaction_20-25_medium',
    'stability_score',
    'age_income_interaction_40+_medium',
    'age_income_interaction_25-30_medium',
    'loan_percent_income_tier_medium',
    'loan_amnt_quartile_Q3',
    'loan_amnt_quartile_Q2',
    'person_emp_length_group_>10',
    'age_income_interaction_30-35_medium',
    'age_income_interaction_25-30_low',
    'cb_person_cred_hist_length_group_5-10',
    'person_age_group_25-30',
    'risk_group_GradeC_Default',
]

INPUT_PATH = os.path.join('..', '..', 'data', 'feature-engineer', 'processed_features.csv')
OUTPUT_PATH = os.path.join('..', '..', 'data', 'feature-engineer', 'processed_features_xgb_top30.csv')

def main():
    print(f"Loading features from {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    missing = [f for f in TOP_XGB_FEATURES if f not in df.columns]
    if missing:
        print(f"Warning: The following features are missing in the input and will be skipped: {missing}")
    selected = [f for f in TOP_XGB_FEATURES if f in df.columns]
    df_xgb = df[selected]
    print(f"Saving top {len(selected)} XGBoost features to {OUTPUT_PATH}")
    df_xgb.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
