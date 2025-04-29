import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    OrdinalEncoder,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import os

# File paths
input_path = r'C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\train-validate-split\train.csv'
output_path = r'C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\feature-engineer'
os.makedirs(output_path, exist_ok=True)

# Load data
df = pd.read_csv(input_path)

# Separate targets
y = df[['loan_status', 'loan_int_rate']]
X = df.drop(['loan_status', 'loan_int_rate'], axis=1)

class FeatureAugmentor(BaseEstimator, TransformerMixin):
    """Creates interaction terms and polynomial features"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        try:
            # Create interaction terms using ORIGINAL features
            X['income_emp_ratio'] = X['person_income'] / (X['person_emp_length'] + 1)
            X['loan_grade_emp'] = X['loan_grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}) * X['person_emp_length']
            X['default_cred_history'] = X['cb_person_default_on_file'].map({'Y':1, 'N':0}) * X['cb_person_cred_hist_length']
            
            # Polynomial features
            X['person_age_sq'] = X['person_age'] ** 2
            X['log_income'] = np.log1p(X['person_income'])  # Explicit log transform
            
            return X
        except KeyError as e:
            raise ValueError(f"Missing required column: {e}") from e

# Define preprocessing steps
preprocessor = ColumnTransformer([
    # Handle binary feature first
    ('binary', OrdinalEncoder(), ['cb_person_default_on_file']),
    
    # Ordinal encoding for loan grade
    ('ordinal', OrdinalEncoder(
        categories=[['A','B','C','D','E','F','G']]), ['loan_grade']),
    
    # One-hot encode categoricals
    ('onehot', OneHotEncoder(
        drop='first', sparse_output=False), 
        ['person_home_ownership', 'loan_intent']),
    
    # Scale numerical features
    ('numerical', Pipeline([
        ('log', FunctionTransformer(np.log1p)),
        ('scale', StandardScaler())
    ]), ['person_income', 'person_emp_length', 'loan_amnt',
         'loan_percent_income', 'cb_person_cred_hist_length',
         'income_emp_ratio', 'default_cred_history', 'person_age_sq'])
], remainder='passthrough')

# Build final pipeline with proper execution order
pipeline = Pipeline([
    ('feature_engineering', FeatureAugmentor()),  # Create new features first
    ('preprocessing', preprocessor)              # Then process all features
])

try:
    # Execute pipeline
    X_processed = pipeline.fit_transform(X)
    
    # Save processed data
    pd.DataFrame(X_processed).to_csv(
        os.path.join(output_path, 'processed_features.csv'), 
        index=False
    )
    y.to_csv(
        os.path.join(output_path, 'targets.csv'), 
        index=False
    )
    
except Exception as e:
    print(f"Pipeline failed: {str(e)}")
    print("Debugging info:")
    print("Current columns after feature engineering:", 
          pipeline.named_steps['feature_engineering'].transform(X).columns.tolist())