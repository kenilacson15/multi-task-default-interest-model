import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

def comprehensive_data_quality_check(file_path):
    # Load data with proper error handling
    try:
        df = pd.read_csv(Path(file_path))
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("✅ Data loaded successfully")
    print(f"Dataset shape: {df.shape}\n")

    # Initialize report
    quality_report = {
        'basic_info': {},
        'missing_data': {},
        'data_issues': {},
        'outliers': {},
        'categorical_checks': {},
        'domain_specific': {}
    }

    # Basic information
    quality_report['basic_info']['head'] = df.head()
    quality_report['basic_info']['dtypes'] = df.dtypes
    quality_report['basic_info']['describe'] = df.describe(include='all')

    # Missing values analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    quality_report['missing_data'] = pd.DataFrame({'missing_count': missing, 'missing_percentage': missing_pct})

    # Duplicates check
    duplicates = df.duplicated().sum()
    quality_report['data_issues']['duplicates'] = duplicates

    # Data type validation
    type_issues = {}
    expected_types = {
        'person_age': 'int64',
        'person_income': 'int64',
        'person_home_ownership': 'object',
        'person_emp_length': 'float64',  # Allow for NaN values
        'loan_intent': 'object',
        'loan_grade': 'object',
        'loan_amnt': 'int64',
        'loan_int_rate': 'float64',
        'loan_status': 'int64',
        'loan_percent_income': 'float64',
        'cb_person_default_on_file': 'object',
        'cb_person_cred_hist_length': 'int64'
    }
    
    for col, expected_type in expected_types.items():
        actual_type = str(df[col].dtype)
        if actual_type != expected_type:
            type_issues[col] = f"Expected {expected_type}, found {actual_type}"
    
    quality_report['data_issues']['dtype_issues'] = type_issues

    # Categorical checks
    categorical_checks = {}
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    
    for col in categorical_cols:
        categorical_checks[col] = {
            'unique_values': df[col].unique(),
            'value_counts': df[col].value_counts(dropna=False)
        }
    
    quality_report['categorical_checks'] = categorical_checks

    # Domain-specific validations
    domain_issues = {}
    
    # Age validation (reasonable range)
    domain_issues['invalid_ages'] = df[(df['person_age'] < 18) | (df['person_age'] > 100)]
    
    # Employment length vs age
    df['emp_length_valid'] = df['person_emp_length'] <= (df['person_age'] - 18)
    domain_issues['invalid_emp_length'] = df[~df['emp_length_valid']]
    
    # Loan percent income validation
    df['calculated_percent'] = df['loan_amnt'] / df['person_income']
    domain_issues['percent_income_mismatch'] = df[abs(df['calculated_percent'] - df['loan_percent_income']) > 0.01]
    
    quality_report['domain_specific'] = domain_issues

    # Outlier detection
    numerical_cols = ['person_age', 'person_income', 'person_emp_length', 
                     'loan_amnt', 'loan_int_rate', 'loan_percent_income']
    
    outlier_results = {}
    for col in numerical_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_results[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers)/len(df))*100,
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    quality_report['outliers'] = outlier_results

    # Generate visual report
    plt.figure(figsize=(15, 10))
    
    # Missing values visualization
    plt.subplot(2, 2, 1)
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Pattern')
    
    # Numerical distributions
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df[numerical_cols].select_dtypes(include='number'))
    plt.title('Numerical Features Distribution')
    plt.xticks(rotation=45)
    
    # Categorical distributions
    plt.subplot(2, 2, 3)
    df[categorical_cols].nunique().plot(kind='bar')
    plt.title('Unique Values in Categorical Features')
    
    # Save visual report
    plt.tight_layout()
    plt.savefig('data_quality_report.png')
    plt.close()

    # Print summary report
    print("\n=== DATA QUALITY REPORT SUMMARY ===")
    print(f"Total records: {len(df)}")
    print(f"Missing values: {missing.sum()} ({missing_pct.sum():.2f}% of total data)")
    print(f"Duplicate records: {duplicates}")
    
    print("\nKey Issues:")
    print(f"- Invalid ages: {len(domain_issues['invalid_ages'])} records")
    print(f"- Employment length inconsistencies: {len(domain_issues['invalid_emp_length'])} records")
    print(f"- Loan percent income mismatches: {len(domain_issues['percent_income_mismatch'])} records")
    
    print("\nTop Outlier Counts:")
    for col, results in outlier_results.items():
        print(f"- {col}: {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")

    print("\n⚠️ Full report available in variables and saved visual report")
    print("✅ Data quality check completed")

    return quality_report

# Execute the function
file_path = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\clean_data\cleaned_credit_risk.csv"
quality_report = comprehensive_data_quality_check(file_path)