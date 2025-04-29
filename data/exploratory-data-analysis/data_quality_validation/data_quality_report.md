# Data Quality Validation Report

## Outlier Detection (IQR method)
- **person_age**: 0.00% outliers outside [12.50, 40.50]. No major outliers.
- **person_income**: 0.00% outliers outside [-22550.00, 140250.00]. No major outliers.
- **person_emp_length**: 0.00% outliers outside [-5.50, 14.50]. No major outliers.
- **loan_amnt**: 0.00% outliers outside [-5875.00, 23125.00]. No major outliers.

## Distributional Skewness
- **person_income**: skewness=0.87. No transform required.
- **loan_amnt**: skewness=0.81. No transform required.

## Class Imbalance
- Class 0: 78.13%
- Class 1: 21.87%
Suggestion: Use stratified sampling or class weights in model training.

## Target Definition Validation
- **loan_status** unique values: [1, 0], counts: {0: 25322, 1: 7089}
- Confirm that 0 = paid, 1 = default/delinquency; map any extra values appropriately.
- **loan_int_rate** summary: {'count': 32411.0, 'mean': 11.014528400851562, 'std': 3.0832343714084507, 'min': 5.42, '25%': 8.49, '50%': 10.99, '75%': 13.11, 'max': 23.22}
- Decide whether to predict raw APR or classify into high/low bins (e.g., above/below median).