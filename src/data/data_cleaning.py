import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def configure_logging() -> None:
    """
    Configure the logging settings for the data cleaning script.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load the raw CSV dataset into a Pandas DataFrame.

    Args:
        input_path (Path): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Raw data loaded from CSV.
    """
    logging.info(f"Loading data from {input_path}")
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Successfully loaded {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame potentially containing duplicates.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates()
    duplicates_removed = initial_count - len(df_clean)
    logging.info(f"Removed {duplicates_removed} duplicate records")
    return df_clean


def handle_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle missing values by dropping columns or imputing.

    Args:
        df (pd.DataFrame): DataFrame with missing values.
        threshold (float): Fractional threshold for dropping a column if missing.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    total_records = len(df)

    # Drop columns with too many missing values
    missing_fraction = df.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
    if cols_to_drop:
        logging.info(f"Dropping columns with > {threshold*100:.0f}% missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Impute remaining missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logging.info(f"Imputed missing numeric values in '{col}' with median: {median_val}")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logging.info(f"Imputed missing categorical values in '{col}' with mode: {mode_val}")

    return df


def clean_age(df: pd.DataFrame, min_age: int = 18, max_age: int = 100) -> pd.DataFrame:
    """
    Clean invalid age values by capping or removing outliers.

    Args:
        df (pd.DataFrame): DataFrame containing 'person_age'.
        min_age (int): Minimum plausible age.
        max_age (int): Maximum plausible age.

    Returns:
        pd.DataFrame: DataFrame with 'person_age' cleaned.
    """
    # Identify invalid ages
    invalid_mask = (df['person_age'] < min_age) | (df['person_age'] > max_age)
    invalid_count = invalid_mask.sum()
    logging.info(f"Found {invalid_count} invalid ages outside [{min_age}, {max_age}]")

    # Remove rows with invalid ages
    df_clean = df.loc[~invalid_mask].copy()
    logging.info(f"Removed {invalid_count} records due to invalid age")
    return df_clean



def clean_employment_length(df: pd.DataFrame) -> pd.DataFrame:

    df['person_emp_length'] = pd.to_numeric(df['person_emp_length'], errors='coerce')
    invalid_emp = df['person_emp_length'] < 0
    count_invalid = invalid_emp.sum()
    if count_invalid > 0:
        logging.warning(f"Found and nullified {count_invalid} negative employment lengths")
    df.loc[invalid_emp, 'person_emp_length'] = np.nan


    # Impute missing employment lengths with median
    median_emp = df['person_emp_length'].median()
    df['person_emp_length'].fillna(median_emp, inplace=True)
    logging.info(f"Imputed missing/invalid employment length with median: {median_emp}")

    return df


def clean_loan_percent_income(df: pd.DataFrame, income_col: str = 'person_income') -> pd.DataFrame:
    """
    Recalculate and correct 'loan_percent_income' based on loan amount and income.

    Args:
        df (pd.DataFrame): DataFrame containing 'loan_amnt', income_col, and 'loan_percent_income'.
        income_col (str): Column name for person income.

    Returns:
        pd.DataFrame: DataFrame with 'loan_percent_income' corrected.
    """
    # Recompute and compare
    recalculated = df['loan_amnt'] / df[income_col]
    mismatches = (np.abs(recalculated - df['loan_percent_income']) > 1e-3)
    mismatch_count = mismatches.sum()
    logging.info(f"Found {mismatch_count} mismatched 'loan_percent_income' values")

    # Replace mismatches with recalculated values
    df.loc[mismatches, 'loan_percent_income'] = recalculated[mismatches]
    logging.info("Corrected mismatched 'loan_percent_income' entries")

    return df




def handle_outliers(
    df: pd.DataFrame,
    cols: Tuple[str, ...] = ('person_income', 'loan_amnt'),
    method: str = 'iqr',
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Cap outliers in specified columns using specified method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (Tuple[str, ...]): Columns to process for outlier capping.
        method (str): Method for outlier detection (currently supports 'iqr').
        factor (float): Multiplier for IQR method (default 1.5).

    Returns:
        pd.DataFrame: DataFrame with outliers capped at boundary values.
    """
    clean_df = df.copy()
    
    for col in cols:
        if col not in clean_df.columns:
            logging.warning(f"Column '{col}' not found in DataFrame. Skipping outlier capping.")
            continue

        if method == 'iqr':
            # Calculate IQR boundaries
            q1 = clean_df[col].quantile(0.25)
            q3 = clean_df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr

            # Count outliers
            below = (clean_df[col] < lower_bound).sum()
            above = (clean_df[col] > upper_bound).sum()
            total_outliers = below + above

            # Cap values
            clean_df[col] = clean_df[col].clip(lower=lower_bound, upper=upper_bound)

            logging.info(
                f"Capped {total_outliers} outliers in '{col}' using IQR method "
                f"(lower: {below}, upper: {above}, bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
            )
        else:
            logging.warning(f"Outlier method '{method}' not supported. Skipping column '{col}'.")

    return clean_df




def save_clean_data(df: pd.DataFrame, output_dir: Path, filename: str = 'cleaned_credit_risk.csv') -> None:
    """
    Save the cleaned DataFrame to a CSV in the specified directory.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        output_dir (Path): Directory to save the file.
        filename (str): Output CSV filename.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned data saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Clean raw credit risk dataset and output cleaned CSV."
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path(r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\raw\credit_risk_dataset.csv"),
        help="Path to raw CSV file."
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path(r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\clean_data"),
        help="Directory to save cleaned CSV."
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the data cleaning script.
    """
    configure_logging()
    args = parse_args()

    # Load raw data
    df = load_data(args.input)

    # Data cleaning pipeline
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = clean_age(df)
    df = clean_employment_length(df)
    df = clean_loan_percent_income(df)
    df = handle_outliers(df, cols=('person_income', 'loan_amnt'))

    # Save results
    save_clean_data(df, args.output)

    logging.info("Data cleaning pipeline completed successfully.")


if __name__ == '__main__':
    main()