import pandas as pd
import numpy as np
from pathlib import Path
import logging

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- File Paths ---
DATA_PATH = Path(r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\clean_data\cleaned_credit_risk.csv")
OUTPUT_PATH = Path(r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\clean_data\capped_credit_risk.csv")


CAP_SETTINGS = {
    "person_age": (12.5, 40.5),
    "person_emp_length": (-5.5, 14.5),
    
}

# --- Load Data ---
def load_data(path: Path) -> pd.DataFrame:
    logging.info(f"Loading data from {path}")
    return pd.read_csv(path)

# --- Cap Outliers Function ---
def cap_outliers(df: pd.DataFrame, caps: dict) -> pd.DataFrame:
    df_capped = df.copy()
    for col, (lower, upper) in caps.items():
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found in dataset. Skipping.")
            continue
        
        num_below = (df_capped[col] < lower).sum()
        num_above = (df_capped[col] > upper).sum()

        logging.info(f"Capping column '{col}': {num_below} values below {lower}, {num_above} values above {upper}")
        df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
        
    return df_capped

# --- Save Data ---
def save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Capped data saved to {path}")

# --- Main Process ---
def main():
    df = load_data(DATA_PATH)
    df_capped = cap_outliers(df, CAP_SETTINGS)
    save_data(df_capped, OUTPUT_PATH)
    logging.info("Outlier capping completed successfully.")

if __name__ == "__main__":
    main()
