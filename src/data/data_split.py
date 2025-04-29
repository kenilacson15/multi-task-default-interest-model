import os
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\clean_data\final_credit_risk.csv"
OUTPUT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\multi-task-default-interest-model\data\train-validate-split"

os.makedirs(OUTPUT_DIR, exist_ok=True)


TRAIN_PATH = os.path.join(OUTPUT_DIR, 'train.csv')
VAL_PATH = os.path.join(OUTPUT_DIR, 'val.csv')
TEST_PATH = os.path.join(OUTPUT_DIR, 'test.csv')



print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

# Check if target column exists
if 'loan_status' not in df.columns:
    raise ValueError("Target column 'loan_status' not found in dataset!")



print("Performing stratified train/val/test split (70%/15%/15%)...")

train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df['loan_status']  # Preserve class distribution
)



train_df, val_df =train_test_split(
    train_val_df,
    test_size=0.18,  # 0.15 / 0.85 ≈ 0.18
    random_state=42,
    stratify=train_val_df['loan_status']


    
)



# --- Save Datasets ---
print(f"💾 Saving datasets to {OUTPUT_DIR}...")



train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)



print("\n Data split complete:")
print(f"Train size: {len(train_df)} rows")
print(f"Validation size: {len(val_df)} rows")
print(f"Test size: {len(test_df)} rows\n")

print(" Class distribution (loan_status):")
for name, subset in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    counts = subset['loan_status'].value_counts().to_dict()
    print(f"  - {name.upper()}: {counts}")