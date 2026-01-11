import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/creditcard.csv")

def validate_data(df):
    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum()
    }
    return report

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    report = validate_data(df)
    
    print("Validation Report")
    for k, v in report.items():
        print(f"{k}: {v}")
