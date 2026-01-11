import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DATA_PATH = Path("data/processed/creditcard_processed.csv")

def build_features():
    df = pd.read_csv(RAW_DATA_PATH)

    # Scale Amount column
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    # Drop original Amount column
    df = df.drop(columns=["Amount"])

    return df

if __name__ == "__main__":
    df_processed = build_features()
    
    # Save processed dataset
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print("Feature engineering completed successfully")
    print("Processed shape:", df_processed.shape)
