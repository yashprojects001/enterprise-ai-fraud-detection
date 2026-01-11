import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/creditcard.csv")

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found in data/raw/")
    
    df = pd.read_csv(DATA_PATH)
    return df

if __name__ == "__main__":
    df = load_data()
    print("Data loaded successfully")
    print(df.head())
    print("\nShape:", df.shape)
