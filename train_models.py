import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

DATA_PATH = Path("data/processed/creditcard_processed.csv")
MODEL_DIR = Path("src/models/saved_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_models():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Baseline Model
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict_proba(X_test)[:, 1]

    lr_auc = roc_auc_score(y_test, lr_preds)

    print("\nLogistic Regression ROC-AUC:", lr_auc)
    print(classification_report(y_test, lr.predict(X_test)))

    # Advanced Model
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_preds)

    print("\nXGBoost ROC-AUC:", xgb_auc)
    print(classification_report(y_test, xgb.predict(X_test)))

    # Save models
    joblib.dump(lr, MODEL_DIR / "logistic_regression.pkl")
    joblib.dump(xgb, MODEL_DIR / "xgboost_model.pkl")

    print("\nModels saved successfully")

if __name__ == "__main__":
    train_models()
