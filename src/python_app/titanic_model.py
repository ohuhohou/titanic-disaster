"""
Titanic survival prediction - Python version.

Loads src/data/train.csv and src/data/test.csv, cleans and engineers features,
fits a logistic regression, reports accuracy, and writes test-set predictions.
Every step prints what it changed so the full workflow is visible in the terminal.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Data folder: defaults to src/data relative to this file; can be overridden with DATA_DIR.
DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
LABELS_PATH = os.path.join(DATA_DIR, "test_labels.csv")  # optional, see README
OUTPUT_PATH = os.path.join(DATA_DIR, "predictions_python.csv")

FEATURES = ["Pclass", "Sex_male", "Age", "Fare_log", "FamilySize", "IsAlone",
            "HasCabin", "Embarked_Q", "Embarked_S",
            "Title_Miss", "Title_Mrs", "Title_Master", "Title_Rare"]
RANDOM_STATE = 42


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def load_csv(path, name):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {name} not found at {path}. "
                 "Download it from Kaggle into src/data/ (see README).")
    df = pd.read_csv(path)
    print(f"Loaded {name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def extract_title(names):
    """Pull the honorific out of 'Surname, Title. Given names' and group rare ones."""
    title = names.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")


def fit_cleaning_params(train):
    """Learn imputation values from the TRAINING data only (no leakage into test)."""
    titles = extract_title(train["Name"])
    return {
        "age_by_title": train.groupby(titles)["Age"].median().to_dict(),
        "age_overall": train["Age"].median(),
        "fare_median": train["Fare"].median(),
        "embarked_mode": train["Embarked"].mode()[0],
    }


def prepare(df, params, name):
    """Clean + engineer features, printing every change."""
    header(f"Q14 - Adjusting the {name} data")
    df = df.copy()

    print(f"Missing values before cleaning ({name}):")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())

    df["Title"] = extract_title(df["Name"])
    print(f"\n[+] Added 'Title' from Name. Counts:\n{df['Title'].value_counts().to_string()}")

    n = df["Age"].isna().sum()
    df["Age"] = df["Age"].fillna(df["Title"].map(params["age_by_title"])).fillna(params["age_overall"])
    print(f"\n[~] Filled {n} missing Age values with the training median age for each Title "
          f"{ {k: round(v, 1) for k, v in params['age_by_title'].items()} }")

    n = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(params["fare_median"])
    print(f"[~] Filled {n} missing Fare values with training median {params['fare_median']:.2f}")

    n = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(params["embarked_mode"])
    print(f"[~] Filled {n} missing Embarked values with training mode '{params['embarked_mode']}'")

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    print(f"[+] Added 'HasCabin' (Cabin is {df['Cabin'].isna().mean():.0%} missing, so keep only "
          f"whether a cabin was recorded). Share with cabin: {df['HasCabin'].mean():.2f}")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"[+] Added 'FamilySize' = SibSp + Parch + 1 (mean {df['FamilySize'].mean():.2f}) "
          f"and 'IsAlone' (share alone {df['IsAlone'].mean():.2f})")

    df["Fare_log"] = np.log1p(df["Fare"])
    print("[+] Added 'Fare_log' = log(1 + Fare) to reduce the heavy right skew of Fare")

    df["Sex_male"] = (df["Sex"] == "male").astype(int)
    for port in ["Q", "S"]:
        df[f"Embarked_{port}"] = (df["Embarked"] == port).astype(int)
    for t in ["Miss", "Mrs", "Master", "Rare"]:
        df[f"Title_{t}"] = (df["Title"] == t).astype(int)
    print("[+] One-hot encoded Sex (base = female), Embarked (base = C), Title (base = Mr)")

    dropped = ["Name", "Ticket", "Cabin", "SibSp", "Parch", "Fare", "Sex", "Embarked", "Title"]
    print(f"[-] Not used as model inputs: {dropped} (replaced by engineered columns or "
          "free-text IDs with no predictive structure)")
    print(f"Final feature set ({len(FEATURES)}): {FEATURES}")
    print(f"Missing values remaining in features: {int(df[FEATURES].isna().sum().sum())}")
    return df


def main():
    header("Q13 - Loading training data")
    train_raw = load_csv(TRAIN_PATH, "train.csv")
    print(train_raw.head().to_string())
    print(f"\nOverall survival rate in train: {train_raw['Survived'].mean():.3f}")
    print("Survival rate by Sex:\n" + train_raw.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("Survival rate by Pclass:\n" + train_raw.groupby("Pclass")["Survived"].mean().round(3).to_string())

    params = fit_cleaning_params(train_raw)
    train = prepare(train_raw, params, "training")
    X, y = train[FEATURES], train["Survived"]

    header("Q15 - Logistic regression model")
    # Hold out 20% of train to get an honest out-of-sample accuracy estimate.
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y,
                                                random_state=RANDOM_STATE)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_tr, y_tr)
    val_acc = accuracy_score(y_val, model.predict(X_val))

    # Refit on all training rows for the final model.
    model.fit(X, y)
    coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
    print("Model: StandardScaler -> LogisticRegression(max_iter=1000)")
    print("Standardised coefficients (positive = higher survival odds):")
    print(coefs.round(3).to_string())

    header("Q16 - Accuracy on the training set")
    train_acc = accuracy_score(y, model.predict(X))
    print(f"Training accuracy (fit and scored on all {len(y)} rows): {train_acc:.4f}")
    print(f"Hold-out validation accuracy (fit on 80%, scored on unseen 20%): {val_acc:.4f}")

    header("Q17 - Predicting on the test set")
    test_raw = load_csv(TEST_PATH, "test.csv")
    test = prepare(test_raw, params, "test")
    header("Q17 - Test predictions")
    test["Survived"] = model.predict(test[FEATURES])
    test[["PassengerId", "Survived"]].to_csv(OUTPUT_PATH, index=False)
    print(f"Predicted survivors: {test['Survived'].sum()} of {len(test)} "
          f"({test['Survived'].mean():.3f})")
    print(f"Saved predictions to {os.path.relpath(OUTPUT_PATH)}")
    print(test[["PassengerId", "Name", "Sex", "Pclass", "Survived"]].head(10).to_string())

    header("Q18 - Accuracy on the test set")
    if os.path.exists(LABELS_PATH):
        labels = pd.read_csv(LABELS_PATH)[["PassengerId", "Survived"]]
        merged = test[["PassengerId", "Survived"]].merge(labels, on="PassengerId",
                                                         suffixes=("_pred", "_true"))
        print(f"Test accuracy against {os.path.basename(LABELS_PATH)} "
              f"({len(merged)} rows): {accuracy_score(merged['Survived_true'], merged['Survived_pred']):.4f}")
    else:
        print("Kaggle's test.csv has NO 'Survived' column - the true labels are hidden, so test "
              "accuracy cannot be computed locally.")
        print(f"Best local estimate of test accuracy = hold-out validation accuracy: {val_acc:.4f}")
        print("To get the real test accuracy, upload predictions_python.csv to the Kaggle "
              "competition, or place a labelled test_labels.csv (PassengerId,Survived) in src/data/.")

    header("Done")


if __name__ == "__main__":
    main()
