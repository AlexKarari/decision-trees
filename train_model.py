"""
train_model.py
==============
Production training script for the Decision Tree churn classifier.

What this script does:
-----------------------
1. Loads and cleans the Telco Customer Churn dataset
2. Encodes categorical features (binary map + one-hot encoding)
3. Trains a Decision Tree from scratch (no sklearn for the model itself)
4. Evaluates on the held-out test set with full metrics
5. Compares against sklearn's DecisionTreeClassifier (sanity check)
6. Saves the trained model artifact to models/churn_tree.pkl

Usage:
------
    python train_model.py

Expected output:
----------------
    Scratch Decision Tree:
      Accuracy  : 0.7793
      Precision : 0.5854
      Recall    : 0.5775
      F1        : 0.5814

    sklearn Decision Tree:
      Accuracy  : 0.7793   ← should be close (not exact — sklearn uses
      ...                      optimised C routines but same algorithm)

    Model saved to models/churn_tree.pkl

Requirements:
-------------
    pip install numpy pandas scikit-learn

Dataset:
--------
    Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    Place at: data/Telco-Customer-Churn.csv
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Make src/ importable when running from the root ──────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.decision_tree import DecisionTreeScratch


# =============================================================================
# Constants — mirrors Logistic Regression's encoding exactly so features are comparable
# =============================================================================

DATA_PATH  = "data/Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_tree.pkl"

# Columns with simple Yes/No or Male/Female values → encoded as 0/1
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
BINARY_MAP  = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

# Columns with 3+ categories → one-hot encoded (drop_first=True to avoid multicollinearity)
MULTI_CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# Decision tree hyperparameters (tune these to balance bias vs variance)
MAX_DEPTH         = 8    # Shallow enough to generalise, deep enough to learn patterns
MIN_SAMPLES_SPLIT = 20   # Don't split nodes with fewer than 20 samples
MIN_SAMPLES_LEAF  = 10   # Each leaf must contain at least 10 samples
RANDOM_STATE      = 42


# =============================================================================
# Data loading and cleaning
# =============================================================================

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load the raw Telco CSV and apply cleaning steps.

    Cleaning steps:
      1. Drop customerID (unique identifier, no predictive value)
      2. Coerce TotalCharges to numeric (11 rows have spaces → NaN → fill 0)
      3. Encode target: 'Yes' → 1, 'No' → 0

    Parameters
    ----------
    path : str — path to the CSV file

    Returns
    -------
    pd.DataFrame — cleaned dataframe
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nDataset not found at: {path}\n"
            "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "Then place it at: data/Telco-Customer-Churn.csv"
        )

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Drop ID column — not a feature
    df = df.drop("customerID", axis=1)

    # TotalCharges is stored as string — 11 rows have ' ' (space) instead of a number
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Encode target variable
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    churn_rate = df["Churn"].mean()
    print(f"Churn rate: {churn_rate:.1%}  ({df['Churn'].sum():,} churners / {len(df):,} total)")

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Encode all categorical features to numeric.

    Two strategies:
      - Binary columns (Yes/No, Male/Female) → simple 0/1 map
      - Multi-category columns               → one-hot encoding, drop first
                                               category to avoid dummy trap

    Parameters
    ----------
    df : pd.DataFrame — cleaned dataframe (Churn already encoded)

    Returns
    -------
    df_encoded   : pd.DataFrame — all-numeric dataframe
    feature_cols : list[str]    — list of feature column names (excludes 'Churn')
    """
    # Binary encoding
    for col in BINARY_COLS:
        df[col] = df[col].map(BINARY_MAP)

    # One-hot encoding for multi-category columns
    df_encoded = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True, dtype=int)

    feature_cols = [c for c in df_encoded.columns if c != "Churn"]

    print(f"Features after encoding: {len(feature_cols)}")
    return df_encoded, feature_cols


# =============================================================================
# Metrics helper
# =============================================================================

def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute and print classification metrics.

    Why these four metrics?
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Accuracy   → "Overall, how often is the model right?"
                 Misleading when classes are imbalanced (26% churn rate here).
                 A model that always predicts 'stay' gets 74% accuracy — useless.

    Precision  → "Of all the customers we predicted would churn,
                  what fraction actually did?"
                 HIGH precision = few false alarms.
                 Business impact: saves money on unnecessary retention offers.

    Recall     → "Of all customers who actually churned,
                  what fraction did we catch?"
                 HIGH recall = fewer missed churners.
                 Business impact: retaining a churner is worth ~3-5x the
                 cost of a retention offer, so missing churners is expensive.

    F1-Score   → Harmonic mean of Precision and Recall.
                 Use when you need ONE number to compare models and the
                 classes are imbalanced. Balances the precision/recall tradeoff.

    For churn prediction: RECALL is often the priority metric because
    missing a churner (false negative) costs more than a wasted offer
    (false positive). The choice depends on the business cost structure.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Parameters
    ----------
    name   : str        — model name for display
    y_true : np.ndarray — ground truth labels
    y_pred : np.ndarray — predicted labels

    Returns
    -------
    dict — metric name → value
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{name}")
    print(f"  {'Accuracy':<12}: {acc:.4f}")
    print(f"  {'Precision':<12}: {prec:.4f}")
    print(f"  {'Recall':<12}: {rec:.4f}")
    print(f"  {'F1-Score':<12}: {f1:.4f}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# =============================================================================
# Main training pipeline
# =============================================================================

def main():
    print("=" * 60)
    print("Day 7 — Decision Tree: Telco Customer Churn")
    print("=" * 60)

    # ── 1. Load and prepare data ──────────────────────────────────────
    df = load_and_clean(DATA_PATH)
    df_encoded, feature_cols = encode_features(df)

    X = df_encoded[feature_cols].values
    y = df_encoded["Churn"].values

    # Stratified split — preserves the 26% churn ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} samples  |  Test: {len(X_test):,} samples")

    # ── 2. Train the Decision Tree ───────────────────────────
    print("\nTraining Decision Tree from scratch...")
    tree = DecisionTreeScratch(
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    tree.fit(X_train, y_train)

    actual_depth = tree.get_depth()
    total_nodes  = tree.count_nodes()
    print(f"Tree built — actual depth: {actual_depth}, total nodes: {total_nodes}")

    # ── 3. Evaluate scratch tree ──────────────────────────────────────
    y_pred_train = tree.predict(X_train)
    y_pred_test  = tree.predict(X_test)

    train_metrics = print_metrics("Scratch Decision Tree (Train)", y_train, y_pred_train)
    test_metrics  = print_metrics("Scratch Decision Tree (Test)",  y_test,  y_pred_test)

    # ── 4. Sanity check vs sklearn ────────────────────────────────────
    print("\nTraining sklearn DecisionTreeClassifier (sanity check)...")
    sk_tree = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    sk_tree.fit(X_train, y_train)
    sk_pred = sk_tree.predict(X_test)
    print_metrics("sklearn Decision Tree (Test)", y_test, sk_pred)

    print("\nNote: Scratch and sklearn scores should be close.")
    print("Minor differences are expected due to tie-breaking in split selection.")

    # ── 5. Print top of the learned tree ─────────────────────────────
    print("\n" + "=" * 60)
    print("Learned Decision Rules (top 3 levels):")
    print("=" * 60)
    tree.print_tree(feature_names=feature_cols, max_print_depth=3)

    # ── 6. Top feature importances ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Top 10 Most Important Features:")
    print("=" * 60)
    importance_pairs = sorted(
        zip(feature_cols, tree.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for rank, (fname, importance) in enumerate(importance_pairs[:10], start=1):
        bar = "█" * int(importance * 100)
        print(f"  {rank:>2}. {fname:<45} {importance:.4f}  {bar}")

    # ── 7. Save artifact ─────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    artifact = {
        "model"               : tree,
        "feature_names"       : feature_cols,
        "max_depth"           : MAX_DEPTH,
        "train_metrics"       : train_metrics,
        "test_metrics"        : test_metrics,
        "feature_importances" : dict(zip(feature_cols, tree.feature_importances_.tolist())),
        "encoding_info"       : {
            "binary_cols"    : BINARY_COLS,
            "multi_cat_cols" : MULTI_CAT_COLS,
            "binary_map"     : BINARY_MAP,
        },
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nModel artifact saved → {MODEL_PATH}")
    print(f"Artifact keys: {list(artifact.keys())}")
    print("\nTraining complete ✓")


if __name__ == "__main__":
    main()