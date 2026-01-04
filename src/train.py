"""Train multiple models, tune them, select the best and save it.

Output:
  - model/model.bin (joblib) containing a sklearn Pipeline
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42


def load_data(path: str = "data/processed/energy_grid_daily.csv") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "grid_stress" not in df.columns:
        raise ValueError("Target column 'grid_stress' not found.")

    X = df.drop(columns=["grid_stress"])
    y = df["grid_stress"].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = ["country"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    return preprocessor


def tune_model(pipe: Pipeline, param_distributions: dict, X_train, y_train, n_iter: int = 12) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def main() -> None:
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X)

    # 1) Baseline: Logistic Regression
    logreg = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    logreg.fit(X_train, y_train)
    logreg_auc = roc_auc_score(y_val, logreg.predict_proba(X_val)[:, 1])
    print(f"LogReg ROC-AUC: {logreg_auc:.4f}")

    # 2) Random Forest with tuning
    rf_pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    rf_params = {
        "model__max_depth": [3, 5, 8, 12, None],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2", None],
    }
    rf_search = tune_model(rf_pipe, rf_params, X_train, y_train, n_iter=15)
    rf_best = rf_search.best_estimator_
    rf_auc = roc_auc_score(y_val, rf_best.predict_proba(X_val)[:, 1])
    print(f"RF tuned ROC-AUC: {rf_auc:.4f} | best_params={rf_search.best_params_}")

    # 3) XGBoost with tuning
    xgb_pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=600,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    # scale_pos_weight for imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = float(neg / max(pos, 1))

    xgb_params = {
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [0.0, 1.0, 5.0],
        "model__scale_pos_weight": [spw],
    }

    xgb_search = tune_model(xgb_pipe, xgb_params, X_train, y_train, n_iter=18)
    xgb_best = xgb_search.best_estimator_
    xgb_auc = roc_auc_score(y_val, xgb_best.predict_proba(X_val)[:, 1])
    print(f"XGB tuned ROC-AUC: {xgb_auc:.4f} | best_params={xgb_search.best_params_}")

    # Select best
    candidates = [("logreg", logreg, logreg_auc), ("rf", rf_best, rf_auc), ("xgb", xgb_best, xgb_auc)]
    best_name, best_model, best_auc = sorted(candidates, key=lambda x: x[2], reverse=True)[0]
    print(f"Selected best model: {best_name} (ROC-AUC={best_auc:.4f})")

    # Refit on full data for production artifact
    best_model.fit(X, y)

    Path("model").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, "model/model.bin")
    print("Saved model to model/model.bin")


if __name__ == "__main__":
    main()
