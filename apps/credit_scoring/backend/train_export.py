"""
Entraîne le modèle de credit scoring retenu dans le Lab 1 (régression logistique régularisée
+ feature engineering) et exporte les artefacts utilisés par l'API et le dashboard.

Usage :  python train_export.py
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import add_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "labs", "data", "german_credit.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

RANDOM_STATE = 42
COST_FN, COST_FP = 5, 1      # matrice de coût UCI : accepter un mauvais payeur coûte 5x plus
REVIEW_FACTOR = 0.75         # zone grise = [75 % du seuil ; seuil[ → étude manuelle
RISK_CLASSES = ["A", "B", "C", "D", "E"]


def make_pipeline(model):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.02, sparse_output=False),
         make_column_selector(dtype_include="object")),
    ])
    return Pipeline([("fe", FunctionTransformer(add_features)), ("pre", preprocessor), ("model", model)])


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns="default"), df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ── Fine-tuning (même grille que le Lab 1) ───────────────────────────────
    search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=3000, solver="liblinear")),
        {"model__C": [0.005, 0.01, 0.03, 0.1, 0.3, 1, 3], "model__penalty": ["l1", "l2"],
         "model__class_weight": [None, "balanced"]},
        cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    pipeline = search.best_estimator_
    print(f"Meilleurs hyperparamètres : {search.best_params_}  —  AUC CV {search.best_score_:.3f}")

    # ── Seuil optimal par coût métier, sur prédictions hors-pli du train ─────
    oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []
    for t in thresholds:
        pred = oof >= t
        costs.append(COST_FN * int(((~pred) & (y_train == 1)).sum()) + COST_FP * int((pred & (y_train == 0)).sum()))
    threshold = float(thresholds[int(np.argmin(costs))])

    # ── Grille de risque A → E (quintiles de PD hors-pli) ────────────────────
    _, bins = pd.qcut(oof, q=5, retbins=True)
    bins[0], bins[-1] = 0.0, 1.0

    # ── Évaluation sur le test ───────────────────────────────────────────────
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    auc = roc_auc_score(y_test, proba_test)
    tn, fp, fn, tp = confusion_matrix(y_test, (proba_test >= threshold).astype(int)).ravel()
    metrics = {
        "auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr)),
        "pr_auc": average_precision_score(y_test, proba_test), "auc_cv": search.best_score_,
        "confusion_at_threshold": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print(f"Test : AUC {auc:.3f} | Gini {2 * auc - 1:.3f} | KS {metrics['ks']:.3f} | seuil {threshold:.2f}")

    # ── Export ───────────────────────────────────────────────────────────────
    joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, "pipeline.joblib"))

    cat_cols = X.select_dtypes("object").columns
    num_cols = X.select_dtypes("number").columns
    config = {
        "model_name": "Régression logistique régularisée + feature engineering",
        "version": datetime.now().strftime("%Y.%m.%d"),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "hyperparameters": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "threshold": threshold,
        "review_factor": REVIEW_FACTOR,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "risk_bins": [float(b) for b in bins],
        "risk_classes": RISK_CLASSES,
        "metrics": {k: (float(v) if not isinstance(v, dict) else v) for k, v in metrics.items()},
        "raw_columns": list(X.columns),
        "numeric_columns": list(num_cols),
        "categories": {c: sorted(X[c].unique().tolist()) for c in cat_cols},
        "defaults": {**{c: X[c].mode()[0] for c in cat_cols}, **{c: int(X[c].median()) for c in num_cols}},
        "numeric_ranges": {c: [int(X[c].min()), int(X[c].max())] for c in num_cols},
    }
    with open(os.path.join(ARTIFACTS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Portefeuille scoré (train : PD hors-pli, test : PD du modèle) pour le dashboard et le monitoring
    portfolio = pd.concat([X_train.assign(default=y_train, split="train", pd_score=oof),
                           X_test.assign(default=y_test, split="test", pd_score=proba_test)])
    portfolio["risk_class"] = pd.cut(portfolio["pd_score"], bins=bins, labels=RISK_CLASSES,
                                     include_lowest=True).astype(str)
    portfolio.sort_index().to_csv(os.path.join(ARTIFACTS_DIR, "portfolio_scored.csv"), index_label="id")

    # Fichier exemple pour le scoring par lot
    X_test.head(50).to_csv(os.path.join(ARTIFACTS_DIR, "sample_applications.csv"), index=False)
    print(f"Artefacts exportés dans {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
