"""
API FastAPI de credit scoring.
Charge le pipeline exporté par train_export.py et expose le scoring unitaire et par lot,
avec décision, classe de risque et facteurs explicatifs.

Lancement :  uvicorn app:app --host 127.0.0.1 --port 8000
"""

import json
import os
import sys
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)  # pour que joblib retrouve le module features
from features import LABELS, add_features  # noqa: E402

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ── Chargement des artefacts ─────────────────────────────────────────────────
pipeline = joblib.load(os.path.join(ARTIFACTS_DIR, "pipeline.joblib"))
with open(os.path.join(ARTIFACTS_DIR, "config.json")) as f:
    config = json.load(f)
portfolio = pd.read_csv(os.path.join(ARTIFACTS_DIR, "portfolio_scored.csv"))

RAW_COLUMNS = config["raw_columns"]
THRESHOLD = config["threshold"]
REVIEW_THRESHOLD = config["review_factor"] * THRESHOLD
RISK_BINS = np.array(config["risk_bins"])
RISK_CLASSES = config["risk_classes"]

# ── Explicabilité : contribution de chaque variable au log-odds ──────────────
# contribution = coefficient × (valeur du demandeur − valeur moyenne du portefeuille),
# agrégée par variable d'origine (somme des colonnes one-hot d'une même variable).
_transform = pipeline[:-1]
_coefs = pipeline["model"].coef_[0]
_fe_columns = list(add_features(portfolio[RAW_COLUMNS].head(1)).columns)
_x_ref = _transform.transform(portfolio.loc[portfolio["split"] == "train", RAW_COLUMNS]).mean(axis=0)


def _source_column(output_name: str) -> str:
    name = output_name.split("__", 1)[1]
    return max((c for c in _fe_columns if name == c or name.startswith(c + "_")), key=len)


_sources = np.array([_source_column(n) for n in pipeline["pre"].get_feature_names_out()])


# ── Schéma d'entrée (valeurs par défaut = mode / médiane du portefeuille) ────
D = config["defaults"]


class Application(BaseModel):
    checking_status: str = Field(default=D["checking_status"], description="Solde du compte courant")
    duration: int = Field(default=D["duration"], ge=1, le=120, description="Durée (mois)")
    credit_history: str = Field(default=D["credit_history"], description="Historique de crédit")
    purpose: str = Field(default=D["purpose"], description="Objet du crédit")
    credit_amount: float = Field(default=D["credit_amount"], gt=0, description="Montant demandé (DM)")
    savings_status: str = Field(default=D["savings_status"], description="Épargne")
    employment: str = Field(default=D["employment"], description="Ancienneté dans l'emploi")
    installment_commitment: int = Field(default=D["installment_commitment"], ge=1, le=4, description="Taux d'effort (1-4)")
    personal_status: str = Field(default=D["personal_status"], description="Sexe & statut marital")
    other_parties: str = Field(default=D["other_parties"], description="Co-emprunteur / garant")
    residence_since: int = Field(default=D["residence_since"], ge=1, le=4, description="Ancienneté de résidence")
    property_magnitude: str = Field(default=D["property_magnitude"], description="Patrimoine")
    age: int = Field(default=D["age"], ge=18, le=100, description="Âge")
    other_payment_plans: str = Field(default=D["other_payment_plans"], description="Autres crédits en cours")
    housing: str = Field(default=D["housing"], description="Logement")
    existing_credits: int = Field(default=D["existing_credits"], ge=1, le=10, description="Crédits existants")
    job: str = Field(default=D["job"], description="Catégorie d'emploi")
    num_dependents: int = Field(default=D["num_dependents"], ge=0, le=10, description="Personnes à charge")
    own_telephone: str = Field(default=D["own_telephone"], description="Téléphone")
    foreign_worker: str = Field(default=D["foreign_worker"], description="Travailleur étranger")


app = FastAPI(
    title="Credit Scoring API",
    description="Probabilité de défaut, classe de risque, décision et facteurs explicatifs d'une demande de crédit.",
    version=config["version"],
)


def _validate(applications: List[Application]) -> pd.DataFrame:
    df = pd.DataFrame([a.model_dump() for a in applications])[RAW_COLUMNS]
    for col, allowed in config["categories"].items():
        bad = set(df[col]) - set(allowed)
        if bad:
            raise HTTPException(status_code=422, detail=f"{col} : valeur(s) inconnue(s) {sorted(bad)} — attendu : {allowed}")
    return df


def _decision(pd_score: float) -> str:
    if pd_score >= THRESHOLD:
        return "Refuser"
    if pd_score >= REVIEW_THRESHOLD:
        return "Revoir"
    return "Accepter"


def _risk_class(pd_score: float) -> str:
    return RISK_CLASSES[int(np.searchsorted(RISK_BINS[1:-1], pd_score, side="right"))]


def _factors(row: pd.DataFrame, top: int) -> list:
    contrib = _coefs * (_transform.transform(row)[0] - _x_ref)
    by_var = pd.Series(contrib).groupby(_sources).sum()
    by_var = by_var.reindex(by_var.abs().sort_values(ascending=False).index).head(top)
    enriched = add_features(row).iloc[0]
    return [{"variable": var, "libelle": LABELS.get(var, var),
             "valeur": enriched[var].item() if hasattr(enriched[var], "item") else enriched[var],
             "contribution": round(float(c), 4),
             "effet": "augmente le risque" if c > 0 else "diminue le risque"} for var, c in by_var.items()]


def _score(df: pd.DataFrame, with_factors: bool, top: int = 6) -> list:
    probas = pipeline.predict_proba(df)[:, 1]
    results = []
    for i, p in enumerate(probas):
        res = {"probabilite_defaut": round(float(p), 4), "score": int(round((1 - p) * 1000)),
               "classe_risque": _risk_class(p), "decision": _decision(p)}
        if with_factors:
            res["facteurs"] = _factors(df.iloc[[i]], top)
        results.append(res)
    return results


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_version": config["version"]}


@app.get("/model-info")
def model_info():
    keys = ["model_name", "version", "trained_at", "hyperparameters", "threshold", "review_factor",
            "cost_fn", "cost_fp", "risk_bins", "risk_classes", "metrics"]
    return {k: config[k] for k in keys} | {"review_threshold": REVIEW_THRESHOLD}


@app.get("/schema")
def schema():
    return {"categories": config["categories"], "defaults": config["defaults"],
            "numeric_ranges": config["numeric_ranges"], "labels": LABELS}


@app.post("/predict")
def predict(application: Application):
    df = _validate([application])
    return _score(df, with_factors=True)[0] | {"seuil_refus": THRESHOLD, "seuil_revue": REVIEW_THRESHOLD}


@app.post("/predict-batch")
def predict_batch(applications: List[Application]):
    if not applications:
        raise HTTPException(status_code=422, detail="Liste de demandes vide")
    if len(applications) > 10000:
        raise HTTPException(status_code=413, detail="10 000 demandes maximum par appel")
    df = _validate(applications)
    results = _score(df, with_factors=False)
    decisions = pd.Series([r["decision"] for r in results])
    return {
        "resultats": results,
        "synthese": {
            "nb_demandes": len(results),
            "pd_moyenne": round(float(np.mean([r["probabilite_defaut"] for r in results])), 4),
            "taux_acceptation": round(float((decisions == "Accepter").mean()), 4),
            "taux_revue": round(float((decisions == "Revoir").mean()), 4),
            "taux_refus": round(float((decisions == "Refuser").mean()), 4),
        },
    }
