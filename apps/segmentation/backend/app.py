"""
API FastAPI de segmentation client.
Charge le modèle K-Prototypes exporté par train_export.py et affecte un client (ou un lot)
à un segment, avec sa persona, ses offres recommandées et sa proximité aux segments.

Lancement :  uvicorn app:app --host 127.0.0.1 --port 8001
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
sys.path.insert(0, BASE_DIR)
from features import CAT_COLS, CAT_IDX, NUM_COLS, SEG_CAT, SEG_NUM, add_features, to_kproto_matrix  # noqa: E402

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ── Chargement des artefacts ─────────────────────────────────────────────────
model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
num_pipe, kproto = model["num_pipe"], model["kproto"]
with open(os.path.join(ARTIFACTS_DIR, "segments.json")) as f:
    meta = json.load(f)

SEGMENTS = meta["segments"]
CENTROIDS_NUM = kproto.cluster_centroids_[:, :len(SEG_NUM)].astype(float)
CENTROIDS_CAT = kproto.cluster_centroids_[:, len(SEG_NUM):]
GAMMA = kproto.gamma

# ── Schéma d'entrée ──────────────────────────────────────────────────────────
D = meta["defaults"]


class Client(BaseModel):
    age: int = Field(default=D["age"], ge=18, le=100, description="Âge")
    balance: float = Field(default=D["balance"], description="Solde moyen annuel (€)")
    job: str = Field(default=D["job"], description="Profession")
    marital: str = Field(default=D["marital"], description="Statut marital")
    education: str = Field(default=D["education"], description="Niveau d'études")
    default: str = Field(default=D["default"], description="Incident de crédit (yes/no)")
    housing: str = Field(default=D["housing"], description="Crédit immobilier (yes/no)")
    loan: str = Field(default=D["loan"], description="Crédit personnel (yes/no)")


app = FastAPI(
    title="Segmentation Client API",
    description="Affectation d'un client à un segment (K-Prototypes), persona et offres recommandées.",
    version=meta["version"],
)


def _validate(clients: List[Client]) -> pd.DataFrame:
    df = pd.DataFrame([c.model_dump() for c in clients])[NUM_COLS + CAT_COLS]
    for col, allowed in meta["categories"].items():
        bad = set(df[col]) - set(allowed)
        if bad:
            raise HTTPException(status_code=422, detail=f"{col} : valeur(s) inconnue(s) {sorted(bad)} — attendu : {allowed}")
    return add_features(df)


def _distances(X: np.ndarray) -> np.ndarray:
    # Distance K-Prototypes à chaque prototype : euclidienne² (numériques) + γ × nb de modalités différentes.
    num = X[:, :len(SEG_NUM)].astype(float)
    cat = X[:, len(SEG_NUM):]
    d_num = ((num[:, None, :] - CENTROIDS_NUM[None, :, :]) ** 2).sum(axis=2)
    d_cat = (cat[:, None, :] != CENTROIDS_CAT[None, :, :]).sum(axis=2)
    return d_num + GAMMA * d_cat


def _assign(df: pd.DataFrame, detailed: bool) -> list:
    X = to_kproto_matrix(df, num_pipe)
    segs = kproto.predict(X, categorical=CAT_IDX)
    dist = _distances(X)
    results = []
    for i, seg in enumerate(segs):
        info = SEGMENTS[str(seg)]
        res = {"segment": int(seg), "persona": info["persona"], "distance": round(float(dist[i, seg]), 3)}
        if detailed:
            proto = info["prototype"]
            row = df.iloc[i]
            res |= {
                "description": info.get("description"),
                "offres": info.get("offres", []),
                "proximite_segments": sorted(
                    [{"segment": s, "persona": SEGMENTS[str(s)]["persona"], "distance": round(float(dist[i, s]), 3)}
                     for s in range(len(SEGMENTS))], key=lambda r: r["distance"]),
                "comparaison_prototype": [
                    {"variable": c, "client": (row[c].item() if hasattr(row[c], "item") else row[c]),
                     "prototype": proto[c],
                     "identique": bool(row[c] == proto[c]) if c in SEG_CAT else None}
                    for c in SEG_NUM + SEG_CAT],
            }
        results.append(res)
    return results


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_version": meta["version"]}


@app.get("/segments")
def segments():
    keys = ["model_name", "version", "trained_at", "k", "gamma", "n_clients", "quality", "segments"]
    return {k: meta[k] for k in keys}


@app.get("/schema")
def schema():
    return {"categories": meta["categories"], "defaults": meta["defaults"]}


@app.post("/assign")
def assign(client: Client):
    return _assign(_validate([client]), detailed=True)[0]


@app.post("/assign-batch")
def assign_batch(clients: List[Client]):
    if not clients:
        raise HTTPException(status_code=422, detail="Liste de clients vide")
    if len(clients) > 50000:
        raise HTTPException(status_code=413, detail="50 000 clients maximum par appel")
    results = _assign(_validate(clients), detailed=False)
    counts = pd.Series([r["persona"] for r in results]).value_counts(normalize=True)
    return {"resultats": results,
            "synthese": {"nb_clients": len(results), "repartition": {k: round(float(v), 4) for k, v in counts.items()}}}
