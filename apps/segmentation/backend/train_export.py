"""
Entraîne la segmentation K-Prototypes retenue dans le Lab 2 et exporte les artefacts
utilisés par l'API et le dashboard.

Usage :  python train_export.py
"""

import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from features import (CAT_COLS, CAT_IDX, NUM_COLS, PERSONAS, SEG_CAT, SEG_NUM, add_features, assign_personas,
                      segment_profile, to_kproto_matrix)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "labs", "data", "bank_marketing.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

RANDOM_STATE = 42
K = 5

# Pipeline.inverse_transform passe un ndarray au PowerTransformer : avertissement sans conséquence
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    df = add_features(raw[NUM_COLS + CAT_COLS + ["y"]])

    # ── Mêmes étapes et mêmes échantillons que le Lab 2 ──────────────────────
    num_pipe = Pipeline([("yeojohnson", PowerTransformer(method="yeo-johnson")), ("scaler", StandardScaler())])
    encoder = ColumnTransformer([("num", num_pipe, SEG_NUM), ("cat", OneHotEncoder(sparse_output=False), SEG_CAT)])
    X_onehot = encoder.fit_transform(df)
    num_pipe.fit(df[SEG_NUM])

    rng = np.random.default_rng(RANDOM_STATE)
    idx_sil = rng.choice(len(df), 5000, replace=False)
    idx_kp = rng.choice(len(df), 10000, replace=False)

    kproto = KPrototypes(n_clusters=K, init="Cao", n_init=5, random_state=RANDOM_STATE, n_jobs=-1)
    kproto.fit(to_kproto_matrix(df.iloc[idx_kp], num_pipe), categorical=CAT_IDX)
    df["segment"] = kproto.predict(to_kproto_matrix(df, num_pipe), categorical=CAT_IDX)
    print("Tailles des segments :", np.bincount(df["segment"]))

    profile = segment_profile(df, "segment")
    persona_names = assign_personas(profile)
    df["persona"] = df["segment"].map(persona_names)

    # ── Qualité ──────────────────────────────────────────────────────────────
    quality = {
        "silhouette_onehot": float(silhouette_score(X_onehot[idx_sil], df["segment"].values[idx_sil])),
        "davies_bouldin_onehot": float(davies_bouldin_score(X_onehot, df["segment"])),
        "cost": float(kproto.cost_),
        "n_iter": int(kproto.n_iter_),
    }

    # ── Projection ACP 2D pour le dashboard ─────────────────────────────────
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_onehot)
    coords = pca.transform(X_onehot)
    df["pc1"], df["pc2"] = coords[:, 0], coords[:, 1]

    # ── Export ───────────────────────────────────────────────────────────────
    joblib.dump({"num_pipe": num_pipe, "kproto": kproto, "persona_names": persona_names},
                os.path.join(ARTIFACTS_DIR, "model.joblib"))

    centroids = kproto.cluster_centroids_
    proto_num = num_pipe.inverse_transform(pd.DataFrame(centroids[:, :len(SEG_NUM)].astype(float), columns=SEG_NUM))
    segments = {}
    for seg in range(K):
        p = profile.loc[seg]
        segments[str(seg)] = {
            "persona": persona_names[seg],
            **PERSONAS.get(persona_names[seg], {}),
            "profil": {k: float(v) for k, v in p.items()},
            "prototype": {**{c: float(round(proto_num[seg, j], 0)) for j, c in enumerate(SEG_NUM)},
                          **{c: str(centroids[seg, len(SEG_NUM) + j]) for j, c in enumerate(SEG_CAT)}},
        }
    meta = {
        "model_name": "K-Prototypes (données mixtes)",
        "version": datetime.now().strftime("%Y.%m.%d"),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "k": K,
        "gamma": float(kproto.gamma),
        "n_clients": int(len(df)),
        "quality": quality,
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "categories": {c: sorted(raw[c].unique().tolist()) for c in CAT_COLS},
        "defaults": {**{c: raw[c].mode()[0] for c in CAT_COLS}, **{c: int(raw[c].median()) for c in NUM_COLS}},
        "segments": segments,
    }
    with open(os.path.join(ARTIFACTS_DIR, "segments.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    df.to_csv(os.path.join(ARTIFACTS_DIR, "clients_segmented.csv"), index_label="client_id")
    raw[NUM_COLS + CAT_COLS].sample(100, random_state=RANDOM_STATE).to_csv(
        os.path.join(ARTIFACTS_DIR, "sample_clients.csv"), index=False)
    print(json.dumps({s: v["persona"] for s, v in segments.items()}, ensure_ascii=False))
    print(f"Qualité : {quality}")
    print(f"Artefacts exportés dans {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
