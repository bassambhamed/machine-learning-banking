"""
Feature engineering, personas et profilage de la segmentation client (identiques au Lab 2).
"""

import numpy as np
import pandas as pd

NUM_COLS = ["age", "balance"]
CAT_COLS = ["job", "marital", "education", "default", "housing", "loan"]
SEG_NUM = ["age", "balance"]
SEG_CAT = ["job_group", "marital", "education", "default", "housing", "loan"]
CAT_IDX = list(range(len(SEG_NUM), len(SEG_NUM) + len(SEG_CAT)))

JOB_GROUPS = {
    "management": "Cadres & indépendants", "entrepreneur": "Cadres & indépendants",
    "self-employed": "Cadres & indépendants",
    "admin.": "Employés & techniciens", "technician": "Employés & techniciens", "services": "Employés & techniciens",
    "blue-collar": "Ouvriers & services", "housemaid": "Ouvriers & services",
    "retired": "Retraités",
    "student": "Étudiants & sans emploi", "unemployed": "Étudiants & sans emploi", "unknown": "Étudiants & sans emploi",
}


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["job_group"] = d["job"].map(JOB_GROUPS).fillna("Étudiants & sans emploi")
    d["nb_credits"] = (d["housing"] == "yes").astype(int) + (d["loan"] == "yes").astype(int)
    d["negative_balance"] = (d["balance"] < 0).astype(int)
    d["age_group"] = pd.cut(d["age"], [0, 30, 40, 50, 60, 120],
                            labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str)
    return d


def to_kproto_matrix(data: pd.DataFrame, num_pipe) -> np.ndarray:
    # Numériques transformées + catégorielles brutes, dans une matrice d'objets.
    return np.column_stack([num_pipe.transform(data[SEG_NUM]), data[SEG_CAT].values]).astype(object)


PERSONAS = {
    "Patrimoniaux": {
        "description": "Soldes nettement plus élevés que la moyenne, très peu de crédit personnel, profil plus diplômé et plus souvent cadre.",
        "offres": ["Gestion de patrimoine / conseil", "Assurance-vie et placements", "Carte premium"],
        "icone": "💎"},
    "Clients fragiles": {
        "description": "Soldes négatifs (découvert), très forte part de crédits immobiliers et personnels : multi-endettement.",
        "offres": ["Accompagnement budgétaire", "Regroupement de crédits", "Suivi du risque renforcé"],
        "icone": "⚠️"},
    "Seniors": {
        "description": "Clients de plus de 50 ans, majoritairement mariés, peu de crédit immobilier, forte part de retraités.",
        "offres": ["Épargne retraite et dépôt à terme", "Prévoyance / dépendance", "Transmission de patrimoine"],
        "icone": "🧓"},
    "Jeunes actifs": {
        "description": "Autour de 30 ans, majoritairement célibataires, souvent primo-accédants (crédit immobilier), soldes modestes.",
        "offres": ["Livret et épargne programmée", "Assurance habitation / emprunteur", "Pack jeune actif / crédit auto"],
        "icone": "🚀"},
    "Familles propriétaires": {
        "description": "Quarantaine, mariés, crédit immobilier en cours pour 7 clients sur 10, soldes modestes ; appétence épargne faible.",
        "offres": ["Prévoyance famille", "Épargne enfants", "Assurance habitation / emprunteur"],
        "icone": "🏠"},
}

PERSONA_RULES = [("Patrimoniaux", "Solde médian (€)", "max"), ("Clients fragiles", "% solde négatif", "max"),
                 ("Seniors", "Âge moyen", "max"), ("Jeunes actifs", "Âge moyen", "min")]


def segment_profile(data: pd.DataFrame, seg_col: str) -> pd.DataFrame:
    g = data.groupby(seg_col)
    prof = pd.DataFrame({
        "Nb clients": g.size(),
        "% portefeuille": g.size() / len(data),
        "Âge moyen": g["age"].mean(),
        "Solde médian (€)": g["balance"].median(),
        "Solde moyen (€)": g["balance"].mean(),
        "% solde négatif": g["negative_balance"].mean(),
        "% crédit immo": g["housing"].apply(lambda s: (s == "yes").mean()),
        "% crédit perso": g["loan"].apply(lambda s: (s == "yes").mean()),
        "% défaut": g["default"].apply(lambda s: (s == "yes").mean()),
        "% diplômés sup.": g["education"].apply(lambda s: (s == "tertiary").mean()),
        "% célibataires": g["marital"].apply(lambda s: (s == "single").mean()),
        "% cadres & indép.": g["job_group"].apply(lambda s: (s == "Cadres & indépendants").mean()),
        "% retraités": g["job_group"].apply(lambda s: (s == "Retraités").mean()),
    })
    if "y" in data:
        prof["Taux souscription"] = g["y"].apply(lambda s: (s == "yes").mean())
        prof["Lift souscription"] = prof["Taux souscription"] / (data["y"] == "yes").mean()
    return prof


def assign_personas(prof: pd.DataFrame) -> dict:
    # Associe un nom de persona à chaque segment à partir de son profil.
    remaining = list(prof.index)
    names = {}
    for persona, col, how in PERSONA_RULES:
        if not remaining:
            break
        sub = prof.loc[remaining, col]
        seg = sub.idxmax() if how == "max" else sub.idxmin()
        names[seg] = persona
        remaining.remove(seg)
    for i, seg in enumerate(remaining):
        names[seg] = "Familles propriétaires" if i == 0 else f"Segment {seg}"
    return names
