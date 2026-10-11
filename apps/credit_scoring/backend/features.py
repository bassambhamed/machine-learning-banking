"""
Feature engineering du credit scoring (identique au Lab 1).
Module séparé pour que le pipeline sérialisé puisse être rechargé par l'API.
"""

import numpy as np
import pandas as pd

EMPLOYMENT_YEARS = {"unemployed": 0, "<1": 0.5, "1<=X<4": 2.5, "4<=X<7": 5.5, ">=7": 10}


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["monthly_payment"] = d["credit_amount"] / d["duration"]
    d["log_credit_amount"] = np.log1p(d["credit_amount"])
    d["employment_years"] = d["employment"].map(EMPLOYMENT_YEARS).fillna(0)
    d["amount_per_employment_year"] = d["credit_amount"] / (1 + d["employment_years"])
    d["age_group"] = pd.cut(d["age"], bins=[0, 25, 35, 50, 120],
                            labels=["18-25", "26-35", "36-50", "50+"]).astype(str)
    d["has_checking_account"] = (d["checking_status"] != "no checking").astype(int)
    return d


# Libellés métier (français) des variables brutes et dérivées
LABELS = {
    "checking_status": "Solde du compte courant",
    "duration": "Durée du crédit (mois)",
    "credit_history": "Historique de crédit",
    "purpose": "Objet du crédit",
    "credit_amount": "Montant demandé",
    "savings_status": "Épargne",
    "employment": "Ancienneté dans l'emploi",
    "installment_commitment": "Taux d'effort (1-4)",
    "personal_status": "Sexe & statut marital",
    "other_parties": "Co-emprunteur / garant",
    "residence_since": "Ancienneté de résidence",
    "property_magnitude": "Patrimoine",
    "age": "Âge",
    "other_payment_plans": "Autres crédits en cours",
    "housing": "Logement",
    "existing_credits": "Crédits existants dans la banque",
    "job": "Catégorie d'emploi",
    "num_dependents": "Personnes à charge",
    "own_telephone": "Téléphone",
    "foreign_worker": "Travailleur étranger",
    "monthly_payment": "Mensualité estimée",
    "log_credit_amount": "Montant (log)",
    "employment_years": "Années d'emploi",
    "amount_per_employment_year": "Montant / année d'emploi",
    "age_group": "Tranche d'âge",
    "has_checking_account": "Détient un compte courant",
}
