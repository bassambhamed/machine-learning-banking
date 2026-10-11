"""
Télécharge (une seule fois) les deux jeux de données de la formation :

- german_credit.csv  : German Credit (UCI / OpenML "credit-g") — Lab 1 credit scoring
- bank_marketing.csv : Bank Marketing (UCI, bank-full.csv)     — Lab 2 segmentation

Usage :  python download_data.py
"""

import io
import os
import urllib.request
import zipfile

import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def download_german_credit():
    path = os.path.join(DATA_DIR, "german_credit.csv")
    if os.path.exists(path):
        print(f"[ok] {path} existe déjà")
        return
    from sklearn.datasets import fetch_openml

    ds = fetch_openml("credit-g", version=1, as_frame=True)
    df = ds.frame.copy()
    # cible : 1 = mauvais payeur (défaut), 0 = bon payeur
    df["default"] = (df["class"] == "bad").astype(int)
    df = df.drop(columns="class")
    df.to_csv(path, index=False)
    print(f"[ok] German Credit : {df.shape} -> {path}")


def download_bank_marketing():
    path = os.path.join(DATA_DIR, "bank_marketing.csv")
    if os.path.exists(path):
        print(f"[ok] {path} existe déjà")
        return
    with urllib.request.urlopen(BANK_URL, timeout=60) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    # l'archive UCI contient un zip imbriqué bank.zip -> bank-full.csv
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
    df = pd.read_csv(inner.open("bank-full.csv"), sep=";")
    df.to_csv(path, index=False)
    print(f"[ok] Bank Marketing : {df.shape} -> {path}")


if __name__ == "__main__":
    download_german_credit()
    download_bank_marketing()
