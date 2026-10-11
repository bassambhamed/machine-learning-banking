"""
Interface Streamlit de credit scoring.
Se connecte à l'API FastAPI (par défaut http://127.0.0.1:8000).

Pages : scoring d'une demande · scoring par lot · dashboard KPIs portefeuille · performance & monitoring

Lancement :  streamlit run app.py --server.port 8501
"""

import io
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

API_URL = os.getenv("CREDIT_API_URL", "http://127.0.0.1:8000")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "artifacts")

GREEN, ORANGE, RED, BLUE, GREY = "#2E8B57", "#E69F00", "#C44E52", "#4C72B0", "#9E9E9E"
DECISION_COLORS = {"Accepter": GREEN, "Revoir": ORANGE, "Refuser": RED}
RISK_COLORS = {"A": "#1A9850", "B": "#91CF60", "C": "#FEE08B", "D": "#FC8D59", "E": "#D73027"}

# Libellés français des modalités (les codes d'origine restent envoyés à l'API)
VALUE_LABELS = {
    "checking_status": {"<0": "Débiteur (< 0 DM)", "0<=X<200": "0 à 200 DM", ">=200": "≥ 200 DM",
                        "no checking": "Pas de compte courant"},
    "credit_history": {"no credits/all paid": "Aucun crédit / tout remboursé",
                       "all paid": "Crédits de la banque remboursés",
                       "existing paid": "Crédits en cours remboursés à temps",
                       "delayed previously": "Retards de paiement passés",
                       "critical/other existing credit": "Compte critique / crédits ailleurs"},
    "purpose": {"new car": "Voiture neuve", "used car": "Voiture d'occasion",
                "furniture/equipment": "Mobilier / équipement", "radio/tv": "Radio / TV",
                "domestic appliance": "Électroménager", "repairs": "Travaux / réparations",
                "education": "Études", "retraining": "Formation", "business": "Professionnel", "other": "Autre"},
    "savings_status": {"<100": "< 100 DM", "100<=X<500": "100 à 500 DM", "500<=X<1000": "500 à 1 000 DM",
                       ">=1000": "≥ 1 000 DM", "no known savings": "Pas d'épargne connue"},
    "employment": {"unemployed": "Sans emploi", "<1": "< 1 an", "1<=X<4": "1 à 4 ans", "4<=X<7": "4 à 7 ans",
                   ">=7": "≥ 7 ans"},
    "personal_status": {"male single": "Homme célibataire", "male mar/wid": "Homme marié / veuf",
                        "male div/sep": "Homme divorcé / séparé", "female div/dep/mar": "Femme div. / sép. / mariée"},
    "other_parties": {"none": "Aucun", "co applicant": "Co-emprunteur", "guarantor": "Garant"},
    "property_magnitude": {"real estate": "Bien immobilier", "life insurance": "Assurance-vie / épargne",
                           "car": "Voiture / autre", "no known property": "Aucun patrimoine connu"},
    "other_payment_plans": {"none": "Aucun", "bank": "Autre banque", "stores": "Magasins"},
    "housing": {"own": "Propriétaire", "rent": "Locataire", "for free": "Logé gratuitement"},
    "job": {"skilled": "Qualifié", "unskilled resident": "Non qualifié (résident)",
            "unemp/unskilled non res": "Sans emploi / non qualifié non résident",
            "high qualif/self emp/mgmt": "Très qualifié / indépendant / cadre"},
    "own_telephone": {"yes": "Oui", "none": "Non"},
    "foreign_worker": {"yes": "Oui", "no": "Non"},
}

st.set_page_config(page_title="Credit Scoring", page_icon="🏦", layout="wide")


# ── Données & helpers ────────────────────────────────────────────────────────
@st.cache_data
def load_config():
    with open(os.path.join(ARTIFACTS_DIR, "config.json")) as f:
        return json.load(f)


@st.cache_data
def load_portfolio():
    df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "portfolio_scored.csv"))
    df["duration_bucket"] = pd.cut(df["duration"], [0, 12, 24, 36, 72],
                                   labels=["≤ 12 mois", "13-24 mois", "25-36 mois", "> 36 mois"]).astype(str)
    df["age_group"] = pd.cut(df["age"], [0, 25, 35, 50, 120], labels=["18-25", "26-35", "36-50", "50+"]).astype(str)
    return df


def api_post(path, payload):
    try:
        resp = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        st.error(f"API injoignable sur {API_URL}. Lancez le backend : `uvicorn app:app --port 8000` "
                 "dans `apps/credit_scoring/backend`.")
        return None
    if resp.status_code != 200:
        st.error(f"Erreur API ({resp.status_code}) : {resp.json().get('detail', resp.text)}")
        return None
    return resp.json()


def label(col, value):
    return VALUE_LABELS.get(col, {}).get(value, value)


def decision_for(pd_score, threshold, review_factor):
    return np.where(pd_score >= threshold, "Refuser",
                    np.where(pd_score >= review_factor * threshold, "Revoir", "Accepter"))


def psi(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-4, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_categorical(reference, current):
    ref = reference.value_counts(normalize=True)
    cur = current.value_counts(normalize=True).reindex(ref.index).fillna(0)
    ref, cur = ref.clip(lower=1e-4), cur.clip(lower=1e-4)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def psi_flag(v):
    return "🟢 Stable" if v < 0.10 else ("🟠 Dérive modérée" if v < 0.25 else "🔴 Dérive significative")


config = load_config()
portfolio = load_portfolio()
THRESHOLD, REVIEW_FACTOR = config["threshold"], config["review_factor"]
CATS, DEFAULTS, RANGES = config["categories"], config["defaults"], config["numeric_ranges"]

# ── Navigation ───────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Credit Scoring")
page = st.sidebar.radio("Navigation", ["Scoring d'une demande", "Scoring par lot",
                                       "Dashboard portefeuille", "Performance & monitoring"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Modèle : {config['model_name']}")
st.sidebar.caption(f"Version {config['version']} · AUC test {config['metrics']['auc']:.3f} · "
                   f"Gini {config['metrics']['gini']:.3f}")
st.sidebar.caption(f"Seuil de refus : PD ≥ {THRESHOLD:.0%} · revue manuelle dès {REVIEW_FACTOR * THRESHOLD:.1%}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Scoring d'une demande
# ════════════════════════════════════════════════════════════════════════════
if page == "Scoring d'une demande":
    st.title("Scoring d'une demande de crédit")
    st.caption("Renseignez la demande : le modèle estime la probabilité de défaut (PD), la classe de risque "
               "et propose une décision, avec les facteurs qui l'expliquent.")

    def select(col, text, container):
        options = CATS[col]
        return container.selectbox(text, options, index=options.index(DEFAULTS[col]),
                                   format_func=lambda v, c=col: label(c, v))

    with st.form("application_form"):
        c1, c2, c3 = st.columns(3)
        c1.subheader("La demande")
        app_in = {
            "purpose": select("purpose", "Objet du crédit", c1),
            "credit_amount": c1.number_input("Montant demandé (DM)", 250, 20000, int(DEFAULTS["credit_amount"]), 250),
            "duration": c1.slider("Durée (mois)", 4, 72, int(DEFAULTS["duration"])),
            "installment_commitment": c1.slider("Taux d'effort (1 = faible, 4 = élevé)", 1, 4,
                                                int(DEFAULTS["installment_commitment"])),
            "other_parties": select("other_parties", "Co-emprunteur / garant", c1),
        }
        c2.subheader("Situation financière")
        app_in |= {
            "checking_status": select("checking_status", "Solde du compte courant", c2),
            "savings_status": select("savings_status", "Épargne", c2),
            "credit_history": select("credit_history", "Historique de crédit", c2),
            "other_payment_plans": select("other_payment_plans", "Autres crédits en cours", c2),
            "existing_credits": c2.slider("Crédits existants dans la banque", 1, 4, int(DEFAULTS["existing_credits"])),
            "property_magnitude": select("property_magnitude", "Patrimoine", c2),
        }
        c3.subheader("Profil")
        app_in |= {
            "age": c3.slider("Âge", 18, 80, int(DEFAULTS["age"])),
            "personal_status": select("personal_status", "Sexe & statut marital", c3),
            "job": select("job", "Catégorie d'emploi", c3),
            "employment": select("employment", "Ancienneté dans l'emploi", c3),
            "housing": select("housing", "Logement", c3),
            "residence_since": c3.slider("Ancienneté de résidence (1-4)", 1, 4, int(DEFAULTS["residence_since"])),
            "num_dependents": c3.slider("Personnes à charge", 0, 5, int(DEFAULTS["num_dependents"])),
            "own_telephone": select("own_telephone", "Téléphone à son nom", c3),
            "foreign_worker": select("foreign_worker", "Travailleur étranger", c3),
        }
        submitted = st.form_submit_button("Calculer le score", type="primary")

    if submitted:
        st.session_state["application"] = app_in
        st.session_state["result"] = api_post("/predict", app_in)

    result, application = st.session_state.get("result"), st.session_state.get("application")
    if result and application:
        st.markdown("---")
        col_g, col_d = st.columns([1, 1.3])
        with col_g:
            pd_pct = result["probabilite_defaut"] * 100
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=pd_pct, number={"suffix": " %", "valueformat": ".1f"},
                title={"text": "Probabilité de défaut"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333333", "thickness": 0.25},
                       "steps": [{"range": [0, REVIEW_FACTOR * THRESHOLD * 100], "color": "#C7E9C0"},
                                 {"range": [REVIEW_FACTOR * THRESHOLD * 100, THRESHOLD * 100], "color": "#FDD49E"},
                                 {"range": [THRESHOLD * 100, 100], "color": "#FCBBA1"}],
                       "threshold": {"line": {"color": RED, "width": 3}, "value": THRESHOLD * 100}}))
            gauge.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
            st.plotly_chart(gauge)
            m1, m2, m3 = st.columns(3)
            m1.metric("Probabilité de défaut", f"{result['probabilite_defaut']:.1%}", border=True)
            m2.metric("Classe de risque", result["classe_risque"], border=True)
            m3.metric("Score (0-1000)", result["score"], border=True)
            color = DECISION_COLORS[result["decision"]]
            st.markdown(f"<div style='padding:10px;border-radius:8px;background:{color};color:white;"
                        f"text-align:center;font-size:1.2em'><b>{result['decision'].upper()}</b></div>",
                        unsafe_allow_html=True)
        with col_d:
            factors = pd.DataFrame(result["facteurs"]).iloc[::-1]
            factors["texte"] = [f"{r.libelle} = {label(r.variable, r.valeur)}" for r in factors.itertuples()]
            fig = go.Figure(go.Bar(x=factors["contribution"], y=factors["texte"], orientation="h",
                                   marker_color=[RED if c > 0 else GREEN for c in factors["contribution"]]))
            fig.update_layout(title="Principaux facteurs (vs demandeur moyen) — rouge : augmente le risque",
                              xaxis_title="Contribution au log-odds du défaut", height=380,
                              margin=dict(t=50, b=40, l=10, r=10))
            st.plotly_chart(fig)

        st.subheader("Simulation « what-if »")
        st.caption("Comment la PD évolue-t-elle si l'on modifie la durée ou le montant, toutes choses égales par ailleurs ?")
        durations = list(range(6, 73, 6))
        amounts = list(range(500, 20001, 1000))
        res_d = api_post("/predict-batch", [application | {"duration": d} for d in durations])
        res_a = api_post("/predict-batch", [application | {"credit_amount": a} for a in amounts])
        if res_d and res_a:
            w1, w2 = st.columns(2)
            for container, xs, res, xlabel, current in [
                (w1, durations, res_d, "Durée (mois)", application["duration"]),
                (w2, amounts, res_a, "Montant (DM)", application["credit_amount"]),
            ]:
                pds = [r["probabilite_defaut"] for r in res["resultats"]]
                fig = go.Figure(go.Scatter(x=xs, y=pds, mode="lines+markers", line=dict(color=BLUE)))
                fig.add_hline(y=THRESHOLD, line_dash="dash", line_color=RED, annotation_text="seuil de refus")
                fig.add_hline(y=REVIEW_FACTOR * THRESHOLD, line_dash="dot", line_color=ORANGE,
                              annotation_text="revue manuelle")
                fig.add_vline(x=current, line_color=GREY, annotation_text="demande actuelle")
                fig.update_layout(xaxis_title=xlabel, yaxis_title="PD", yaxis_tickformat=".0%", height=320,
                                  margin=dict(t=30, b=40))
                container.plotly_chart(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Scoring par lot
# ════════════════════════════════════════════════════════════════════════════
elif page == "Scoring par lot":
    st.title("Scoring par lot")
    st.caption("Chargez un fichier CSV de demandes (mêmes colonnes que le jeu German Credit, sans la cible).")

    with open(os.path.join(ARTIFACTS_DIR, "sample_applications.csv"), "rb") as f:
        st.download_button("📥 Télécharger un fichier exemple (50 demandes)", f, "sample_applications.csv", "text/csv")

    uploaded = st.file_uploader("Fichier CSV", type="csv")
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [c for c in config["raw_columns"] if c not in batch.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
            st.stop()
        st.write(f"**{len(batch)} demandes** chargées.")
        st.dataframe(batch.head(), hide_index=True)

        if st.button("Lancer le scoring", type="primary"):
            res = api_post("/predict-batch", batch[config["raw_columns"]].to_dict(orient="records"))
            if res:
                scored = pd.concat([batch.reset_index(drop=True), pd.DataFrame(res["resultats"])], axis=1)
                s = res["synthese"]
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Demandes", s["nb_demandes"], border=True)
                k2.metric("PD moyenne", f"{s['pd_moyenne']:.1%}", border=True)
                k3.metric("Acceptées", f"{s['taux_acceptation']:.0%}", border=True)
                k4.metric("En revue", f"{s['taux_revue']:.0%}", border=True)
                k5.metric("Refusées", f"{s['taux_refus']:.0%}", border=True)

                g1, g2 = st.columns(2)
                dec = scored["decision"].value_counts().reindex(["Accepter", "Revoir", "Refuser"]).fillna(0)
                g1.plotly_chart(go.Figure(go.Pie(labels=dec.index, values=dec.values, hole=0.5,
                                                 marker_colors=[DECISION_COLORS[d] for d in dec.index]))
                                .update_layout(title="Décisions", height=330))
                fig = px.histogram(scored, x="probabilite_defaut", color="classe_risque", nbins=25,
                                   color_discrete_map=RISK_COLORS, category_orders={"classe_risque": list("ABCDE")})
                fig.add_vline(x=THRESHOLD, line_dash="dash", line_color=RED, annotation_text="seuil")
                fig.update_layout(title="Distribution des PD", xaxis_tickformat=".0%", height=330)
                g2.plotly_chart(fig)

                st.dataframe(scored.style.map(lambda d: f"color:{DECISION_COLORS.get(d, 'black')};font-weight:bold",
                                              subset=["decision"]), hide_index=True)
                buf = io.StringIO()
                scored.to_csv(buf, index=False)
                st.download_button("📥 Télécharger les résultats", buf.getvalue(), "demandes_scorees.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Dashboard portefeuille
# ════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard portefeuille":
    st.title("Dashboard KPIs — portefeuille de demandes")
    st.caption("Historique des 1 000 demandes scorées (PD hors-pli pour le train, PD du modèle pour le test) "
               "et défauts réellement observés.")

    st.sidebar.markdown("### Filtres")
    purposes = st.sidebar.multiselect("Objet du crédit (vide = tous)", CATS["purpose"],
                                      format_func=lambda v: label("purpose", v)) or CATS["purpose"]
    age_min, age_max = st.sidebar.slider("Âge", 18, 80, (18, 80))
    threshold = st.sidebar.slider("Seuil de refus (PD)", 0.05, 0.60, float(THRESHOLD), 0.01)
    lgd = st.sidebar.slider("LGD (perte en cas de défaut)", 0.10, 1.0, 0.45, 0.05,
                            help="Hypothèse de taux de perte en cas de défaut ; 45 % = valeur réglementaire "
                                 "de référence (approche IRB fondation) pour une créance senior non garantie.")

    df = portfolio[portfolio["purpose"].isin(purposes) & portfolio["age"].between(age_min, age_max)].copy()
    if df.empty:
        st.warning("Aucune demande pour ces filtres.")
        st.stop()
    df["decision"] = decision_for(df["pd_score"], threshold, REVIEW_FACTOR)
    accepted = df[df["decision"] != "Refuser"]
    df["expected_loss"] = df["pd_score"] * lgd * df["credit_amount"]

    st.subheader("Portefeuille")
    k = st.columns(4)
    k[0].metric("Demandes", f"{len(df):,}".replace(",", " "), border=True)
    k[1].metric("Montant total demandé", f"{df['credit_amount'].sum() / 1e6:.2f} M DM", border=True)
    k[2].metric("Taux de défaut observé", f"{df['default'].mean():.1%}", border=True)
    k[3].metric("PD moyenne prédite", f"{df['pd_score'].mean():.1%}", border=True)

    st.subheader(f"Politique d'octroi au seuil {threshold:.0%}")
    k = st.columns(4)
    k[0].metric("Taux d'acceptation (hors refus)", f"{len(accepted) / len(df):.0%}", border=True)
    k[1].metric("Montant accordé", f"{accepted['credit_amount'].sum() / 1e6:.2f} M DM", border=True)
    k[2].metric("Taux de défaut des acceptés", f"{accepted['default'].mean():.1%}" if len(accepted) else "—",
                delta=f"{accepted['default'].mean() - df['default'].mean():+.1%} vs portefeuille" if len(accepted) else None,
                delta_color="inverse", border=True)
    el = (accepted["pd_score"] * lgd * accepted["credit_amount"]).sum()
    k[3].metric("Perte attendue (PD × LGD × EAD)", f"{el / 1e3:,.0f} k DM".replace(",", " "),
                help="Somme sur les dossiers acceptés de PD × LGD × montant", border=True)

    c1, c2 = st.columns(2)
    risk = df.groupby("risk_class").agg(n=("default", "size"), taux=("default", "mean"), pd=("pd_score", "mean"))
    risk = risk.reindex(list("ABCDE"))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=risk.index, y=risk["n"], name="Nb demandes", marker_color=[RISK_COLORS[c] for c in risk.index])
    fig.add_scatter(x=risk.index, y=risk["taux"], name="Défaut observé", mode="lines+markers",
                    line=dict(color="black"), secondary_y=True)
    fig.add_scatter(x=risk.index, y=risk["pd"], name="PD moyenne", mode="lines+markers",
                    line=dict(color=BLUE, dash="dash"), secondary_y=True)
    fig.update_yaxes(title_text="Taux", tickformat=".0%", secondary_y=True)
    fig.update_layout(title="Grille de risque : volumes et taux de défaut par classe", height=380)
    c1.plotly_chart(fig)

    fig = px.histogram(df, x="pd_score", color=df["default"].map({0: "Bon payeur", 1: "Défaut"}), nbins=30,
                       barmode="overlay", opacity=0.65, color_discrete_sequence=[BLUE, RED])
    fig.add_vline(x=threshold, line_dash="dash", line_color=RED, annotation_text="seuil")
    fig.update_layout(title="Distribution des PD selon le statut réel", xaxis_tickformat=".0%",
                      legend_title="", height=380)
    c2.plotly_chart(fig)

    def rate_chart(col, title, order=None):
        g = df.groupby(col).agg(observe=("default", "mean"), predit=("pd_score", "mean"), n=("default", "size"))
        g = g.reindex(order) if order else g.sort_values("observe")
        g.index = [label(col, v) for v in g.index]
        fig = go.Figure([go.Bar(x=g.index, y=g["observe"], name="Défaut observé", marker_color=RED,
                                text=[f"n={n}" for n in g["n"]], textposition="outside"),
                         go.Bar(x=g.index, y=g["predit"], name="PD moyenne prédite", marker_color=BLUE)])
        fig.update_layout(title=title, yaxis_tickformat=".0%", barmode="group", height=380,
                          legend=dict(orientation="h", y=1.12))
        return fig

    c1, c2 = st.columns(2)
    c1.plotly_chart(rate_chart("purpose", "Taux de défaut par objet du crédit"))
    c2.plotly_chart(rate_chart("checking_status", "Taux de défaut par solde du compte courant"))
    c1, c2 = st.columns(2)
    c1.plotly_chart(rate_chart("duration_bucket", "Taux de défaut par durée",
                               ["≤ 12 mois", "13-24 mois", "25-36 mois", "> 36 mois"]))
    c2.plotly_chart(rate_chart("age_group", "Taux de défaut par tranche d'âge", ["18-25", "26-35", "36-50", "50+"]))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Performance & monitoring
# ════════════════════════════════════════════════════════════════════════════
else:
    st.title("Performance du modèle & monitoring")
    test = portfolio[portfolio["split"] == "test"]
    train = portfolio[portfolio["split"] == "train"]
    m = config["metrics"]

    st.subheader("Performance sur le jeu de test (200 demandes jamais vues)")
    k = st.columns(5)
    k[0].metric("AUC", f"{m['auc']:.3f}", help="Probabilité qu'un défaut ait une PD plus élevée qu'un bon payeur",
                border=True)
    k[1].metric("Gini", f"{m['gini']:.3f}", help="Gini = 2 × AUC − 1", border=True)
    k[2].metric("KS", f"{m['ks']:.3f}", help="Écart max entre les distributions cumulées des bons et des mauvais",
                border=True)
    k[3].metric("PR-AUC", f"{m['pr_auc']:.3f}", border=True)
    k[4].metric("AUC validation croisée", f"{m['auc_cv']:.3f}", border=True)

    threshold = st.slider("Seuil de refus à analyser", 0.05, 0.80, float(THRESHOLD), 0.01)
    y, p = test["default"].values, test["pd_score"].values
    pred = (p >= threshold).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    fp, tn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum())
    cost = config["cost_fn"] * fn + config["cost_fp"] * fp

    k = st.columns(4)
    k[0].metric("Taux d'acceptation", f"{(pred == 0).mean():.0%}", border=True)
    k[1].metric("Rappel (défauts détectés)", f"{tp / max(tp + fn, 1):.0%}", border=True)
    k[2].metric("Précision des refus", f"{tp / max(tp + fp, 1):.0%}", border=True)
    k[3].metric(f"Coût métier ({config['cost_fn']}×FN + {config['cost_fp']}×FP)", cost, border=True)

    c1, c2, c3 = st.columns([1, 1, 1.2])
    fpr_list, tpr_list = [], []
    for t in np.linspace(1, 0, 201):
        pr = p >= t
        tpr_list.append((pr & (y == 1)).sum() / (y == 1).sum())
        fpr_list.append((pr & (y == 0)).sum() / (y == 0).sum())
    fig = go.Figure(go.Scatter(x=fpr_list, y=tpr_list, mode="lines", name=f"AUC = {m['auc']:.3f}",
                               line=dict(color=BLUE, width=3)))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="Hasard")
    fig.add_scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode="markers", marker=dict(size=12, color=RED),
                    name="Seuil choisi")
    fig.update_layout(title="Courbe ROC", xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs",
                      height=360, legend=dict(x=0.45, y=0.05))
    c1.plotly_chart(fig)

    cm = np.array([[tn, fp], [fn, tp]])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Prédit bon", "Prédit défaut"], y=["Réel bon", "Réel défaut"])
    fig.update_layout(title=f"Matrice de confusion (seuil {threshold:.2f})", height=360, coloraxis_showscale=False)
    c2.plotly_chart(fig)

    ts = np.linspace(0.05, 0.80, 76)
    costs = [config["cost_fn"] * int(((p < t) & (y == 1)).sum()) + config["cost_fp"] * int(((p >= t) & (y == 0)).sum())
             for t in ts]
    acc = [(p < t).mean() for t in ts]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=ts, y=costs, name="Coût métier", line=dict(color=RED))
    fig.add_scatter(x=ts, y=acc, name="Taux d'acceptation", line=dict(color=GREEN, dash="dash"), secondary_y=True)
    fig.add_vline(x=threshold, line_color="black", annotation_text="choisi")
    fig.add_vline(x=THRESHOLD, line_color=GREY, line_dash="dot", annotation_text="optimal (train)",
                  annotation_position="bottom right")
    fig.update_yaxes(title_text="Taux d'acceptation", tickformat=".0%", secondary_y=True)
    fig.update_layout(title="Arbitrage coût / acceptation", xaxis_title="Seuil", height=360,
                      legend=dict(orientation="h", y=1.15))
    c3.plotly_chart(fig)
    st.info("💡 Le seuil optimal en coût pur est bas (matrice de coût 5:1) et réduit fortement le taux d'acceptation. "
            "En pratique, la politique d'octroi fixe un compromis sous contrainte commerciale : c'est une décision métier.")

    st.markdown("---")
    st.subheader("Monitoring : stabilité de la population (PSI)")
    st.caption("Simulez un lot de production dont la population a dérivé, puis comparez-le à la population "
               "de référence (train) avec le Population Stability Index.")
    s1, s2, s3, s4 = st.columns(4)
    amount_mult = s1.slider("Montants × ", 0.5, 2.0, 1.0, 0.1)
    duration_add = s2.slider("Durées + (mois)", 0, 24, 0, 3)
    age_shift = s3.slider("Âge − (années)", 0, 15, 0, 1)
    share_debtor = s4.slider("Part de comptes débiteurs", 0.0, 0.8, float((train["checking_status"] == "<0").mean()), 0.05)

    rng = np.random.default_rng(0)
    prod = test[config["raw_columns"]].sample(400, replace=True, random_state=0).reset_index(drop=True)
    prod["credit_amount"] = (prod["credit_amount"] * amount_mult).round()
    prod["duration"] = (prod["duration"] + duration_add).clip(upper=72)
    prod["age"] = (prod["age"] - age_shift).clip(lower=18)
    # Ajuste la part de comptes débiteurs en modifiant le minimum de dossiers
    debtors = prod.index[prod["checking_status"] == "<0"]
    n_target = int(round(share_debtor * len(prod)))
    if n_target > len(debtors):
        others = prod.index[prod["checking_status"] != "<0"]
        prod.loc[rng.choice(others, n_target - len(debtors), replace=False), "checking_status"] = "<0"
    elif n_target < len(debtors):
        prod.loc[rng.choice(debtors, len(debtors) - n_target, replace=False), "checking_status"] = "0<=X<200"

    res = api_post("/predict-batch", prod.to_dict(orient="records"))
    if res:
        prod_scores = np.array([r["probabilite_defaut"] for r in res["resultats"]])
        rows = [{"Variable": "Score (PD)", "PSI": psi(train["pd_score"], prod_scores)}]
        rows += [{"Variable": c, "PSI": psi(train[c], prod[c])} for c in ["credit_amount", "duration", "age"]]
        rows += [{"Variable": c, "PSI": psi_categorical(train[c], prod[c])}
                 for c in ["checking_status", "savings_status", "purpose"]]
        psi_df = pd.DataFrame(rows)
        psi_df["Statut"] = psi_df["PSI"].apply(psi_flag)

        c1, c2 = st.columns([1, 1.3])
        c1.dataframe(psi_df.style.format({"PSI": "{:.3f}"}), hide_index=True, height=300)
        global_psi = psi_df.loc[0, "PSI"]
        c1.metric("PSI du score", f"{global_psi:.3f}", psi_flag(global_psi), border=True,
                  delta_color="off")
        fig = go.Figure()
        fig.add_histogram(x=train["pd_score"], name="Référence (train)", histnorm="probability", opacity=0.6,
                          marker_color=BLUE, xbins=dict(size=0.05))
        fig.add_histogram(x=prod_scores, name="Lot de production", histnorm="probability", opacity=0.6,
                          marker_color=ORANGE, xbins=dict(size=0.05))
        fig.update_layout(barmode="overlay", title="Distribution des scores : référence vs production",
                          xaxis_tickformat=".0%", height=360)
        c2.plotly_chart(fig)
        st.caption("Règle usuelle : PSI < 0,10 stable · 0,10–0,25 dérive modérée (analyser) · "
                   "> 0,25 dérive significative (revoir / réentraîner le modèle).")
