"""
Interface Streamlit de segmentation client.
Se connecte à l'API FastAPI (par défaut http://127.0.0.1:8001).

Pages : affecter un client · affectation par lot · dashboard segments · qualité & dérive

Lancement :  streamlit run app.py --server.port 8502
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

API_URL = os.getenv("SEGMENTATION_API_URL", "http://127.0.0.1:8001")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "artifacts")

PERSONA_COLORS = {"Patrimoniaux": "#4C72B0", "Seniors": "#8172B3", "Jeunes actifs": "#55A868",
                  "Familles propriétaires": "#DD8452", "Clients fragiles": "#C44E52"}

VALUE_LABELS = {
    "job": {"admin.": "Employé administratif", "blue-collar": "Ouvrier", "entrepreneur": "Entrepreneur",
            "housemaid": "Employé de maison", "management": "Cadre", "retired": "Retraité",
            "self-employed": "Indépendant", "services": "Services", "student": "Étudiant",
            "technician": "Technicien", "unemployed": "Sans emploi", "unknown": "Inconnu"},
    "marital": {"married": "Marié(e)", "single": "Célibataire", "divorced": "Divorcé(e) / veuf(ve)"},
    "education": {"primary": "Primaire", "secondary": "Secondaire", "tertiary": "Supérieur", "unknown": "Inconnu"},
    "default": {"no": "Non", "yes": "Oui"},
    "housing": {"no": "Non", "yes": "Oui"},
    "loan": {"no": "Non", "yes": "Oui"},
}
VAR_LABELS = {"age": "Âge", "balance": "Solde (€)", "job_group": "Groupe socio-professionnel",
              "marital": "Statut marital", "education": "Études", "default": "Incident de crédit",
              "housing": "Crédit immobilier", "loan": "Crédit personnel"}

st.set_page_config(page_title="Segmentation client", page_icon="🧩", layout="wide")


# ── Données & helpers ────────────────────────────────────────────────────────
@st.cache_data
def load_meta():
    with open(os.path.join(ARTIFACTS_DIR, "segments.json")) as f:
        return json.load(f)


@st.cache_data
def load_clients():
    return pd.read_csv(os.path.join(ARTIFACTS_DIR, "clients_segmented.csv"))


def api_post(path, payload):
    try:
        resp = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
    except requests.exceptions.ConnectionError:
        st.error(f"API injoignable sur {API_URL}. Lancez le backend : `uvicorn app:app --port 8001` "
                 "dans `apps/segmentation/backend`.")
        return None
    if resp.status_code != 200:
        st.error(f"Erreur API ({resp.status_code}) : {resp.json().get('detail', resp.text)}")
        return None
    return resp.json()


def label(col, value):
    return VALUE_LABELS.get(col, {}).get(value, value)


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


def fmt_int(v):
    return f"{v:,.0f}".replace(",", " ")


def psi_flag(v):
    return "🟢 Stable" if v < 0.10 else ("🟠 Dérive modérée" if v < 0.25 else "🔴 Dérive significative")


meta = load_meta()
clients = load_clients()
SEGMENTS = meta["segments"]
PERSONA_ORDER = [SEGMENTS[s]["persona"] for s in sorted(SEGMENTS, key=lambda s: -SEGMENTS[s]["profil"]["Nb clients"])]
PERSONA_INFO = {v["persona"]: v for v in SEGMENTS.values()}
GLOBAL_SUB = (clients["y"] == "yes").mean()

# ── Navigation ───────────────────────────────────────────────────────────────
st.sidebar.title("🧩 Segmentation client")
page = st.sidebar.radio("Navigation", ["Affecter un client", "Affectation par lot", "Dashboard segments",
                                       "Qualité & dérive"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Modèle : {meta['model_name']} · k = {meta['k']} · γ = {meta['gamma']:.2f}")
st.sidebar.caption(f"Version {meta['version']} · {meta['n_clients']:,} clients segmentés".replace(",", " "))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Affecter un client
# ════════════════════════════════════════════════════════════════════════════
if page == "Affecter un client":
    st.title("Affecter un client à un segment")
    st.caption("Renseignez le profil du client : le modèle K-Prototypes l'affecte au segment le plus proche "
               "et propose les offres associées à sa persona.")

    cats, defaults = meta["categories"], meta["defaults"]

    def select(col, text, container):
        return container.selectbox(text, cats[col], index=cats[col].index(defaults[col]),
                                   format_func=lambda v, c=col: label(c, v))

    with st.form("client_form"):
        c1, c2, c3 = st.columns(3)
        client = {
            "age": c1.slider("Âge", 18, 95, int(defaults["age"])),
            "balance": c1.number_input("Solde moyen annuel (€)", -10000, 150000, int(defaults["balance"]), 100),
            "job": select("job", "Profession", c2),
            "marital": select("marital", "Statut marital", c2),
            "education": select("education", "Niveau d'études", c2),
            "housing": select("housing", "Crédit immobilier en cours", c3),
            "loan": select("loan", "Crédit personnel en cours", c3),
            "default": select("default", "Incident de crédit", c3),
        }
        submitted = st.form_submit_button("Affecter", type="primary")

    if submitted:
        st.session_state["client"] = client
        st.session_state["assignment"] = api_post("/assign", client)

    res, client = st.session_state.get("assignment"), st.session_state.get("client")
    if res and client:
        persona = res["persona"]
        info = PERSONA_INFO[persona]
        color = PERSONA_COLORS.get(persona, "#555")
        st.markdown("---")
        left, right = st.columns([1.1, 1])
        with left:
            offers = "".join(f"<li>{o}</li>" for o in res["offres"])
            st.markdown(
                f"<div style='border-left:8px solid {color};padding:16px 20px;border-radius:8px;"
                f"background:rgba(128,128,128,0.08)'>"
                f"<div style='font-size:2em'>{info.get('icone', '')} <b>{persona}</b></div>"
                f"<p style='margin-top:6px'>{res['description']}</p>"
                f"<b>Offres recommandées</b><ul style='margin-bottom:0'>{offers}</ul></div>",
                unsafe_allow_html=True)
            p = info["profil"]
            m = st.columns(3)
            m[0].metric("Taille du segment", f"{p['% portefeuille']:.0%}", border=True)
            m[1].metric("Appétence épargne", f"{p['Taux souscription']:.1%}",
                        delta=f"lift {p['Lift souscription']:.2f}", delta_color="off", border=True)
            m[2].metric("Distance au prototype", f"{res['distance']:.2f}", border=True,
                        help="Distance K-Prototypes : plus elle est faible, plus le client est typique du segment")

            comp = pd.DataFrame(res["comparaison_prototype"])
            comp["Variable"] = comp["variable"].map(VAR_LABELS)
            comp["Client"] = [label(v, c) if isinstance(c, str) else f"{c:,.0f}".replace(",", " ")
                              for v, c in zip(comp["variable"], comp["client"])]
            comp["Profil type du segment"] = [label(v, c) if isinstance(c, str) else f"{c:,.0f}".replace(",", " ")
                                              for v, c in zip(comp["variable"], comp["prototype"])]
            comp["Écart"] = comp["identique"].map({True: "✅", False: "≠"}).fillna("")
            st.dataframe(comp[["Variable", "Client", "Profil type du segment", "Écart"]], hide_index=True)

        with right:
            prox = pd.DataFrame(res["proximite_segments"]).sort_values("distance", ascending=False)
            fig = go.Figure(go.Bar(x=prox["distance"], y=prox["persona"], orientation="h",
                                   marker_color=[PERSONA_COLORS.get(p, "#999") for p in prox["persona"]]))
            fig.update_layout(title="Distance du client à chaque segment (plus court = plus proche)",
                              height=300, margin=dict(t=50, b=30))
            st.plotly_chart(fig)

            sample = clients.sample(4000, random_state=0)
            fig = px.scatter(sample, x="age", y="balance", color="persona", opacity=0.35,
                             color_discrete_map=PERSONA_COLORS, category_orders={"persona": PERSONA_ORDER})
            fig.add_scatter(x=[client["age"]], y=[client["balance"]], mode="markers", name="Ce client",
                            marker=dict(symbol="star", size=22, color="black", line=dict(color="white", width=1)))
            fig.update_layout(title="Position du client (âge × solde, zoom sur −3 000 € à 15 000 €)", height=400,
                              yaxis_range=[-3000, 15000],
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Affectation par lot
# ════════════════════════════════════════════════════════════════════════════
elif page == "Affectation par lot":
    st.title("Affectation par lot")
    st.caption("Chargez un fichier CSV de clients (colonnes : age, balance, job, marital, education, default, housing, loan).")
    with open(os.path.join(ARTIFACTS_DIR, "sample_clients.csv"), "rb") as f:
        st.download_button("📥 Télécharger un fichier exemple (100 clients)", f, "sample_clients.csv", "text/csv")

    uploaded = st.file_uploader("Fichier CSV", type="csv")
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        required = ["age", "balance", "job", "marital", "education", "default", "housing", "loan"]
        missing = [c for c in required if c not in batch.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
            st.stop()
        st.write(f"**{len(batch)} clients** chargés.")
        st.dataframe(batch.head(), hide_index=True)
        if st.button("Lancer l'affectation", type="primary"):
            res = api_post("/assign-batch", batch[required].to_dict(orient="records"))
            if res:
                out = pd.concat([batch.reset_index(drop=True), pd.DataFrame(res["resultats"])], axis=1)
                out["offres"] = out["persona"].map(lambda p: ", ".join(PERSONA_INFO.get(p, {}).get("offres", [])))
                c1, c2 = st.columns([1, 1.4])
                rep = out["persona"].value_counts()
                c1.plotly_chart(go.Figure(go.Pie(labels=rep.index, values=rep.values, hole=0.5,
                                                 marker_colors=[PERSONA_COLORS.get(p) for p in rep.index]))
                                .update_layout(title="Répartition du lot", height=360))
                comp = pd.DataFrame({"Lot chargé": out["persona"].value_counts(normalize=True),
                                     "Portefeuille de référence": clients["persona"].value_counts(normalize=True)})
                fig = px.bar(comp.reset_index(names="persona"), x="persona",
                             y=["Lot chargé", "Portefeuille de référence"], barmode="group",
                             color_discrete_sequence=["#DD8452", "#4C72B0"])
                fig.update_layout(title="Lot vs portefeuille", yaxis_tickformat=".0%", height=360, legend_title="",
                                  xaxis_title="", yaxis_title="Part des clients")
                c2.plotly_chart(fig)
                st.dataframe(out, hide_index=True)
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button("📥 Télécharger les résultats", buf.getvalue(), "clients_segmentes.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Dashboard segments
# ════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard segments":
    st.title("Dashboard des segments clients")
    selected = st.sidebar.multiselect("Segments affichés (vide = tous)", PERSONA_ORDER) or PERSONA_ORDER
    df = clients[clients["persona"].isin(selected)]

    k = st.columns(5)
    k[0].metric("Clients", f"{len(df):,}".replace(",", " "), border=True)
    k[1].metric("Âge moyen", f"{df['age'].mean():.1f} ans", border=True)
    k[2].metric("Solde médian", f"{df['balance'].median():,.0f} €".replace(",", " "), border=True)
    k[3].metric("% crédit immobilier", f"{(df['housing'] == 'yes').mean():.0%}", border=True)
    k[4].metric("Taux de souscription", f"{(df['y'] == 'yes').mean():.1%}",
                delta=(f"{(df['y'] == 'yes').mean() - GLOBAL_SUB:+.1%} vs portefeuille"
                       if len(selected) < len(PERSONA_ORDER) else None), border=True)

    st.subheader("Les personas")
    cols = st.columns(len(selected))
    for col, persona in zip(cols, selected):
        info = PERSONA_INFO[persona]
        p = info["profil"]
        col.markdown(
            f"<div style='border-top:6px solid {PERSONA_COLORS.get(persona)};padding:10px;border-radius:6px;"
            f"background:rgba(128,128,128,0.08);min-height:210px'>"
            f"<div style='font-size:1.15em'>{info.get('icone', '')} <b>{persona}</b></div>"
            f"<div style='font-size:1.6em;font-weight:600'>{p['% portefeuille']:.0%}</div>"
            f"<div style='font-size:0.85em'>{fmt_int(p['Nb clients'])} clients · {p['Âge moyen']:.0f} ans · "
            f"solde méd. {fmt_int(p['Solde médian (€)'])} €<br>Souscription {p['Taux souscription']:.1%} "
            f"(lift {p['Lift souscription']:.2f})</div>"
            f"<div style='font-size:0.8em;margin-top:6px;opacity:0.8'>{info.get('description', '')}</div></div>",
            unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.3])
    sizes = df["persona"].value_counts().reset_index()
    sizes.columns = ["persona", "clients"]
    fig = px.treemap(sizes, path=["persona"], values="clients", color="persona", color_discrete_map=PERSONA_COLORS)
    fig.update_traces(textinfo="label+value+percent root")
    fig.update_layout(title="Poids des segments dans le portefeuille", height=400, margin=dict(t=50, l=5, r=5, b=5))
    c1.plotly_chart(fig)

    prof = pd.DataFrame({p: PERSONA_INFO[p]["profil"] for p in selected}).T
    heat_cols = ["Âge moyen", "Solde médian (€)", "% crédit immo", "% crédit perso", "% solde négatif",
                 "% diplômés sup.", "% célibataires", "% retraités", "Taux souscription"]
    norm = (prof[heat_cols] - prof[heat_cols].min()) / (prof[heat_cols].max() - prof[heat_cols].min()).replace(0, 1)
    text = prof[heat_cols].apply(lambda s: s.map(lambda v: f"{v:.0%}" if s.name.startswith(("%", "Taux"))
                                                 else f"{v:,.0f}".replace(",", " ")))
    fig = go.Figure(go.Heatmap(z=norm.values, x=heat_cols, y=norm.index, text=text.values, texttemplate="%{text}",
                               colorscale="YlGnBu", showscale=False))
    fig.update_layout(title="Profil comparé des segments (couleur = position relative)", height=400,
                      margin=dict(t=50, b=10))
    c2.plotly_chart(fig)

    c1, c2 = st.columns(2)
    sub = df.groupby("persona")["y"].apply(lambda s: (s == "yes").mean()).reindex(selected).sort_values()
    fig = go.Figure(go.Bar(x=sub.values, y=sub.index, orientation="h", text=[f"{v:.1%}" for v in sub.values],
                           marker_color=[PERSONA_COLORS.get(p) for p in sub.index]))
    fig.add_vline(x=GLOBAL_SUB, line_dash="dash", annotation_text="moyenne")
    fig.update_layout(title="Appétence au dépôt à terme par segment", xaxis_tickformat=".0%", height=380)
    c1.plotly_chart(fig)

    var = c2.selectbox("Composition des segments selon :", ["job_group", "marital", "education", "age_group",
                                                            "nb_credits"],
                       format_func=lambda v: {"job_group": "Groupe socio-professionnel", "marital": "Statut marital",
                                              "education": "Niveau d'études", "age_group": "Tranche d'âge",
                                              "nb_credits": "Nombre de crédits (immo + perso)"}[v])
    compo = pd.crosstab(df["persona"], df[var].map(lambda v: label(var, v)), normalize="index").reindex(selected)
    fig = px.bar(compo, orientation="h", barmode="stack")
    fig.update_layout(height=380, xaxis_tickformat=".0%", legend_title="", yaxis_title="",
                      legend=dict(orientation="h", y=-0.2))
    c2.plotly_chart(fig)

    c1, c2 = st.columns(2)
    fig = px.box(df.sample(min(len(df), 8000), random_state=0), x="persona", y="age", color="persona",
                 color_discrete_map=PERSONA_COLORS, category_orders={"persona": selected})
    fig.update_layout(title="Âge par segment", showlegend=False, height=380)
    c1.plotly_chart(fig)
    sample = df.sample(min(len(df), 5000), random_state=0)
    ev = meta["pca_explained_variance"]
    fig = px.scatter(sample, x="pc1", y="pc2", color="persona", opacity=0.5, color_discrete_map=PERSONA_COLORS,
                     category_orders={"persona": selected})
    fig.update_traces(marker_size=4)
    fig.update_layout(title="Projection ACP 2D des clients", height=380,
                      xaxis_title=f"Axe 1 ({ev[0]:.0%})", yaxis_title=f"Axe 2 ({ev[1]:.0%})", legend_title="")
    c2.plotly_chart(fig)

    with st.expander("Tableau détaillé des profils"):
        st.dataframe(prof.style.format({c: "{:.1%}" for c in prof.columns if c.startswith(("%", "Taux"))}
                                       | {"Nb clients": "{:,.0f}", "Âge moyen": "{:.1f}", "Solde médian (€)": "{:,.0f}",
                                          "Solde moyen (€)": "{:,.0f}", "Lift souscription": "{:.2f}"}))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Qualité & dérive
# ════════════════════════════════════════════════════════════════════════════
else:
    st.title("Qualité de la segmentation & suivi de la dérive")
    q = meta["quality"]
    k = st.columns(4)
    k[0].metric("Nombre de segments (k)", meta["k"], border=True)
    k[1].metric("Silhouette (one-hot)", f"{q['silhouette_onehot']:.3f}", border=True,
                help="Entre −1 et 1 ; des valeurs faibles sont fréquentes sur des données clients (continuum)")
    k[2].metric("Davies-Bouldin", f"{q['davies_bouldin_onehot']:.2f}", border=True, help="Plus faible = mieux séparé")
    k[3].metric("γ (poids des catégorielles)", f"{meta['gamma']:.2f}", border=True)
    st.info("💡 Les métriques internes restent modestes : la clientèle forme un continuum plutôt que des groupes "
            "bien séparés. La valeur d'une segmentation se juge d'abord à son utilité métier et à sa stabilité.")

    st.markdown("---")
    st.subheader("Simulation d'un nouveau lot de clients")
    st.caption("Construisez un lot dont la population a dérivé (par ex. après une campagne digitale), "
               "affectez-le via l'API et comparez-le au portefeuille de référence.")
    s1, s2, s3 = st.columns(3)
    young_share = s1.slider("Part de clients de moins de 35 ans", 0.0, 1.0,
                            float((clients["age"] < 35).mean()), 0.05)
    balance_change = s2.slider("Variation des soldes (%)", -50, 100, 0, 10) / 100
    housing_share = s3.slider("Part de crédits immobiliers", 0.0, 1.0, float((clients["housing"] == "yes").mean()), 0.05)

    n = 3000
    rng = np.random.default_rng(0)
    young = clients[clients["age"] < 35].sample(int(n * young_share), random_state=0, replace=True)
    older = clients[clients["age"] >= 35].sample(n - len(young), random_state=0, replace=True)
    batch = pd.concat([young, older])[["age", "balance", "job", "marital", "education", "default", "housing",
                                       "loan"]].reset_index(drop=True)
    batch["balance"] = (batch["balance"] * (1 + balance_change)).round()
    has_housing = batch.index[batch["housing"] == "yes"]
    n_target = int(round(housing_share * len(batch)))
    if n_target > len(has_housing):
        others = batch.index[batch["housing"] == "no"]
        batch.loc[rng.choice(others, n_target - len(has_housing), replace=False), "housing"] = "yes"
    elif n_target < len(has_housing):
        batch.loc[rng.choice(has_housing, len(has_housing) - n_target, replace=False), "housing"] = "no"

    res = api_post("/assign-batch", batch.to_dict(orient="records"))
    if res:
        batch["persona"] = [r["persona"] for r in res["resultats"]]
        rep = pd.DataFrame({"Référence": clients["persona"].value_counts(normalize=True),
                            "Nouveau lot": batch["persona"].value_counts(normalize=True)}).fillna(0).reindex(PERSONA_ORDER)
        c1, c2 = st.columns([1.3, 1])
        fig = px.bar(rep.reset_index(names="persona"), x="persona", y=["Référence", "Nouveau lot"], barmode="group",
                     color_discrete_sequence=["#4C72B0", "#DD8452"])
        fig.update_layout(title="Répartition des segments : référence vs nouveau lot", yaxis_tickformat=".0%",
                          height=380, legend_title="", xaxis_title="", yaxis_title="Part des clients")
        c1.plotly_chart(fig)

        rows = [{"Indicateur": "Répartition des segments", "PSI": psi_categorical(clients["persona"], batch["persona"])}]
        rows += [{"Indicateur": VAR_LABELS[c], "PSI": psi(clients[c], batch[c])} for c in ["age", "balance"]]
        rows += [{"Indicateur": VAR_LABELS.get(c, c), "PSI": psi_categorical(clients[c], batch[c])}
                 for c in ["marital", "education", "housing", "loan"]]
        rows += [{"Indicateur": "Profession", "PSI": psi_categorical(clients["job"], batch["job"])}]
        psi_df = pd.DataFrame(rows)
        psi_df["Statut"] = psi_df["PSI"].apply(psi_flag)
        c2.dataframe(psi_df.style.format({"PSI": "{:.3f}"}), hide_index=True, height=320)
        seg_psi = psi_df.loc[0, "PSI"]
        c2.metric("PSI de la répartition des segments", f"{seg_psi:.3f}", psi_flag(seg_psi), delta_color="off",
                  border=True)
        st.caption("Règle usuelle : PSI < 0,10 stable · 0,10–0,25 dérive modérée · > 0,25 dérive significative "
                   "(analyser le changement de mix client ; re-segmenter si la dérive persiste).")
