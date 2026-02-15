import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# --------------------------------------------------
# CONFIGURATION DE LA PAGE
# --------------------------------------------------
st.set_page_config(
    page_title="Tableau de Bord de Segmentation des Clients",
    layout="wide" , 
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# BACKGROUND GRADIENT VIOLET - ROUGE - BLEU
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,
        #4b0082 0%,
        #9d174d 45%,
        #1e3a8a 100%);
    color: white;
}

.block-container {
    background-color: rgba(0,0,0,0.30);
    padding: 2rem;
    border-radius: 18px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# STYLE GLOBAL PROFESSIONNEL
# --------------------------------------------------
st.markdown("""
<style>

/* TITRE PRINCIPAL */
.main-title {
    color: white;
    font-size: 42px;
    font-weight: 800;
    text-align: center;
}

/* SOUS-TITRE */
.sub-title {
    color: #cfe8ff;
    font-size: 22px;
    text-align: center;
}

/* TITRE DE SECTION */
.section-title {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin-top: 40px;
}

/* TEXTE NORMAL */
p, li, span {
    color: #e5e5e5;
    font-size: 18px;
}

/* =========================
   BOUTON BLEU MARINE
   ========================= */
div.stButton > button {
    background: linear-gradient(135deg, #0f1f3d, #1e3a8a);
    color: white;
    font-size: 20px;
    font-weight: 700;
    padding: 12px 30px;
    border-radius: 12px;
    border: none;
    width: 100%;
    cursor: pointer;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    transform: scale(1.03);
}

</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# TITRE
# --------------------------------------------------
st.markdown("<div class='main-title'>Tableau de Bord de Segmentation des Clients</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Analyse et compréhension du comportement client</div>", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# OBJECTIF DU PROJET
# --------------------------------------------------
st.markdown("<div class='section-title'>Objectif du Projet</div>", unsafe_allow_html=True)

st.markdown("""
L’objectif principal de ce projet est de *segmenter les clients d’un centre commercial*
en groupes homogènes afin de mieux comprendre leurs comportements d’achat.

Cette segmentation permet de :
- améliorer le ciblage des campagnes marketing,
- personnaliser les offres commerciales,
- identifier les clients à forte valeur,
- faciliter la prise de décision stratégique.
""")

# --------------------------------------------------
# CHARGEMENT DES DONNÉES
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# --------------------------------------------------
# DESCRIPTION DU DATASET
# --------------------------------------------------
st.markdown("<div class='section-title'>Description du Dataset</div>", unsafe_allow_html=True)

st.markdown("""
Le dataset *Mall_Customers* contient 5 variables :

- *CustomerID* : identifiant unique du client  
- *Gender* : sexe du client  
- *Age* : âge du client  
- *Annual Income (k$)* : revenu annuel  
- *Spending Score (1–100)* : score représentant le comportement d’achat  

Dans cette étude, seules les variables *Age, **Annual Income* et
*Spending Score* sont utilisées pour la segmentation,
car elles sont numériques et directement liées au comportement d’achat.
""")

# --------------------------------------------------
# APERÇU DES DONNÉES
# --------------------------------------------------
st.markdown("<div class='section-title'>Aperçu du Dataset</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.write(df.head())

with col2:
    st.write(df.describe())

# Nombre optimal de clusters déterminé avec la méthode Elbow
k = 4


# --------------------------------------------------
# PRÉTRAITEMENT DES DONNÉES
# --------------------------------------------------
st.markdown("<div class='section-title'>Prétraitement des Données</div>", unsafe_allow_html=True)

st.markdown("""
Avant d’appliquer l’algorithme K-Means, les données sont *standardisées*
afin d’avoir :
- une moyenne égale à 0,
- un écart-type égal à 1.

Cette étape est indispensable car K-Means est basé sur le calcul des distances.
""")

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# MODÈLE K-MEANS
# --------------------------------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# --------------------------------------------------
# VALIDATION VISUELLE DES CLUSTERS (COUDE + SILHOUETTE)
# --------------------------------------------------
st.markdown("<div class='section-title'>Validation Visuelle des Clusters</div>", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

# ======================
# MÉTHODE DU COUDE (GAUCHE)
# ======================
with col_left:
    st.markdown("### Méthode du Coude")

    inertia = []
    K_range = range(1, 11)

    for i in K_range:
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(6,4))
    ax_elbow.plot(K_range, inertia, marker='o')
    ax_elbow.set_xlabel("Nombre de clusters (K)")
    ax_elbow.set_ylabel("Inertie")
    ax_elbow.set_title("Détermination du nombre optimal de clusters")
    st.pyplot(fig_elbow)

# ======================
# SILHOUETTE (DROITE)
# ======================
with col_right:
    st.markdown("### Diagramme de Silhouette")

    fig_silhouette, ax = plt.subplots(figsize=(6,4))

    sample_silhouette_values = silhouette_samples(X_scaled, df['Cluster'])
    y_lower = 10

    for i in range(k):
        vals = sample_silhouette_values[df['Cluster'] == i]
        vals.sort()
        size_cluster_i = vals.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    sil_score = silhouette_score(X_scaled, df['Cluster'])
    ax.axvline(x=sil_score, color="red", linestyle="--")
    ax.set_xlabel("Coefficient de silhouette")
    ax.set_ylabel("Cluster")
    ax.set_title("Qualité de la segmentation")
    st.pyplot(fig_silhouette)

# --------------------------------------------------
# 2️⃣ VISUALISATION DES CLUSTERS
# --------------------------------------------------
st.markdown("<div class='section-title'>Visualisation des Clusters</div>", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x=df['Annual Income (k$)'],
        y=df['Spending Score (1-100)'],
        hue=df['Cluster'],
        palette='viridis',
        ax=ax2
    )
    ax2.set_title("Revenu Annuel vs Score de Dépense")
    st.pyplot(fig2)

with col4:
    fig3, ax3 = plt.subplots()
    sns.scatterplot(
        x=df['Age'],
        y=df['Spending Score (1-100)'],
        hue=df['Cluster'],
        palette='viridis',
        ax=ax3
    )
    ax3.set_title("Âge vs Score de Dépense")
    st.pyplot(fig3)


# --------------------------------------------------
# 4️⃣ VALIDATION MATHÉMATIQUE
# --------------------------------------------------
st.markdown("<div class='section-title'>Validation Mathématique</div>", unsafe_allow_html=True)

st.write(f"Silhouette Score pour K = {k} : *{sil_score:.3f}*")

# Ajout de la ligne demandée
st.write("Acceptable / chevauchement modéré")

# --------------------------------------------------
# 5️⃣ ANALYSE AVANCÉE DE VALIDATION DES CLUSTERS
# --------------------------------------------------
st.markdown("<div class='section-title'>Analyse Avancée de Validation</div>", unsafe_allow_html=True)

col5, col6 = st.columns(2)

# ==========================
# ELBOW METHOD DETAILLEE
# ==========================
with col5:
    inertia_values = []
    K_values = range(2, 7)

    for i in K_values:
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia_values.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(K_values, inertia_values, marker='o')
    ax_elbow.set_title("Elbow Method")
    ax_elbow.set_xlabel("K")
    ax_elbow.set_ylabel("Inertie")
    st.pyplot(fig_elbow)

# ==========================
# SILHOUETTE PAR K
# ==========================
with col6:
    silhouette_scores = []

    for i in K_values:
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

    fig_sil, ax_sil = plt.subplots()
    ax_sil.plot(K_values, silhouette_scores, marker='o')
    ax_sil.set_title("Silhouette par K")
    ax_sil.set_xlabel("K")
    ax_sil.set_ylabel("Score")
    st.pyplot(fig_sil)

# --------------------------------------------------
# PROFIL MOYEN PAR CLUSTER
# --------------------------------------------------
st.markdown("<div class='section-title'>Profil Moyen par Cluster</div>", unsafe_allow_html=True)

cluster_profile = df.groupby('Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean()
st.dataframe(cluster_profile)

# --------------------------------------------------
# INTERPRÉTATION BUSINESS
# --------------------------------------------------
st.markdown("<div class='section-title'>Interprétation Business des Segments</div>", unsafe_allow_html=True)

st.markdown("""
- *Cluster 0* : clients à revenu moyen avec un comportement d’achat modéré  
- *Cluster 1* : clients à faible revenu et faible dépense  
- *Cluster 2* : clients à revenu élevé et forte dépense (clients premium)  
- *Cluster 3* : clients jeunes avec une forte impulsivité d’achat  

Cette segmentation aide l’entreprise à adapter ses stratégies marketing
et à maximiser la valeur client.
""")

# --------------------------------------------------
# TEST PROFIL CLIENT (POUR LA SOCIÉTÉ)
# --------------------------------------------------
st.markdown("<div class='section-title'>Test Profil Client</div>", unsafe_allow_html=True)

st.markdown("""
Cette section permet à un *membre de la société* de tester le profil d’un client réel
en entrant ses informations (âge, revenu annuel et score de dépense).  

Le système retournera le *cluster estimé* et une *interprétation business* adaptée.
""")

age_input = st.number_input("Âge du client", min_value=0, max_value=100, value=30)
income_input = st.number_input("Revenu Annuel du client (k$)", min_value=0, max_value=200, value=50)
score_input = st.number_input("Score de Dépense du client (1-100)", min_value=1, max_value=100, value=50)

if st.button("Tester le Profil du Client"):

    user_data = pd.DataFrame([[age_input, income_input, score_input]],
                             columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    user_scaled = scaler.transform(user_data)
    user_cluster = kmeans.predict(user_scaled)[0]

    st.write(f"Le client appartient au *Cluster {user_cluster}*")

    interpretations = {
        0: "Clients à revenu moyen avec un comportement d’achat modéré.",
        1: "Clients à faible revenu et faible dépense.",
        2: "Clients à revenu élevé et forte dépense (clients premium).",
        3: "Clients jeunes avec une forte impulsivité d’achat."
    }

    st.info(f"Interprétation Business : {interpretations.get(user_cluster, 'Segment inconnu')}")

# --------------------------------------------------
# CONCLUSION
# --------------------------------------------------
st.divider()

st.markdown("""
*Projet académique – Data Science & Machine Learning*  
Outils utilisés : Python, Pandas, Scikit-learn, Streamlit, Matplotlib, Seaborn
""")