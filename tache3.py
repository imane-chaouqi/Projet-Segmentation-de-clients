"""
Author: Wiame Boujamaai
Role: Intelligent Cluster Analysis and Validation
Goals: Customer Segmentation for Marketing Strategy
Objectives:
Business interpretation of clusters
Mathematical validation using the Silhouette Score
Comparative analysis and critique
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. CHARGEMENT DES DONNÉES AVEC CLUSTERS
print("="*60)
print("CHARGEMENT DES DONNÉES")
print("="*60)

df = pd.read_csv("Mall_Customers_With_Clusters.csv")

# Conversion inverse pour avoir les valeurs originales (non normalisées)
# On recrée les données originales à partir du fichier initial
df_original = pd.read_csv("Mall_Customers.csv")
df_original['Cluster'] = df['Cluster']

print(f"Nombre total de clients : {len(df)}")
print(f"Nombre de clusters : {df['Cluster'].nunique()}")
print("\nRépartition des clients par cluster :")
print(df_original['Cluster'].value_counts().sort_index())

# 2. INTERPRÉTATION BUSINESS DES CLUSTERS
print("\n" + "="*60)
print("1. INTERPRÉTATION BUSINESS DES CLUSTERS")
print("="*60)

# Analyse statistique par cluster
cluster_stats = df_original.groupby('Cluster').agg({
    'Age': ['mean', 'std'],
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std'],
    'Gender': lambda x: (x == 'Male').sum() / len(x) * 100  # Pourcentage d'hommes
}).round(2)

cluster_stats.columns = ['Age_mean', 'Age_std', 
                         'Income_mean', 'Income_std',
                         'Spending_mean', 'Spending_std',
                         'Male_percent']

print("\nStatistiques descriptives par cluster :")
print(cluster_stats)

# Interprétation intelligente des clusters
print("\n" + "-"*40)
print("INTERPRÉTATION DES SEGMENTS CLIENT :")
print("-"*40)

for cluster_id in sorted(df_original['Cluster'].unique()):
    cluster_data = df_original[df_original['Cluster'] == cluster_id]
    
    # Calcul des caractéristiques
    age_mean = cluster_data['Age'].mean()
    income_mean = cluster_data['Annual Income (k$)'].mean()
    spending_mean = cluster_data['Spending Score (1-100)'].mean()
    male_percent = (cluster_data['Gender'] == 'Male').mean() * 100
    
    # Détermination des catégories
    age_cat = "jeunes" if age_mean < 35 else "âgés" if age_mean > 50 else "adultes"
    income_cat = "faible" if income_mean < 40 else "élevé" if income_mean > 70 else "moyen"
    spending_cat = "faible" if spending_mean < 40 else "élevé" if spending_mean > 60 else "moyen"
    
    print(f"\n CLUSTER {cluster_id} ({len(cluster_data)} clients) :")
    print(f"   • Âge moyen : {age_mean:.1f} ans ({age_cat})")
    print(f"   • Revenu moyen : ${income_mean:.1f}k ({income_cat})")
    print(f"   • Score de dépense moyen : {spending_mean:.1f}/100 ({spending_cat})")
    print(f"   • Proportion hommes : {male_percent:.1f}%")
    
    # Profil business
    if income_mean < 40 and spending_mean < 40:
        profile = "Clients occasionnels / prudents"
        recommendations = ["Offres bas prix", "Fidélisation progressive", "Produits essentiels"]
    elif income_mean > 70 and spending_mean > 60:
        profile = "Clients premium / dépensiers"
        recommendations = ["Produits haut de gamme", "Services VIP", "Expériences exclusives"]
    elif income_mean > 70 and spending_mean < 40:
        profile = "Clients à fort potentiel / épargnants"
        recommendations = ["Éducation financière", "Produits d'investissement", "Services premium"]
    elif age_mean < 35 and spending_mean > 60:
        profile = "Jeunes impulsifs / influenceurs"
        recommendations = ["Marketing digital", "Tendances", "Abonnements"]
    elif age_mean > 50 and spending_mean > 60:
        profile = "Retraités actifs / dépensiers"
        recommendations = ["Loisirs", "Voyages", "Santé/bien-être"]
    else:
        profile = "Clients standards"
        recommendations = ["Offres équilibrées", "Fidélité", "Cross-selling"]
    
    print(f"    Profil business : {profile}")
    print(f"    Recommandations marketing : {', '.join(recommendations)}")

# 3. VALIDATION MATHÉMATIQUE DU CLUSTERING
print("\n" + "="*60)
print("2. VALIDATION DU CLUSTERING (SILHOUETTE SCORE)")
print("="*60)

# Préparation des données pour le calcul du score
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
labels = df['Cluster'].values

# Calcul du Silhouette Score
sil_score = silhouette_score(X, labels)
print(f"\n Silhouette Score pour K=4 : {sil_score:.3f}")

# Interprétation du score
print("\n Interprétation du Silhouette Score :")
print(f"Score obtenu : {sil_score:.3f}")
if sil_score > 0.7:
    print(" Excellent : Clusters bien séparés et distincts")
elif sil_score > 0.5:
    print(" Bon : Structure de clusters raisonnable")
elif sil_score > 0.25:
    print(" Faible : Chevauchement entre certains clusters")
else:
    print(" Mauvais : Pas de structure claire")

# Silhouette plot pour chaque cluster
print("\n Analyse détaillée par cluster :")
silhouette_vals = silhouette_samples(X, labels)

fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10

for i in range(4):
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.tab10(float(i) / 4)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.set_xlabel("Coefficient de silhouette")
ax.set_ylabel("Cluster")
ax.set_title("Diagramme de silhouette - Analyse de qualité")
ax.axvline(x=sil_score, color="red", linestyle="--")
ax.set_yticks([])

plt.tight_layout()
plt.show()

# 4. ANALYSE COMPARATIVE (K=3 vs K=4 vs K=5)
print("\n" + "="*60)
print("3. ANALYSE COMPARATIVE DU NOMBRE DE CLUSTERS")
print("="*60)

from sklearn.cluster import KMeans

# Données originales (non normalisées pour K-means)
X_original = df_original[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Évaluation de différents K
k_values = [2, 3, 4, 5, 6]
inertia_values = []
silhouette_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = kmeans.fit_predict(X_original)
    
    inertia_values.append(kmeans.inertia_)
    
    if k > 1:  # Silhouette score nécessite au moins 2 clusters
        sil = silhouette_score(X_original, labels_k)
        silhouette_values.append(sil)
    else:
        silhouette_values.append(0)

# Visualisation comparative
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique Elbow Method
axes[0].plot(k_values, inertia_values, 'bo-', linewidth=2)
axes[0].set_xlabel('Nombre de clusters (K)')
axes[0].set_ylabel('Inertie')
axes[0].set_title('Méthode du Coude (Elbow Method)')
axes[0].grid(True, alpha=0.3)

# Graphique Silhouette Score
axes[1].plot(k_values[1:], silhouette_values[1:], 'ro-', linewidth=2)
axes[1].set_xlabel('Nombre de clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Score de Silhouette par valeur de K')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n Comparaison des performances :")
print("-"*40)
for i, k in enumerate(k_values):
    if k > 1:
        print(f"K = {k} : Inertie = {inertia_values[i]:.0f}, Silhouette = {silhouette_values[i]:.3f}")

# 5. ANALYSE CRITIQUE ET LIMITES
print("\n" + "="*60)
print("4. ANALYSE CRITIQUE ET LIMITES")
print("="*60)

print("\n AVANTAGES DE K=4 :")
print("1. Segments business clairs et interprétables")
print("2. Bon équilibre entre granularité et simplicité")
print("3. Correspond à des profils marketing identifiables")
print("4. Silhouette score acceptable pour un dataset de cette taille")

print("\n LIMITES DE K-MEANS :")
print("1. Sensible aux outliers (clients extrêmes)")
print("2. Assume des clusters sphériques de taille similaire")
print("3. Nécessite de spécifier K à l'avance")
print("4. Sensible à l'initialisation aléatoire")

print("\n ANALYSE DES OUTLIERS :")
# Identification des points avec silhouette négative
outliers = df_original[silhouette_vals < 0]
print(f"Nombre de points mal classés (silhouette < 0) : {len(outliers)}")
if len(outliers) > 0:
    print("Ces clients pourraient appartenir à d'autres clusters ou être atypiques")

# 6. VISUALISATIONS FINALES POUR LA PRÉSENTATION
print("\n" + "="*60)
print("5. VISUALISATIONS POUR LA PRÉSENTATION")
print("="*60)

fig = plt.figure(figsize=(16, 10))

# 1. Distribution 2D - Revenu vs Dépenses
ax1 = fig.add_subplot(221)
scatter = ax1.scatter(df_original['Annual Income (k$)'], 
                      df_original['Spending Score (1-100)'],
                      c=df_original['Cluster'], cmap='viridis', alpha=0.7)
ax1.set_xlabel('Revenu annuel ($k)')
ax1.set_ylabel('Score de dépense')
ax1.set_title('Segmentation clients : Revenu vs Dépenses')
plt.colorbar(scatter, ax=ax1, label='Cluster')

# 2. Distribution 2D - Âge vs Dépenses
ax2 = fig.add_subplot(222)
scatter2 = ax2.scatter(df_original['Age'], 
                       df_original['Spending Score (1-100)'],
                       c=df_original['Cluster'], cmap='viridis', alpha=0.7)
ax2.set_xlabel('Âge')
ax2.set_ylabel('Score de dépense')
ax2.set_title('Segmentation clients : Âge vs Dépenses')
plt.colorbar(scatter2, ax=ax2, label='Cluster')

# 3. Boxplot par cluster pour le revenu
ax3 = fig.add_subplot(223)
df_original.boxplot(column='Annual Income (k$)', by='Cluster', ax=ax3)
ax3.set_title('Distribution du revenu par cluster')
ax3.set_ylabel('Revenu annuel ($k)')
ax3.set_xlabel('Cluster')

# 4. Boxplot par cluster pour les dépenses
ax4 = fig.add_subplot(224)
df_original.boxplot(column='Spending Score (1-100)', by='Cluster', ax=ax4)
ax4.set_title('Distribution du score de dépense par cluster')
ax4.set_ylabel('Score de dépense')
ax4.set_xlabel('Cluster')

plt.suptitle('ANALYSE COMPLÈTE DES CLUSTERS - PROJET DE SEGMENTATION CLIENTS', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. SYNTHÈSE FINALE
print("\n" + "="*60)
print("SYNTHÈSE FINALE")
print("="*60)

print("\n CONCLUSIONS :")
print("1. Les 4 clusters identifiés ont une signification business claire")
print("2. Le Silhouette Score de", f"{sil_score:.3f}", "indique une structure raisonnable")
print("3. K=4 semble être un bon compromis entre interprétabilité et précision")
print("4. Chaque segment nécessite des stratégies marketing différenciées")

print("\n RECOMMANDATIONS POUR L'ÉQUIPE MARKETING :")
print("• Développer des campagnes ciblées pour chaque segment")
print("• Allouer le budget en fonction du potentiel de chaque cluster")
print("• Suivre l'évolution des clients entre les segments dans le temps")
print("• Tester des approches de personnalisation pour les clusters les plus rentables")

# 8. SAUVEGARDE DU RAPPORT D'ANALYSE
print("\n" + "="*60)
print("SAUVEGARDE DES RÉSULTATS")
print("="*60)

# Création d'un dataframe de synthèse
cluster_summary = df_original.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean'
}).round(2)

cluster_summary = cluster_summary.rename(columns={'CustomerID': 'Nombre_Clients'})
cluster_summary['Pourcentage'] = (cluster_summary['Nombre_Clients'] / len(df) * 100).round(1)

# Ajout des profils
profiles = {
    0: "Clients âgés à revenu faible/moyen",
    1: "Clients premium dépensiers",
    2: "Jeunes à faible revenu et dépenses variables",
    3: "Revenu élevé mais dépenses prudentes"
}

cluster_summary['Profil'] = cluster_summary.index.map(profiles)

print("\n Tableau de synthèse des clusters :")
print(cluster_summary)

# Sauvegarde
cluster_summary.to_csv('Cluster_Analysis_Summary.csv')
df_original.to_csv('Mall_Customers_Final_Analysis.csv', index=False)

print("\n Fichiers sauvegardés :")
print("   - 'Cluster_Analysis_Summary.csv' : Synthèse des clusters")
print("   - 'Mall_Customers_Final_Analysis.csv' : Données complètes avec clusters")
print("\n Tâche 3 terminée avec succès !")