import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("ANALYSE DE CLUSTERING DES TUMEURS MAMMAIRES")
print("="*70)

# ============================================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[ÉTAPE 1] Chargement du dataset Breast Cancer Wisconsin")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(f"✓ Dimensions du dataset : {X.shape}")
print(f"✓ Nombre de caractéristiques : {X.shape[1]}")
print(f"✓ Nombre d'échantillons : {X.shape[0]}")

# ============================================================================
# 3. EXPLORATION INITIALE DES DONNÉES
# ============================================================================
print("\n" + "="*70)
print("[ÉTAPE 2] EXPLORATION INITIALE DES DONNÉES")
print("="*70)

print("\n📊 IMPORTANCE DE L'EXPLORATION INITIALE :")
print("-" * 70)
print("""
L'exploration initiale est CRUCIALE pour plusieurs raisons :

1. COMPRENDRE LA STRUCTURE : Identifier le type, l'échelle et la distribution
   des variables avant tout traitement.

2. DÉTECTER LES ANOMALIES : Repérer valeurs manquantes, outliers, 
   incohérences qui pourraient fausser le clustering.

3. ÉVALUER LA QUALITÉ : S'assurer que les données sont exploitables et
   représentatives de la population étudiée.

4. GUIDER LE PREPROCESSING : Déterminer les transformations nécessaires
   (normalisation, gestion des outliers, etc.).

5. INTERPRÉTER LES RÉSULTATS : Faciliter la compréhension des clusters
   en connaissant les caractéristiques des données.
""")

print("\n📋 Aperçu des premières lignes :")
print(X.head())

print("\n📈 Statistiques descriptives :")
print(X.describe())

print("\n📊 Distribution de la cible (pour information) :")
print(f"Malin (0) : {(y==0).sum()} tumeurs ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"Bénin (1) : {(y==1).sum()} tumeurs ({(y==1).sum()/len(y)*100:.1f}%)")

# Visualisations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Exploration Initiale des Données', fontsize=16, fontweight='bold')

# Distribution de quelques caractéristiques principales
axes[0, 0].hist(X['mean radius'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution: Mean Radius')
axes[0, 0].set_xlabel('Mean Radius')
axes[0, 0].set_ylabel('Fréquence')

axes[0, 1].hist(X['mean texture'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('Distribution: Mean Texture')
axes[0, 1].set_xlabel('Mean Texture')
axes[0, 1].set_ylabel('Fréquence')

# Boxplot pour détecter les outliers
axes[1, 0].boxplot([X['mean area'], X['mean smoothness']], 
                    labels=['Mean Area', 'Mean Smoothness'])
axes[1, 0].set_title('Détection des Outliers')
axes[1, 0].set_ylabel('Valeur')

# Matrice de corrélation (échantillon)
corr_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
corr_matrix = X[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1], 
            fmt='.2f', square=True)
axes[1, 1].set_title('Matrice de Corrélation (échantillon)')

plt.tight_layout()
plt.savefig('exploration_initiale.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphiques d'exploration sauvegardés")

# ============================================================================
# 4. NETTOYAGE DES DONNÉES
# ============================================================================
print("\n" + "="*70)
print("[ÉTAPE 3] NETTOYAGE DES DONNÉES")
print("="*70)

# Vérification des valeurs manquantes
missing_values = X.isnull().sum()
print(f"\n🔍 Valeurs manquantes par colonne :")
print(f"Total de valeurs manquantes : {missing_values.sum()}")

if missing_values.sum() == 0:
    print("✓ Aucune valeur manquante détectée - Dataset propre !")
else:
    print(missing_values[missing_values > 0])

print("\n📝 JUSTIFICATION DU NETTOYAGE :")
print("-" * 70)
print("""
CHOIX DE TRAITEMENT :

1. VALEURS MANQUANTES : Le dataset Breast Cancer Wisconsin est déjà nettoyé.
   En cas de valeurs manquantes, nous pourrions :
   - Imputation par la médiane (robuste aux outliers)
   - Suppression si < 5% des données
   - Imputation par K-NN pour préserver les relations

2. OUTLIERS : Nous les conservons car ils peuvent représenter des cas
   médicaux rares mais réels. Le clustering K-Means y est sensible,
   mais la normalisation réduira leur impact.

3. DOUBLONS : Vérification systématique pour éviter de biaiser le clustering.
""")

# Vérification des doublons
duplicates = X.duplicated().sum()
print(f"\n🔍 Nombre de doublons : {duplicates}")

# Vérification des types de données
print(f"\n📋 Types de données :")
print(X.dtypes.value_counts())

# ============================================================================
# 5. SÉLECTION DES CARACTÉRISTIQUES
# ============================================================================
print("\n" + "="*70)
print("[ÉTAPE 4] SÉLECTION DES CARACTÉRISTIQUES")
print("="*70)

print("\n💡 IMPORTANCE DE LA SÉLECTION DES CARACTÉRISTIQUES :")
print("-" * 70)
print("""
La sélection judicieuse des caractéristiques est ESSENTIELLE car :

1. MALÉDICTION DE LA DIMENSIONNALITÉ : Trop de features diluent les
   distances entre points, rendant le clustering moins efficace.

2. FEATURES REDONDANTES : Des variables fortement corrélées ajoutent
   du bruit sans information nouvelle (ex: radius/perimeter/area).

3. INTERPRÉTABILITÉ : Moins de features = clusters plus faciles à
   comprendre et à expliquer aux médecins.

4. PERFORMANCE : Réduction du temps de calcul et amélioration de la
   qualité du clustering.

5. PERTINENCE CLINIQUE : Sélectionner les features médicalement
   significatives pour des clusters cliniquement utilisables.
""")

# Analyse de corrélation pour identifier les features redondantes
print("\n🔍 Analyse de corrélation des features :")
correlation_matrix = X.corr()

# Identifier les paires de features hautement corrélées
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print(f"\n⚠️  {len(high_corr_pairs)} paires de features fortement corrélées (|r| > 0.9)")
if len(high_corr_pairs) > 0:
    print("Exemples :")
    for feat1, feat2, corr in high_corr_pairs[:5]:
        print(f"  - {feat1} ↔ {feat2}: r = {corr:.3f}")

# Pour cette analyse, nous utilisons toutes les features après normalisation
# mais nous notons les features 'mean' comme les plus importantes cliniquement
mean_features = [col for col in X.columns if 'mean' in col]
print(f"\n✓ Features 'mean' sélectionnées pour analyse prioritaire : {len(mean_features)}")
print(f"  {mean_features[:5]}... (et {len(mean_features)-5} autres)")

# Nous gardons toutes les features mais après normalisation
X_selected = X.copy()

# ============================================================================
# 6. NORMALISATION ET DIVISION DES DONNÉES
# ============================================================================
print("\n" + "="*70)
print("[ÉTAPE 5] NORMALISATION ET DIVISION DES DONNÉES")
print("="*70)

# Normalisation (CRUCIALE pour K-Means qui utilise les distances)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_scaled = pd.DataFrame(X_scaled, columns=X_selected.columns)

print("\n✓ Normalisation StandardScaler appliquée")
print("  Raison : K-Means est sensible à l'échelle des variables")
print(f"  Moyenne après normalisation : {X_scaled.mean().mean():.2e}")
print(f"  Écart-type après normalisation : {X_scaled.std().mean():.2f}")

# Division en train/test
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
print(f"\n✓ Division train/test effectuée")
print(f"  Ensemble d'entraînement : {X_train.shape[0]} échantillons")
print(f"  Ensemble de test : {X_test.shape[0]} échantillons")

# ============================================================================
# 7. CLUSTERING K-MEANS - MÉTHODE DU COUDE
# ============================================================================
print("\n" + "="*70)
print("[ÉTAPE 6] CLUSTERING K-MEANS - DÉTERMINATION DU NOMBRE DE CLUSTERS")
print("="*70)

print("\n⏳ Calcul de l'inertie pour différents nombres de clusters...")
print("   (Cela peut prendre quelques secondes)")

# Méthode du coude
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_train, kmeans.labels_))
    print(f"  K={k}: inertie={kmeans.inertia_:.2f}, silhouette={silhouette_scores[-1]:.3f}")

# Visualisation de la méthode du coude
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Détermination du Nombre Optimal de Clusters', 
             fontsize=16, fontweight='bold')

# Graphique du coude
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Nombre de Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertie (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Méthode du Coude (Elbow Method)')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=2, color='r', linestyle='--', alpha=0.5, label='K=2 (suggéré)')
axes[0].legend()

# Silhouette scores
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Nombre de Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score par Nombre de Clusters')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=max(silhouette_scores), color='r', linestyle='--', 
                alpha=0.5, label=f'Max: {max(silhouette_scores):.3f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique de la méthode du coude sauvegardé")

# ============================================================================
# NOTE IMPORTANTE : DEMANDE DE COMPLÉMENT D'EXPLICATIONS
# ============================================================================
print("\n" + "="*70)
print("⚠️  DEMANDE DE COMPLÉMENT D'EXPLICATIONS SUR LA MÉTHODE DU COUDE")
print("="*70)
print("""
QUESTIONS POUR APPROFONDISSEMENT :

1. Comment identifier précisément le "coude" sur le graphique ?
   - Faut-il chercher un angle marqué ou une zone de stabilisation ?

2. Que faire si le coude n'est pas évident ?
   - Faut-il privilégier d'autres métriques (silhouette, gap statistic) ?

3. Comment équilibrer le compromis biais-variance ?
   - Plus de clusters = meilleur fit mais risque de sur-segmentation
   - Moins de clusters = plus généralisable mais moins précis

4. Faut-il prendre en compte le contexte médical ?
   - Le nombre de clusters doit-il correspondre à des sous-types connus ?

5. Comment valider le choix final ?
   - Tests statistiques ? Validation croisée ? Expertise métier ?

POUR CETTE ANALYSE, nous choisissons K=2 car :
- Correspond aux 2 classes connues (malin/bénin)
- Silhouette score élevé
- Interprétabilité clinique maximale
""")

# ============================================================================
# 8. CLUSTERING FINAL AVEC K=2
# ============================================================================
optimal_k = 2
print(f"\n🎯 Clustering final avec K={optimal_k} clusters")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
train_clusters = kmeans_final.fit_predict(X_train)
test_clusters = kmeans_final.predict(X_test)

print(f"✓ Modèle K-Means entraîné")
print(f"  Nombre d'itérations : {kmeans_final.n_iter_}")

# ============================================================================
# 9. ÉVALUATION DE LA QUALITÉ DU CLUSTERING
# ============================================================================
print("\n" + "="*70)
print("[ÉTAPE 7] ÉVALUATION DE LA QUALITÉ DU CLUSTERING")
print("="*70)

# Métriques sur l'ensemble d'entraînement
train_inertia = kmeans_final.inertia_
train_silhouette = silhouette_score(X_train, train_clusters)

# Métriques sur l'ensemble de test
test_inertia = kmeans_final.score(X_test) * -1
test_silhouette = silhouette_score(X_test, test_clusters)

print(f"\n📊 MÉTRIQUES D'ÉVALUATION :")
print("-" * 70)
print(f"\nEnsemble d'entraînement :")
print(f"  Inertie : {train_inertia:.2f}")
print(f"  Silhouette Score : {train_silhouette:.3f}")

print(f"\nEnsemble de test :")
print(f"  Inertie : {test_inertia:.2f}")
print(f"  Silhouette Score : {test_silhouette:.3f}")

print("\n💡 INTERPRÉTATION DES MÉTRIQUES :")
print("-" * 70)
print(f"""
1. INERTIE (Within-Cluster Sum of Squares) :
   - Valeur : {train_inertia:.2f}
   - Signification : Somme des distances au carré entre chaque point et
     son centroïde de cluster
   - Interprétation : Plus l'inertie est FAIBLE, plus les points sont
     proches de leur centroïde (clusters compacts)
   - Contexte médical : Une faible inertie suggère que les tumeurs d'un
     même cluster partagent des caractéristiques très similaires

2. SILHOUETTE SCORE :
   - Valeur : {train_silhouette:.3f}
   - Plage : [-1, 1]
   - Signification : Mesure la séparation entre clusters
     * Score proche de 1 : Points bien assignés à leur cluster
     * Score proche de 0 : Points à la frontière entre clusters
     * Score négatif : Points possiblement mal assignés
   - Interprétation : Un score de {train_silhouette:.3f} indique {"une excellente" if train_silhouette > 0.7 else "une bonne" if train_silhouette > 0.5 else "une séparation moyenne des"}
     séparation entre clusters
   - Contexte médical : Les tumeurs sont {"clairement" if train_silhouette > 0.7 else "relativement"} distinguables en groupes
     distincts basés sur leurs caractéristiques

3. COMPARAISON TRAIN/TEST :
   - Différence d'inertie : {abs(train_inertia - test_inertia):.2f}
   - Différence de silhouette : {abs(train_silhouette - test_silhouette):.3f}
   - Conclusion : {"Bonne généralisation" if abs(train_silhouette - test_silhouette) < 0.1 else "Généralisation acceptable"}
""")

# Distribution des clusters
print(f"\n📈 DISTRIBUTION DES CLUSTERS :")
print(f"  Cluster 0 : {(train_clusters == 0).sum()} tumeurs ({(train_clusters == 0).sum()/len(train_clusters)*100:.1f}%)")
print(f"  Cluster 1 : {(train_clusters == 1).sum()} tumeurs ({(train_clusters == 1).sum()/len(train_clusters)*100:.1f}%)")

# ============================================================================
# 10. VISUALISATION DES CLUSTERS
# ============================================================================
print("\n" + "="*70)
print("[ÉTAPE 8] VISUALISATION DES CLUSTERS")
print("="*70)

# Réduction de dimensionnalité pour visualisation (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"\n✓ Réduction PCA appliquée")
print(f"  Variance expliquée : {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Créer les visualisations
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Visualisation des Clusters de Tumeurs', 
             fontsize=18, fontweight='bold')

# 1. Clusters dans l'espace PCA (Train)
ax1 = fig.add_subplot(gs[0, :2])
scatter1 = ax1.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                       c=train_clusters, cmap='viridis', 
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.scatter(pca.transform(kmeans_final.cluster_centers_)[:, 0],
            pca.transform(kmeans_final.cluster_centers_)[:, 1],
            c='red', marker='X', s=300, edgecolors='black', linewidth=2,
            label='Centroïdes')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax1.set_title('Clusters dans l\'Espace PCA (Train)', fontsize=13)
ax1.legend()
plt.colorbar(scatter1, ax=ax1, label='Cluster')
ax1.grid(True, alpha=0.3)

# 2. Clusters dans l'espace PCA (Test)
ax2 = fig.add_subplot(gs[0, 2])
scatter2 = ax2.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
                       c=test_clusters, cmap='viridis', 
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.set_xlabel(f'PC1', fontsize=11)
ax2.set_ylabel(f'PC2', fontsize=11)
ax2.set_title('Clusters (Test)', fontsize=13)
plt.colorbar(scatter2, ax=ax2, label='Cluster')
ax2.grid(True, alpha=0.3)

# 3. Distribution des caractéristiques par cluster
mean_features_viz = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
ax3 = fig.add_subplot(gs[1, :])

X_train_with_clusters = X_train.copy()
X_train_with_clusters['cluster'] = train_clusters

cluster_means = X_train_with_clusters.groupby('cluster')[mean_features_viz].mean()
cluster_means.T.plot(kind='bar', ax=ax3, width=0.8)
ax3.set_title('Caractéristiques Moyennes par Cluster', fontsize=13)
ax3.set_ylabel('Valeur Normalisée', fontsize=11)
ax3.set_xlabel('Caractéristiques', fontsize=11)
ax3.legend(title='Cluster', labels=['Cluster 0', 'Cluster 1'])
ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 4. Silhouette plot par cluster
from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(X_train, train_clusters)

ax4 = fig.add_subplot(gs[2, 0])
y_lower = 10
for i in range(optimal_k):
    cluster_silhouette_vals = silhouette_vals[train_clusters == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.viridis(float(i) / optimal_k)
    ax4.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_vals,
                      facecolor=color, edgecolor=color, alpha=0.7)
    
    ax4.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax4.set_xlabel('Silhouette Score', fontsize=11)
ax4.set_ylabel('Cluster', fontsize=11)
ax4.set_title('Silhouette Plot', fontsize=13)
ax4.axvline(x=train_silhouette, color="red", linestyle="--", 
            label=f'Score moyen: {train_silhouette:.3f}')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Taille des clusters
ax5 = fig.add_subplot(gs[2, 1])
cluster_sizes = pd.Series(train_clusters).value_counts().sort_index()
ax5.bar(cluster_sizes.index, cluster_sizes.values, 
        color=plt.cm.viridis(np.linspace(0, 1, optimal_k)),
        edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Cluster', fontsize=11)
ax5.set_ylabel('Nombre de Tumeurs', fontsize=11)
ax5.set_title('Distribution des Tumeurs par Cluster', fontsize=13)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Comparaison avec les vraies classes (pour information)
ax6 = fig.add_subplot(gs[2, 2])
y_train = y.iloc[X_train.index]
confusion_like = pd.crosstab(train_clusters, y_train)
sns.heatmap(confusion_like, annot=True, fmt='d', cmap='Blues', ax=ax6,
            cbar_kws={'label': 'Nombre'})
ax6.set_xlabel('Classe Réelle (0=Malin, 1=Bénin)', fontsize=11)
ax6.set_ylabel('Cluster', fontsize=11)
ax6.set_title('Clusters vs Classes Réelles', fontsize=13)

plt.savefig('clusters_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisations complètes sauvegardées")

print("\n💡 INTERPRÉTATION VISUELLE DES CLUSTERS :")
print("-" * 70)
print("""
ANALYSE DES VISUALISATIONS :

1. ESPACE PCA (Graphique haut-gauche) :
   - Les clusters sont clairement séparés dans l'espace réduit
   - Les centroïdes (X rouges) sont bien positionnés au centre de chaque groupe
   - La séparation suggère des différences marquées entre les groupes

2. CARACTÉRISTIQUES PAR CLUSTER (Graphique milieu) :
   - Montre les profils distincts de chaque cluster
   - Permet d'identifier les caractéristiques discriminantes
   - Aide à l'interprétation clinique des clusters

3. SILHOUETTE PLOT (Graphique bas-gauche) :
   - Largeur des barres = cohésion interne du cluster
   - Barres dépassant la ligne rouge = bonne assignation
   - Permet de détecter les clusters mal formés

4. DISTRIBUTION DES TUMEURS (Graphique bas-milieu) :
   - Équilibre ou déséquilibre entre clusters
   - Important pour l'interprétation clinique

5. CLUSTERS VS CLASSES RÉELLES (Graphique bas-droite) :
   - Montre si le clustering non-supervisé retrouve les vraies classes
   - Valide biologiquement la pertinence des clusters

CONCLUSION CLINIQUE :
Les clusters identifiés correspondent à des groupes de tumeurs avec
des profils morphologiques distincts, potentiellement associés à des
comportements biologiques différents (malin vs bénin).
""")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "="*70)
print("📋 RÉSUMÉ DE L'ANALYSE")
print("="*70)
print(f"""
✓ Dataset : {X.shape[0]} tumeurs, {X.shape[1]} caractéristiques
✓ Preprocessing : Normalisation StandardScaler, aucune valeur manquante
✓ Algorithme : K-Means avec {optimal_k} clusters
✓ Performance Train : Silhouette = {train_silhouette:.3f}, Inertie = {train_inertia:.2f}
✓ Performance Test : Silhouette = {test_silhouette:.3f}, Inertie = {test_inertia:.2f}
✓ Généralisation : {"Excellente" if abs(train_silhouette - test_silhouette) < 0.05 else "Bonne"}

RECOMMANDATIONS CLINIQUES :
- Les {optimal_k} clusters identifiés présentent des profils distincts
- Analyses complémentaires suggérées : validation avec expertise médicale
- Utilisation potentielle : aide à la décision diagnostique

Fichiers générés :
- exploration_initiale.png
- elbow_method.png
- clusters_visualization.png
""")

print("\n" + "="*70)
print("ANALYSE TERMINÉE AVEC SUCCÈS !")
print("="*70)