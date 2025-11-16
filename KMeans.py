import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

X, _ = make_blobs(n_samples=300, centers=7, random_state=42)

# =====Elbow Method=====
inertias = []
silhouette_scores = []
K_range = range(2, 12)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)


silhouette_avg = silhouette_score(X, y_pred)
print(f"\nScore de Silhouette moyen : {silhouette_avg:.4f}")
sample_silhouette_values = silhouette_samples(X, y_pred)
print("\nScore de Silhouette par cluster :")
for i in range(7):
    cluster_silhouette_values = sample_silhouette_values[y_pred == i]
    print(f"  Cluster {i} : {cluster_silhouette_values.mean():.4f} "
          f"(taille: {len(cluster_silhouette_values)})")

print(f"\nInterprétation du score de Silhouette :")
if silhouette_avg > 0.7:
    print("Excellent clustering (> 0.7)")
elif silhouette_avg > 0.5:
    print("Bon clustering (0.5 - 0.7)")
elif silhouette_avg > 0.25:
    print("Clustering faible (0.25 - 0.5)")
else:
    print("Mauvais clustering (< 0.25)")



fig = plt.figure(figsize=(18, 12))
ax1 = plt.subplot(2, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50, edgecolors='k')
plt.title('Données brutes (sans clustering)', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)
ax2 = plt.subplot(2, 3, 2)
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=7, color='r', linestyle='--', label='K=7 (optimal)')
plt.xlabel('Nombre de clusters (K)', fontsize=12)
plt.ylabel('Inertie', fontsize=12)
plt.title('Méthode du Coude (Elbow Method)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
ax3 = plt.subplot(2, 3, 3)
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.axvline(x=7, color='r', linestyle='--', label='K=7 (optimal)')
plt.axhline(y=silhouette_avg, color='orange', linestyle=':', label=f'Score K=7: {silhouette_avg:.3f}')
plt.xlabel('Nombre de clusters (K)', fontsize=12)
plt.ylabel('Score de Silhouette', fontsize=12)
plt.title('Score de Silhouette par K', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
ax4 = plt.subplot(2, 3, 4)
colors = cm.nipy_spectral(np.linspace(0, 1, 7))
for i in range(7):
    cluster_data = X[y_pred == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                s=50, c=[colors[i]], label=f'Cluster {i}',
                alpha=0.6, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', edgecolors='black', linewidths=2,
            label='Centres', zorder=10)
plt.title('Clusters K-Means (K=7)', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
ax5 = plt.subplot(2, 3, 5)
y_lower = 10
for i in range(7):
    cluster_silhouette_values = sample_silhouette_values[y_pred == i]
    cluster_silhouette_values.sort()
    size_cluster_i = cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / 7)
    ax5.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax5.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
ax5.set_title('Diagramme de Silhouette', fontsize=14, fontweight='bold')
ax5.set_xlabel('Coefficient de Silhouette')
ax5.set_ylabel('Cluster')
ax5.axvline(x=silhouette_avg, color="red", linestyle="--", 
            label=f'Moyenne: {silhouette_avg:.3f}')
ax5.set_yticks([])
ax5.legend()
ax5.grid(True, alpha=0.3, axis='x')
ax6 = plt.subplot(2, 3, 6)
cluster_sizes = [np.sum(y_pred == i) for i in range(7)]
bars = plt.bar(range(7), cluster_sizes, color=colors, edgecolor='black', alpha=0.7)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Nombre de points', fontsize=12)
plt.title('Taille des clusters', fontsize=14, fontweight='bold')
plt.xticks(range(7))
plt.grid(True, alpha=0.3, axis='y')
for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{size}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()