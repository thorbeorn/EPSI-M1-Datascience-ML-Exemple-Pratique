# 🤖 Exercices Machine Learning avec Scikit-Learn

Ce repository contient une série d'exercices pratiques pour apprendre le Machine Learning avec Python et Scikit-Learn.

## 📚 Table des matières

1. [Régression Linéaire](#1-régression-linéaire)
2. [Régression Logistique](#2-régression-logistique)
3. [Clustering K-Means](#3-clustering-k-means)
4. [Comparaison de Modèles](#4-comparaison-de-modèles)
5. [Analyse du Churn – Télécommunications](#5.-Analyse-du-Churn---Télécommunications)

---

## 🛠️ Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
# Créer un environnement virtuel
python3 -m venv venv

# Activer l'environnement virtuel
# Sur macOS/Linux :
source venv/bin/activate
# Sur Windows :
venv\Scripts\activate

# Installer les packages nécessaires
pip install -r requirements.txt
```

---

## 1. Régression Linéaire

### 🎯 Objectif
Prédire les prix des logements en Californie en utilisant la régression linéaire.

### 📊 Dataset
- **Source** : California Housing Dataset (Scikit-Learn)
- **Taille** : 20 640 échantillons
- **Features** : 8 (revenu médian, âge des maisons, nombre de pièces, etc.)
- **Target** : Prix des logements (en centaines de milliers de dollars)

### 📝 Code

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Coefficient R² :", r2)
print("MSE :", mse)
print("Coefficients :", model.coef_)
print("Intercept :", model.intercept_)
```

### 📈 Métriques attendues

- **R² Score** : ~0.60 (60% de variance expliquée)
- **MSE** : ~0.53
- **RMSE** : ~0.73 (≈ 73 000$ d'erreur moyenne)

### 🔍 Interprétation

- **R² = 0.60** : Le modèle explique 60% de la variabilité des prix
- **Coefficients positifs** : Revenu médian, nombre de chambres → augmentent le prix
- **Coefficients négatifs** : Latitude/Longitude → impact géographique

---

## 2. Régression Logistique

### 🎯 Objectif
Classifier les espèces de fleurs Iris (classification multi-classes).

### 📊 Dataset
- **Source** : Iris Dataset (Scikit-Learn)
- **Taille** : 150 échantillons
- **Features** : 4 (longueur/largeur des sépales et pétales)
- **Classes** : 3 (Setosa, Versicolor, Virginica)

### 📝 Code

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

X, y = load_iris().data, load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred, average='macro'))
print("Recall :", recall_score(y_test, y_pred, average='macro'))
print("F1-score :", f1_score(y_test, y_pred, average='macro'))
```

### 📈 Métriques attendues

- **Accuracy** : 1.00 (100%)
- **Precision** : 1.00
- **Recall** : 1.00
- **F1-Score** : 1.00

### 🔍 Pourquoi un score parfait ?

Le dataset Iris est **linéairement séparable** :
- Classes bien distinctes
- Features discriminantes
- Dataset simple et propre
- Idéal pour l'apprentissage !

---

## 3. Clustering K-Means

### 🎯 Objectif
Regrouper des données non étiquetées en clusters homogènes.

### 📊 Dataset
- **Source** : Données synthétiques (`make_blobs`)
- **Taille** : 300 échantillons
- **Features** : 2D (pour visualisation)
- **Clusters** : 7 centres

### 📝 Code

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=7, random_state=42)

# Méthode du coude pour trouver K optimal
inertias = []
for k in range(2, 12):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Entraîner avec K=7
kmeans = KMeans(n_clusters=7, random_state=42)
y_pred = kmeans.fit_predict(X)

# Évaluation
silhouette_avg = silhouette_score(X, y_pred)
print(f"Score de Silhouette : {silhouette_avg:.4f}")

# Visualisation
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='red', marker='X', label='Centres')
plt.legend()
plt.show()
```

### 📈 Métriques

| Score de Silhouette | Qualité |
|---------------------|---------|
| 0.7 - 1.0 | ✅ Excellent |
| 0.5 - 0.7 | ✅ Bon |
| 0.25 - 0.5 | ⚠️ Faible |
| < 0.25 | ❌ Mauvais |

### 🔍 Techniques utilisées

- **Méthode du Coude** : Trouver le K optimal via l'inertie
- **Score de Silhouette** : Mesurer la qualité des clusters
- **Visualisation** : Graphiques des clusters et centres

---

## 4. Comparaison de Modèles

### 🎯 Objectif
Comparer les performances de 3 modèles sur la reconnaissance de chiffres manuscrits.

### 📊 Dataset
- **Source** : Digits Dataset (Scikit-Learn)
- **Taille** : 1 797 images
- **Features** : 64 (images 8×8 pixels)
- **Classes** : 10 (chiffres 0-9)

### 📝 Code

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'SVM': SVC(kernel='rbf'),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{name}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred, average='macro'):.4f}")
```

### 📈 Résultats comparatifs

| Modèle | Accuracy | F1-Score | Vitesse | Avantages |
|--------|----------|----------|---------|-----------|
| **SVM** | ~0.99 | ~0.99 | Moyen | 🥇 Meilleure performance |
| **Random Forest** | ~0.97 | ~0.97 | Rapide | ⚡ Bon compromis |
| **Logistic Regression** | ~0.96 | ~0.96 | Très rapide | 🚀 Simple et efficace |

### 🔍 Métriques détaillées

- **Accuracy** : Pourcentage de prédictions correctes
- **Precision** : Parmi les prédictions positives, combien sont vraies ?
- **Recall** : Parmi les vrais positifs, combien ont été détectés ?
- **F1-Score** : Moyenne harmonique (precision + recall)

### 📊 Visualisations générées

1. Exemples d'images (chiffres manuscrits)
2. Distribution des classes
3. Comparaison des 4 métriques
4. Matrices de confusion (par modèle)
5. Temps d'exécution
6. Validation croisée (5-fold)
7. Classement par F1-Score

---

# 5. Analyse du Churn – Télécommunications

## 🎯 Objectif du Projet

Ce projet vise à réduire la perte de clients (churn) pour une entreprise de télécommunications grâce à :
- La prédiction des clients à risque
- La segmentation comportementale
- Des recommandations personnalisées

---

## 📁 Structure du Projet
```txt
telecom-churn-analysis/
│
├── telecom_churn.csv              # Dataset (à fournir)
├── client_lost_telecom.py         # Script principal
├── churn_analysis_results.png     # Visualisations générées
├── README.md                      # Ce fichier
└── requirements.txt               # Dépendances Python
```

---

## 🔧 Installation

    1. Cloner le projet
    git clone https://github.com/thorbeorn/EPSI-M1-Datascience-ML-Exemple-Pratique.git
    cd EPSI-M1-Datascience-ML-Exemple-Pratique

    2. Installer les dépendances
    pandas>=1.3.0  
    numpy>=1.21.0  
    matplotlib>=3.4.0  
    seaborn>=0.11.0  
    scikit-learn>=1.0.0  

---

## 📊 Structure des Données
```csv
Colonne	Description
customerID	Identifiant unique du client
tenure	Ancienneté
Contract	Type de contrat
InternetService	Type d'abonnement internet
MonthlyCharges	Coût mensuel
TotalCharges	Coût total
Churn	Client parti (Yes/No) – target
```

(+ toutes les autres colonnes du jeu IBM)	

---

## 🚀 Exécution
Lancer l’analyse complète :
```bash
python client_lost_telecom/client_lost_telecom.py
```

Le script génère :
- Rapport de classification
- Score AUC-ROC
- Top 10 features
- Résultats du clustering
- Recommandations business
- Fichier churn_analysis_results.png (6 graphiques)

## 🧠 Méthodes Employées
### 1️⃣ Régression Logistique (Supervisée)
- Interprétable
- Probabilités de churn
- Baseline robuste

- Évaluation :
    - Accuracy, Recall, F1
    - AUC-ROC
    - Matrice de confusion

### 2️⃣ K-Means (Non supervisé)
- Segmentation des comportements
- Identification des clusters à risque
- Support des stratégies de rétention

- Sorties :
    - Méthode du coude
    - Score de silhouette
    - Taux de churn par cluster

## 📈 Résultats Clés Attendus
    - Exemple console :
    📊 AUC-ROC: 0.85
    🔝 Top features: Contract, tenure, OnlineSecurity...
    ⚠️ Clients à haut risque: 342

### Graphiques générés :
- Courbe ROC
- Matrice de confusion
- Features importantes
- Courbe du coude
- Silhouette
- Taux de churn par cluster

## 💡 Recommandations
Actions immédiates
- Contacter les clients avec probabilité > 70%
- Proposer des offres d'engagement
- Améliorer les services critiques (TechSupport, OnlineSecurity)

Stratégies par cluster
- Cluster haut risque : actions rapides
- Cluster modéré : analyse satisfaction
- Cluster loyal : programme fidélité

---

## 📖 Ressources

### Documentation officielle
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/docs/)
- [NumPy](https://numpy.org/doc/)
- [Matplotlib](https://matplotlib.org/stable/contents.html)

### Tutoriels recommandés
- [Scikit-Learn Tutorial](https://scikit-learn.org/stable/tutorial/index.html)
- [Machine Learning Crash Course (Google)](https://developers.google.com/machine-learning/crash-course)
- [Kaggle Learn](https://www.kaggle.com/learn)

### Datasets pour pratiquer
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-Learn Datasets](https://scikit-learn.org/stable/datasets.html)

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à :
- 🐛 Reporter des bugs
- 💡 Proposer de nouvelles idées
- 📝 Améliorer la documentation
- ✨ Ajouter de nouveaux exercices

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## ✨ Auteur

Créé avec ❤️ pour apprendre le Machine Learning
