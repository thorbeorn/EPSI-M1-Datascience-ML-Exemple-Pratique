# 📊 Analyse du Churn - Entreprise de Télécommunications

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)

## 🎯 Objectif du Projet

Ce projet vise à **réduire la perte de clients (churn)** d'une entreprise de télécommunications en :
- ✅ Identifiant les clients à risque de départ
- ✅ Segmentant les clients selon leurs comportements
- ✅ Proposant des stratégies de rétention ciblées et personnalisées

---

## 📁 Structure du Projet

```
telecom-churn-analysis/
│
├── telecom_churn.csv              # Dataset (à fournir)
├── churn_analysis.py              # Script principal
├── churn_analysis_results.png    # Visualisations générées
├── README.md                      # Ce fichier
└── requirements.txt               # Dépendances Python
```

---

## 🔧 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)

### Étape 1 : Cloner le projet
```bash
git clone https://github.com/votre-repo/telecom-churn-analysis.git
cd telecom-churn-analysis
```

### Étape 2 : Installer les dépendances
```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt` :**
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

---

## 📊 Structure des Données

Le fichier CSV doit contenir les colonnes suivantes :

| Colonne | Type | Description |
|---------|------|-------------|
| `customerID` | string | Identifiant unique du client |
| `gender` | string | Genre (Male/Female) |
| `SeniorCitizen` | int | Senior (0/1) |
| `Partner` | string | A un partenaire (Yes/No) |
| `Dependents` | string | A des personnes à charge (Yes/No) |
| `tenure` | int | Ancienneté en mois |
| `PhoneService` | string | Service téléphonique (Yes/No) |
| `MultipleLines` | string | Lignes multiples (Yes/No/No phone service) |
| `InternetService` | string | Type d'internet (DSL/Fiber optic/No) |
| `OnlineSecurity` | string | Sécurité en ligne (Yes/No/No internet service) |
| `OnlineBackup` | string | Sauvegarde en ligne (Yes/No/No internet service) |
| `DeviceProtection` | string | Protection des appareils (Yes/No/No internet service) |
| `TechSupport` | string | Support technique (Yes/No/No internet service) |
| `StreamingTV` | string | Streaming TV (Yes/No/No internet service) |
| `StreamingMovies` | string | Streaming films (Yes/No/No internet service) |
| `Contract` | string | Type de contrat (Month-to-month/One year/Two year) |
| `PaperlessBilling` | string | Facturation sans papier (Yes/No) |
| `PaymentMethod` | string | Méthode de paiement |
| `MonthlyCharges` | float | Charges mensuelles |
| `TotalCharges` | float | Charges totales |
| `Churn` | string | Client parti (Yes/No) - **Variable cible** |

---

## 🚀 Utilisation

### Exécuter l'analyse complète
```bash
python churn_analysis.py
```

### Résultat attendu
Le script va :
1. ✅ Charger et nettoyer les données
2. ✅ Entraîner un modèle de **Régression Logistique**
3. ✅ Effectuer un **Clustering K-Means**
4. ✅ Générer un fichier `churn_analysis_results.png` avec 6 graphiques
5. ✅ Afficher dans la console :
   - Rapport de classification
   - Score AUC-ROC
   - Top 10 features importantes
   - Profil de chaque cluster
   - Liste des clients à haut risque
   - Recommandations stratégiques

---

## 🧠 Méthodologie

### 1️⃣ **Approche Supervisée : Régression Logistique**

**Pourquoi ce modèle ?**
- ✅ Variable cible disponible (`Churn` : Yes/No)
- ✅ **Interprétabilité** : les coefficients montrent l'impact de chaque variable
- ✅ Fournit des **probabilités de churn** (0% à 100%)
- ✅ Efficace pour la classification binaire
- ✅ Baseline robuste avant d'explorer des modèles complexes (XGBoost, Random Forest...)

**Processus :**
```
Données → Encodage → Normalisation → Train/Test Split (80/20) 
→ Entraînement → Prédictions → Évaluation
```

**Métriques d'évaluation :**
- **Précision, Rappel, F1-Score** : performance globale
- **AUC-ROC** : capacité à discriminer les churners
- **Matrice de confusion** : faux positifs vs vrais positifs

---

### 2️⃣ **Approche Non Supervisée : K-Means Clustering**

**Pourquoi cette approche ?**
- 🎯 Complète l'approche supervisée
- 🎯 Segmente les clients en **groupes comportementaux**
- 🎯 Permet des stratégies de rétention **personnalisées par segment**

**Processus :**
```
Données normalisées → Méthode du coude → Choix de K=4 clusters 
→ Entraînement K-Means → Analyse des profils
```

**Utilité :**
- Cluster 1 : Clients loyaux (faible risque)
- Cluster 2 : Clients à risque modéré
- Cluster 3 : Clients premium (haute valeur)
- Cluster 4 : Clients à haut risque (action immédiate)

---

## 📈 Résultats Attendus

### **Sorties Console**
```
📊 Score AUC-ROC: 0.85
🔝 Top features: Contract, tenure, InternetService, OnlineSecurity...
⚠️  Clients à HAUT RISQUE: 342
🎯 Taux de churn par cluster:
   Cluster 0: 15.2%
   Cluster 1: 42.8% ← Action prioritaire
   Cluster 2: 8.1%
   Cluster 3: 31.4%
```

### **Visualisations Générées**
Le fichier `churn_analysis_results.png` contient 6 graphiques :
1. **Courbe ROC** - Performance du modèle
2. **Matrice de confusion** - Prédictions vs réalité
3. **Top 8 features** - Variables les plus influentes
4. **Méthode du coude** - Choix du nombre de clusters
5. **Score de silhouette** - Qualité du clustering
6. **Taux de churn par cluster** - Segmentation des risques

---

## 💡 Recommandations Stratégiques

### **Actions Immédiates**
1. 🚨 **Contacter les clients avec probabilité > 70%**
   - Appel personnalisé
   - Offre de rétention exclusive

2. 📞 **Programme de fidélité pour contrats courts**
   - Inciter à passer en contrat 1 ou 2 ans
   - Réduction sur engagement long-terme

3. 🛡️ **Améliorer les services critiques**
   - OnlineSecurity
   - TechSupport
   - OnlineBackup

### **Stratégies par Cluster**
```
🎯 CLUSTER À HAUT RISQUE (> 40% churn):
   → Offres agressives de rétention
   → Support proactif
   → Programme VIP

⚡ CLUSTER À RISQUE MODÉRÉ (20-40% churn):
   → Enquêtes de satisfaction
   → Upgrade de services
   → Avantages fidélité

✅ CLUSTER LOYAL (< 20% churn):
   → Maintien de la qualité
   → Récompenses fidélité
   → Programme de parrainage
```

---

## 🔍 Détails Techniques

### **Prétraitement des Données**
- Gestion des valeurs manquantes dans `TotalCharges`
- Encodage des variables catégorielles (Label Encoding)
- Normalisation avec `StandardScaler`
- Équilibrage des classes avec `class_weight='balanced'`

### **Hyperparamètres**
```python
# Régression Logistique
LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

# K-Means
KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)
```

### **Split des Données**
- **80%** entraînement
- **20%** test
- Stratification pour préserver la distribution du churn

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :
1. Forkez le projet
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Committez vos changements (`git commit -m 'Ajout fonctionnalité X'`)
4. Poussez vers la branche (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request

---

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 🙏 Remerciements

- Dataset inspiré de [IBM Watson Analytics](https://www.ibm.com/watson-analytics)
- scikit-learn pour les outils de Machine Learning
- Communauté Data Science

---

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !**