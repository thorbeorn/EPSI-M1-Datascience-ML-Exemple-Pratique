import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, silhouette_score)
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

df_clean = df.drop('customerID', axis=1)

cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
cat_cols.remove('Churn')  # Exclure la variable cible
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
le = LabelEncoder()
df_encoded = df_clean.copy()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

df_encoded['Churn'] = le.fit_transform(df_encoded['Churn'])

print("MODÈLE SUPERVISÉ - RÉGRESSION LOGISTIQUE")
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
cm = confusion_matrix(y_test, y_pred)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False).head(10)

print("MODÈLE NON SUPERVISÉ - K-MEANS CLUSTERING")
X_all_scaled = scaler.fit_transform(X)
inertias = []
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_all_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_all_scaled, kmeans_temp.labels_))
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_all_scaled)
df_encoded['Cluster'] = clusters
cluster_churn = df_encoded.groupby('Cluster')['Churn'].agg(['mean', 'count'])
cluster_churn.columns = ['Taux_Churn', 'Nb_Clients']
cluster_churn['Taux_Churn'] = cluster_churn['Taux_Churn'] * 100
cluster_profiles = df_encoded.groupby('Cluster')[num_cols].mean()


print("GÉNÉRATION DES VISUALISATIONS")
plt.figure(figsize=(16, 12))
plt.subplot(2, 3, 1)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC - Régression Logistique')
plt.legend()
plt.grid(True, alpha=0.3)
plt.subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion')
plt.ylabel('Vrai Label')
plt.xlabel('Prédiction')
plt.subplot(2, 3, 3)
top_features = feature_importance.head(8)
colors = ['red' if x < 0 else 'green' for x in top_features['Coefficient']]
plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
plt.xlabel('Coefficient')
plt.title('Top 8 Features Influençant le Churn')
plt.grid(True, alpha=0.3)
plt.subplot(2, 3, 4)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Nombre de Clusters')
plt.ylabel('Inertie')
plt.title('Méthode du Coude - K-Means')
plt.grid(True, alpha=0.3)
plt.subplot(2, 3, 5)
plt.plot(K_range, silhouette_scores, 'go-')
plt.xlabel('Nombre de Clusters')
plt.ylabel('Score de Silhouette')
plt.title('Score de Silhouette par K')
plt.grid(True, alpha=0.3)
plt.subplot(2, 3, 6)
plt.bar(cluster_churn.index, cluster_churn['Taux_Churn'], 
        color=['green' if x < 20 else 'orange' if x < 40 else 'red' 
               for x in cluster_churn['Taux_Churn']])
plt.xlabel('Cluster')
plt.ylabel('Taux de Churn (%)')
plt.title('Taux de Churn par Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('churn_analysis_results.png', dpi=300, bbox_inches='tight')

print("IDENTIFICATION DES CLIENTS À RISQUE")
df['Churn_Probability'] = log_reg.predict_proba(scaler.transform(X))[:, 1]
df['Cluster'] = clusters
high_risk = df[df['Churn_Probability'] > 0.7].copy()

print(f"\n🔴 Top 10 clients à surveiller:")
top_risk = df.nlargest(10, 'Churn_Probability')[
    ['customerID', 'Churn_Probability', 'Cluster', 'tenure', 
     'MonthlyCharges', 'Contract']
]
print(top_risk)

print("RECOMMANDATIONS STRATÉGIQUES DE RÉTENTION")
print("""
📌 STRATÉGIES PAR CLUSTER:
""")

for cluster_id in range(optimal_k):
    cluster_data = df_encoded[df_encoded['Cluster'] == cluster_id]
    churn_rate = cluster_data['Churn'].mean() * 100
    avg_tenure = cluster_data['tenure'].mean()
    avg_charges = cluster_data['MonthlyCharges'].mean()
    
    print(f"\n🎯 CLUSTER {cluster_id}:")
    print(f"   • Taux de churn: {churn_rate:.1f}%")
    print(f"   • Ancienneté moyenne: {avg_tenure:.1f} mois")
    print(f"   • Charges mensuelles: {avg_charges:.2f}€")
    
    if churn_rate > 40:
        print(f"   ⚠️  PRIORITÉ ÉLEVÉE - Actions immédiates requises")
    elif churn_rate > 20:
        print(f"   ⚡ PRIORITÉ MOYENNE - Surveillance active")
    else:
        print(f"   ✅ PRIORITÉ FAIBLE - Maintien de la satisfaction")

print(f"\n💡 ACTIONS RECOMMANDÉES:")
print("""
1. Contacter proactivement les clients avec probabilité > 70%
2. Proposer des offres personnalisées selon le cluster
3. Focus sur les contrats courts (month-to-month)
4. Améliorer les services techniques et sécurité en ligne
5. Programme de fidélité pour clients long-terme
""")