import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import time

digits = load_digits()
X, y = digits.data, digits.target

unique, counts = np.unique(y, return_counts=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'SVM (RBF kernel)': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}
results = []

for name, model in models.items():
    if name == 'Random Forest':
        X_tr, X_te = X_train, X_test
    else:
        X_tr, X_te = X_train_scaled, X_test_scaled
    
    start_time = time.time()
    model.fit(X_tr, y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    y_pred = model.predict(X_te)
    prediction_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')

    results.append({
        'Modèle': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Training Time': training_time,
        'Prediction Time': prediction_time,
        'y_pred': y_pred
    })

df_results = pd.DataFrame(results)
print("\n📊 Tableau récapitulatif :")
print(df_results[['Modèle', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 
                   'CV Mean', 'Training Time']].to_string(index=False))

best_model_idx = df_results['F1-Score'].idxmax()
best_model = df_results.loc[best_model_idx, 'Modèle']
print(f"\nMeilleur modèle (F1-Score) : {best_model}")


fig = plt.figure(figsize=(20, 12))
ax1 = plt.subplot(3, 4, 1)
for i in range(10):
    plt.subplot(3, 4, i+1) if i < 10 else None
    if i < len(digits.images):
        plt.imshow(digits.images[i], cmap='gray')
        plt.title(f'Chiffre: {digits.target[i]}', fontsize=10)
        plt.axis('off')
plt.subplot(3, 4, 11)
plt.bar(unique, counts, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Chiffre', fontsize=11)
plt.ylabel('Nombre d\'échantillons', fontsize=11)
plt.title('Distribution des classes', fontsize=12, fontweight='bold')
plt.xticks(unique)
plt.grid(True, alpha=0.3, axis='y')
plt.subplot(3, 4, 12)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results))
width = 0.2
for i, metric in enumerate(metrics):
    values = [r[metric] for r in results]
    plt.bar(x + i*width, values, width, label=metric, alpha=0.8)
plt.xlabel('Modèle', fontsize=11)
plt.ylabel('Score', fontsize=11)
plt.title('Comparaison des métriques', fontsize=12, fontweight='bold')
plt.xticks(x + width*1.5, [r['Modèle'].split()[0] for r in results], rotation=15)
plt.legend(loc='lower right', fontsize=9)
plt.ylim(0.9, 1.0)
plt.grid(True, alpha=0.3, axis='y')
for idx, result in enumerate(results):
    ax = plt.subplot(3, 3, idx+4)
    cm = confusion_matrix(y_test, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                square=True, linewidths=0.5, ax=ax)
    ax.set_title(f"{result['Modèle']}\nF1: {result['F1-Score']:.4f}", 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Prédiction', fontsize=10)
    ax.set_ylabel('Vérité', fontsize=10)
ax = plt.subplot(3, 3, 7)
model_names = [r['Modèle'].split()[0] for r in results]
train_times = [r['Training Time'] for r in results]
pred_times = [r['Prediction Time'] * 1000 for r in results]
x_pos = np.arange(len(model_names))
ax.bar(x_pos - 0.2, train_times, 0.4, label='Entraînement (s)', color='coral', alpha=0.8)
ax.bar(x_pos + 0.2, pred_times, 0.4, label='Prédiction (ms)', color='lightgreen', alpha=0.8)
ax.set_xlabel('Modèle', fontsize=11)
ax.set_ylabel('Temps', fontsize=11)
ax.set_title('Temps d\'exécution', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax = plt.subplot(3, 3, 8)
for idx, result in enumerate(results):
    model_name = result['Modèle'].split()[0]
    cv_mean = result['CV Mean']
    cv_std = result['CV Std']
    ax.errorbar(idx, cv_mean, yerr=cv_std, marker='o', markersize=10, 
                capsize=5, capthick=2, linewidth=2, label=model_name)
ax.set_xlabel('Modèle', fontsize=11)
ax.set_ylabel('Accuracy (CV)', fontsize=11)
ax.set_title('Validation Croisée (5-fold)', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(results)))
ax.set_xticklabels([r['Modèle'].split()[0] for r in results], rotation=15)
ax.set_ylim(0.9, 1.0)
ax.grid(True, alpha=0.3)
ax.legend()
ax = plt.subplot(3, 3, 9)
colors = ['#ff9999' if i != best_model_idx else '#66b266' for i in range(len(results))]
f1_scores = [r['F1-Score'] for r in results]
bars = ax.barh([r['Modèle'] for r in results], f1_scores, color=colors, 
               edgecolor='black', alpha=0.8)
ax.set_xlabel('F1-Score', fontsize=11)
ax.set_title('Classement par F1-Score', fontsize=12, fontweight='bold')
ax.set_xlim(0.9, 1.0)
ax.grid(True, alpha=0.3, axis='x')
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    ax.text(score - 0.002, bar.get_y() + bar.get_height()/2, 
            f'{score:.4f}', ha='right', va='center', fontweight='bold', color='white')
plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphiques sauvegardés : 'model_comparison_results.png'")
plt.show()

best_result = results[best_model_idx]
print(f"\nRecommandation : Utiliser {best_model} pour ce problème")
print(f"   → F1-Score : {best_result['F1-Score']:.4f}")
print(f"   → Accuracy : {best_result['Accuracy']:.4f}")
print(f"   → Temps d'entraînement : {best_result['Training Time']:.4f}s")