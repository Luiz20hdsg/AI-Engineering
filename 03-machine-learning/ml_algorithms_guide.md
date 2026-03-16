# 🤖 Machine Learning — Guia Completo de Algoritmos

---

## 🗺️ Mapa Mental: Qual Algoritmo Usar?

```
Qual é o seu problema?
│
├── Tenho labels (Y) → SUPERVISED LEARNING
│   ├── Y é contínuo → REGRESSÃO
│   │   ├── Relação linear? → Linear/Ridge/Lasso Regression
│   │   ├── Relação não-linear? → Decision Tree / Random Forest / XGBoost
│   │   └── Muitas features? → Lasso (L1) / ElasticNet
│   │
│   └── Y é categórico → CLASSIFICAÇÃO
│       ├── Binário simples? → Logistic Regression
│       ├── Dados tabulares? → XGBoost / LightGBM (MELHOR para tabular)
│       ├── Fronteira não-linear? → SVM / Random Forest
│       ├── Baseline rápido? → Naive Bayes / KNN
│       └── Interpretabilidade? → Decision Tree / Logistic Regression
│
├── NÃO tenho labels → UNSUPERVISED LEARNING
│   ├── Encontrar grupos? → CLUSTERING
│   │   ├── Sei quantos grupos? → K-Means
│   │   ├── Clusters de forma arbitrária? → DBSCAN
│   │   └── Hierarquia de clusters? → Hierarchical Clustering
│   │
│   ├── Reduzir dimensões? → DIMENSIONALITY REDUCTION
│   │   ├── Linear? → PCA
│   │   └── Visualização? → t-SNE / UMAP
│   │
│   └── Detectar anomalias? → ANOMALY DETECTION
│       ├── Isolation Forest
│       └── Local Outlier Factor (LOF)
│
└── Tenho poucos labels → SEMI-SUPERVISED LEARNING
```

---

## 1. REGRESSÃO

### 1.1 Linear Regression

$$\hat{y} = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n = \mathbf{w}^T\mathbf{x} + b$$

Objetivo: Minimizar MSE = $\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression (sem regularização)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Ridge (L2 regularization) — penaliza coeficientes grandes
# Loss = MSE + α * Σwᵢ²
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1 regularization) — pode zerar coeficientes (feature selection!)
# Loss = MSE + α * Σ|wᵢ|
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
# Ver quais features foram "zeradas"
print(f"Features eliminadas: {sum(lasso.coef_ == 0)}/{len(lasso.coef_)}")

# ElasticNet (L1 + L2)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```

### 1.2 Regularização — Entendendo L1, L2, ElasticNet

| Tipo | Fórmula da Loss | Efeito | Quando Usar |
|------|----------------|--------|-------------|
| **Nenhuma** | MSE | Overfitting se muitas features | Poucas features, dados limpos |
| **Ridge (L2)** | MSE + α∑w² | Reduz coeficientes | Multicolinearidade |
| **Lasso (L1)** | MSE + α∑\|w\| | Zera coeficientes (sparse) | Feature selection automática |
| **ElasticNet** | MSE + α₁∑\|w\| + α₂∑w² | Combina ambos | Muitas features correlacionadas |

---

## 2. CLASSIFICAÇÃO

### 2.1 Logistic Regression

> Apesar do nome, é um classificador! Usa a função sigmoid para mapear para probabilidades.

$$P(y=1|x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}$$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# Treinar
model = LogisticRegression(
    C=1.0,               # inverso da regularização (menor = mais regularização)
    penalty='l2',         # tipo de regularização
    class_weight='balanced',  # ajusta para dados desbalanceados
    max_iter=1000
)
model.fit(X_train, y_train)

# Predição
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probabilidades

# Métricas completas
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
```

### 2.2 Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Treinar
tree = DecisionTreeClassifier(
    max_depth=5,          # profundidade máxima (evita overfitting)
    min_samples_split=10, # mínimo de amostras para dividir
    min_samples_leaf=5,   # mínimo de amostras em folha
    random_state=42
)
tree.fit(X_train, y_train)

# Visualizar a árvore
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=feature_names, class_names=['No', 'Yes'], 
          filled=True, rounded=True, max_depth=3)
plt.savefig('tree.png', dpi=150, bbox_inches='tight')

# Ver regras em texto
print(export_text(tree, feature_names=feature_names))

# Feature importance
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)
```

### 2.3 XGBoost / LightGBM — Os Reis do Tabular

> "Se seus dados são tabulares, provavelmente XGBoost/LightGBM é o melhor modelo." — Todo Kaggle Grand Master

```python
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# ---- XGBoost ----
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,        # L1
    reg_lambda=1.0,        # L2
    scale_pos_weight=1,    # para dados desbalanceados
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

# Treinar com early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ---- LightGBM (mais rápido, melhor com categorias) ----
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=-1,          # sem limite (usa num_leaves)
    num_leaves=31,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)

# Cross-validation
cv_scores = cross_val_score(lgb_model, X, y, cv=5, scoring='roc_auc')
print(f"AUC-ROC CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### 2.4 SVM (Support Vector Machine)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SVM PRECISA de scaling!
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',      # 'linear', 'poly', 'rbf', 'sigmoid'
        C=1.0,             # regularização
        gamma='scale',     # kernel coefficient
        probability=True,  # para predict_proba
        random_state=42
    ))
])
svm_pipeline.fit(X_train, y_train)
```

---

## 3. UNSUPERVISED LEARNING

### 3.1 K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Método do Cotovelo para escolher K
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, kmeans.labels_))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, 'bx-')
ax1.set_title('Método do Cotovelo')
ax1.set_xlabel('K')
ax1.set_ylabel('Inertia')

ax2.plot(K_range, silhouettes, 'rx-')
ax2.set_title('Silhouette Score')
ax2.set_xlabel('K')
ax2.set_ylabel('Score')
plt.savefig('elbow_method.png')

# Treinar com K escolhido
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)
```

### 3.2 PCA — Redução de Dimensionalidade

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA
pca = PCA(n_components=0.95)  # manter 95% da variância
X_pca = pca.fit_transform(X)

print(f"Dimensões originais: {X.shape[1]}")
print(f"Dimensões após PCA: {X_pca.shape[1]}")
print(f"Variância explicada: {pca.explained_variance_ratio_.sum():.2%}")

# Plot da variância explicada acumulada
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variância')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Explicada Acumulada')
plt.legend()
plt.title('PCA — Scree Plot')
plt.savefig('pca_scree.png')
```

---

## 4. MODEL EVALUATION

### 4.1 Métricas de Classificação

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# [[TN, FP],
#  [FN, TP]]

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Classe 0', 'Classe 1']))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')

# Precision-Recall Curve (MELHOR para dados desbalanceados)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'b-', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('pr_curve.png')
```

### 4.2 Quando usar qual métrica?

| Métrica | Quando Usar | Exemplo |
|---------|-------------|---------|
| **Accuracy** | Classes balanceadas | Classificação de imagens |
| **Precision** | Custo alto de FP | Detecção de spam |
| **Recall** | Custo alto de FN | Diagnóstico médico |
| **F1** | Balanço entre Precision e Recall | Geral |
| **AUC-ROC** | Ranking de probabilidades | Scores de crédito |
| **Log Loss** | Calibração de probabilidades | Probabilidades de evento |

### 4.3 Cross-Validation e Hyperparameter Tuning

```python
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, 
    GridSearchCV, RandomizedSearchCV
)
from scipy.stats import uniform, randint
import optuna  # Bayesian optimization

# ---- Cross-Validation ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"AUC-ROC: {scores.mean():.4f} ± {scores.std():.4f}")

# ---- Grid Search ----
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9]
}
grid_search = GridSearchCV(
    xgb_model, param_grid, cv=5, scoring='roc_auc', 
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
print(f"Melhores params: {grid_search.best_params_}")
print(f"Melhor score: {grid_search.best_score_:.4f}")

# ---- Optuna (Bayesian — MAIS EFICIENTE) ----
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return cv_scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Melhores params: {study.best_params}")
print(f"Melhor AUC: {study.best_value:.4f}")
```

---

## 5. PIPELINE COMPLETO

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Definir colunas
numeric_features = ['idade', 'salario', 'experiencia']
categorical_features = ['departamento', 'cidade', 'cargo']

# Preprocessamento por tipo
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Pipeline completo
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    ))
])

# Treinar
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Salvar pipeline
import joblib
joblib.dump(pipeline, 'model_pipeline.pkl')

# Carregar
pipeline_loaded = joblib.load('model_pipeline.pkl')
```

---

## 🏋️ Exercícios

1. **Projeto Regressão:** Prever preço de imóveis (use Kaggle House Prices)
2. **Projeto Classificação:** Detectar fraude em transações (use Kaggle Credit Card Fraud)
3. **Projeto Clustering:** Segmentação de clientes (use Mall Customers dataset)
4. **Compare** pelo menos 5 modelos diferentes no mesmo dataset
5. **Implemente** hyperparameter tuning com Optuna e compare com GridSearch
6. **Crie** um pipeline completo e reutilizável

---

## 📝 Notas

> Adicione aqui suas anotações pessoais conforme avança nos estudos.
