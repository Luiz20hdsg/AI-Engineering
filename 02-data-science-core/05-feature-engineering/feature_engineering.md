# 📊 Feature Engineering — A Arte de Criar Features

> "Feature Engineering é a arte de extrair mais informação dos seus dados. É frequentemente a diferença entre um modelo medíocre e um modelo excelente." — Kaggle Grand Masters

---

## 1. Encoding Categórico

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

df = pd.DataFrame({
    'cor': ['vermelho', 'azul', 'verde', 'azul', 'vermelho'],
    'tamanho': ['P', 'M', 'G', 'GG', 'M'],
    'cidade': ['SP', 'RJ', 'MG', 'SP', 'RJ'],
    'preco': [100, 200, 150, 250, 180]
})

# ---- One-Hot Encoding (variáveis nominais sem ordem) ----
# Use quando: categorias sem ordem natural e poucas categorias
df_onehot = pd.get_dummies(df, columns=['cor'], drop_first=True)  # drop_first evita multicolinearidade

# ---- Ordinal Encoding (variáveis com ordem) ----
# Use quando: categorias com ordem natural
ordem_tamanho = {'P': 1, 'M': 2, 'G': 3, 'GG': 4}
df['tamanho_ordinal'] = df['tamanho'].map(ordem_tamanho)

# ---- Label Encoding ----
# Use quando: árvores de decisão (não requer one-hot)
le = LabelEncoder()
df['cor_label'] = le.fit_transform(df['cor'])

# ---- Target Encoding (PODEROSO) ----
# Use quando: alta cardinalidade (muitas categorias únicas)
# Substitui categoria pela média do target daquela categoria
# ⚠️ Cuidado com data leakage! Use com cross-validation
te = TargetEncoder(cols=['cidade'])
df['cidade_target'] = te.fit_transform(df['cidade'], df['preco'])

# ---- Frequency Encoding ----
# Use quando: a frequência da categoria é informativa
freq = df['cidade'].value_counts(normalize=True)
df['cidade_freq'] = df['cidade'].map(freq)
```

## 2. Scaling e Normalização

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

# Dados de exemplo
X = np.array([[1, 1000, 0.1],
              [2, 2000, 0.2],
              [3, 3000, 0.3],
              [100, 50000, 0.9]])  # outlier

# ---- StandardScaler (Z-score) ----
# Use quando: dados ~normais, sem outliers extremos
# Formula: z = (x - μ) / σ
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)
# Resultado: média=0, desvio=1

# ---- MinMaxScaler ----
# Use quando: quer valores entre [0, 1], dados sem outliers
# Formula: x' = (x - min) / (max - min)
scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)

# ---- RobustScaler (MELHOR para outliers) ----
# Use quando: há outliers nos dados
# Formula: x' = (x - mediana) / IQR
scaler = RobustScaler()
X_robust = scaler.fit_transform(X)

# ---- Log Transform ----
# Use quando: distribuição muito assimétrica (skewed)
X_log = np.log1p(X)  # log(1+x) para lidar com zeros

# ---- Box-Cox / Yeo-Johnson ----
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')  # funciona com negativos
X_power = pt.fit_transform(X)
```

### Quando usar qual Scaler?

| Scaler | Quando Usar | Sensível a Outliers? |
|--------|------------|---------------------|
| StandardScaler | Dados ~normais, SVM, Logistic Regression | ✅ Sim |
| MinMaxScaler | Neural Networks, dados uniformes | ✅ Sim |
| RobustScaler | Dados com outliers | ❌ Não |
| Log Transform | Distribuições muito assimétricas | Parcial |
| PowerTransformer | Tornar dados mais gaussianos | Parcial |

## 3. Criação de Features

```python
import pandas as pd
import numpy as np

# ---- Features Temporais ----
df['data'] = pd.to_datetime(df['data'])
df['ano'] = df['data'].dt.year
df['mes'] = df['data'].dt.month
df['dia_semana'] = df['data'].dt.dayofweek
df['eh_fim_de_semana'] = df['data'].dt.dayofweek.isin([5, 6]).astype(int)
df['trimestre'] = df['data'].dt.quarter
df['dia_do_ano'] = df['data'].dt.dayofyear
df['hora'] = df['data'].dt.hour

# Componentes cíclicos (para modelos que não entendem ciclicidade)
df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)

# ---- Features de Interação ----
df['area'] = df['comprimento'] * df['largura']
df['ratio_preco_area'] = df['preco'] / df['area']
df['bmi'] = df['peso'] / (df['altura'] ** 2)

# ---- Features Polinomiais ----
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# ---- Features de Agregação (por grupo) ----
df['salario_medio_dept'] = df.groupby('departamento')['salario'].transform('mean')
df['salario_std_dept'] = df.groupby('departamento')['salario'].transform('std')
df['salario_rank_dept'] = df.groupby('departamento')['salario'].rank(pct=True)
df['salario_vs_media_dept'] = df['salario'] / df['salario_medio_dept']

# ---- Binning ----
df['idade_bin'] = pd.cut(df['idade'], bins=[0, 18, 30, 50, 100], 
                          labels=['jovem', 'adulto_jovem', 'adulto', 'senior'])
df['salario_quantile'] = pd.qcut(df['salario'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# ---- Text Features ----
df['nome_length'] = df['nome'].str.len()
df['num_palavras'] = df['descricao'].str.split().str.len()
df['tem_email'] = df['texto'].str.contains('@').astype(int)
```

## 4. Feature Selection

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# ---- Filter Methods (estatísticos) ----

# Correlação (remover features muito correlacionadas)
def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop

# Variância (remover features de baixa variância)
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# SelectKBest
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
feature_scores = pd.DataFrame({
    'feature': feature_names,
    'score': selector.scores_
}).sort_values('score', ascending=False)

# ---- Wrapper Methods ----

# Recursive Feature Elimination
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model, n_features_to_select=10, step=1)
X_selected = rfe.fit_transform(X, y)
selected_features = [f for f, s in zip(feature_names, rfe.support_) if s]

# ---- Embedded Methods ----

# Feature importance de modelos
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# L1 Regularization (Lasso) — zera features irrelevantes
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected = [f for f, c in zip(feature_names, lasso.coef_) if c != 0]
```

## 5. Lidando com Dados Desbalanceados

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Ver distribuição
print(y.value_counts())
# 0    9500 (95%)
# 1     500 (5%)  ← classe minoritária

# ---- Oversampling ----
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ---- Undersampling ----
under = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = under.fit_resample(X_train, y_train)

# ---- Combinação (melhor abordagem) ----
pipeline = ImbPipeline([
    ('over', SMOTE(sampling_strategy=0.5)),
    ('under', RandomUnderSampler(sampling_strategy=0.8)),
])
X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

# ---- Class Weights (alternativa sem reamostrar) ----
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced', random_state=42)
```

---

## 🏋️ Exercícios

1. Pegue um dataset do Kaggle e crie pelo menos 20 features novas
2. Compare a performance de um modelo com e sem feature engineering
3. Implemente target encoding com cross-validation para evitar leakage
4. Crie um pipeline completo de feature engineering reutilizável
