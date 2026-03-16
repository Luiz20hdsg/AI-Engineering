# 📐 Matemática para Machine Learning

## 1. Cálculo Diferencial

### 1.1 Derivadas — A Base do Gradient Descent

> **Por que isso importa?** O treinamento de modelos de ML é essencialmente um problema de **otimização**. 
> Derivadas nos dizem a **direção** e **taxa de variação** de uma função — exatamente o que precisamos para minimizar a loss function.

#### Conceito Fundamental

A derivada $f'(x)$ mede a taxa de variação instantânea de $f(x)$:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

#### Regras Essenciais

| Regra | Fórmula | Exemplo |
|-------|---------|---------|
| Constante | $\frac{d}{dx}c = 0$ | $\frac{d}{dx}5 = 0$ |
| Potência | $\frac{d}{dx}x^n = nx^{n-1}$ | $\frac{d}{dx}x^3 = 3x^2$ |
| Soma | $(f+g)' = f' + g'$ | $(x^2 + 3x)' = 2x + 3$ |
| Produto | $(fg)' = f'g + fg'$ | |
| Quociente | $(\frac{f}{g})' = \frac{f'g - fg'}{g^2}$ | |
| Cadeia | $(f(g(x)))' = f'(g(x)) \cdot g'(x)$ | |

#### Derivadas Importantes em ML

```
Sigmoid:  σ(x) = 1/(1+e^(-x))     →  σ'(x) = σ(x)(1 - σ(x))
ReLU:     f(x) = max(0, x)          →  f'(x) = 0 se x<0, 1 se x>0
Tanh:     tanh(x)                    →  tanh'(x) = 1 - tanh²(x)
Log:      ln(x)                      →  1/x
Exp:      e^x                        →  e^x
MSE:      L = (y - ŷ)²              →  dL/dŷ = -2(y - ŷ)
```

### 1.2 Derivadas Parciais e Gradiente

Quando temos funções de múltiplas variáveis (como loss functions com múltiplos parâmetros):

$$f(x, y) = x^2 + 3xy + y^2$$

$$\frac{\partial f}{\partial x} = 2x + 3y \quad \quad \frac{\partial f}{\partial y} = 3x + 2y$$

O **Gradiente** é o vetor de todas as derivadas parciais:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}$$

> **Intuição:** O gradiente aponta na direção de **maior crescimento** da função. Para **minimizar** (o que queremos em ML), vamos na direção **oposta** ao gradiente.

### 1.3 Gradient Descent

O algoritmo mais importante em ML:

$$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$$

Onde:
- $\theta$ = parâmetros do modelo
- $\alpha$ = learning rate (taxa de aprendizado)
- $\nabla L$ = gradiente da loss function

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """Gradient Descent para Regressão Linear"""
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = X @ weights + bias
        
        # Calcular loss (MSE)
        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)
        
        # Calcular gradientes
        dw = -(2/m) * X.T @ (y - y_pred)  # ∂L/∂w
        db = -(2/m) * np.sum(y - y_pred)   # ∂L/∂b
        
        # Atualizar parâmetros
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return weights, bias, losses

# Exemplo
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

weights, bias, losses = gradient_descent(X, y)
print(f"Pesos aprendidos: {weights}")  # ≈ [2, 3, -1]
```

#### Variantes do Gradient Descent

| Variante | Batch Size | Prós | Contras |
|----------|-----------|------|---------|
| **Batch GD** | Todo dataset | Convergência estável | Lento, muita memória |
| **Stochastic GD** | 1 amostra | Rápido, escapa mínimos locais | Ruidoso |
| **Mini-Batch GD** | N amostras | Melhor dos dois mundos | Precisa tunar batch size |

---

## 2. Álgebra Linear

### 2.1 Vetores

```python
import numpy as np

# Vetores
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

# Operações básicas
soma = v + w                    # [5, 7, 9]
escalar = 3 * v                 # [3, 6, 9]
dot_product = np.dot(v, w)      # 1*4 + 2*5 + 3*6 = 32
norma = np.linalg.norm(v)       # √(1² + 2² + 3²) = √14
unitario = v / norma            # vetor unitário

# Distâncias
euclidiana = np.linalg.norm(v - w)
cosseno_sim = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
```

> **Aplicação em ML:** Embeddings são vetores! Similaridade de cosseno é usada em NLP, sistemas de recomendação, e busca semântica (RAG).

### 2.2 Matrizes

```python
# Matrizes
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Operações
produto = A @ B                  # multiplicação de matrizes
transposta = A.T                 # transposta
inversa = np.linalg.inv(A)      # inversa
determinante = np.linalg.det(A) # determinante
trace = np.trace(A)             # soma da diagonal

# Identidade
I = np.eye(3)  # matriz identidade 3x3
```

#### Multiplicação de Matrizes — Regra

Para $C = AB$, onde $A$ é $(m \times n)$ e $B$ é $(n \times p)$:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

> **Em ML:** Um forward pass de neural network é basicamente: $\hat{y} = \sigma(Wx + b)$ — multiplicação de matrizes!

### 2.3 Autovalores e Autovetores

$$Av = \lambda v$$

Onde $v$ é o autovetor e $\lambda$ é o autovalor.

```python
# Autovalores e Autovetores
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Autovalores: {eigenvalues}")      # [5, 2]
print(f"Autovetores:\n{eigenvectors}")
```

> **Aplicação:** PCA (Principal Component Analysis) usa autovetores para encontrar as direções de maior variância nos dados.

### 2.4 SVD — Singular Value Decomposition

$$A = U \Sigma V^T$$

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U, S, Vt = np.linalg.svd(A)

# Redução de dimensionalidade com SVD
k = 2  # manter apenas 2 componentes
A_reduced = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"Forma original: {A.shape}")
print(f"Erro de reconstrução: {np.linalg.norm(A - A_reduced):.4f}")
```

> **Aplicações:** Compressão de imagens, sistemas de recomendação (Netflix Prize), redução de dimensionalidade.

---

## 3. Estatística e Probabilidade

### 3.1 Estatística Descritiva

```python
import numpy as np
from scipy import stats

dados = np.random.normal(loc=100, scale=15, size=1000)

# Medidas de tendência central
media = np.mean(dados)
mediana = np.median(dados)
moda = stats.mode(dados.round(), keepdims=True)

# Medidas de dispersão
variancia = np.var(dados)
desvio_padrao = np.std(dados)
iqr = np.percentile(dados, 75) - np.percentile(dados, 25)  # Interquartile Range

# Medidas de forma
assimetria = stats.skew(dados)    # skewness
curtose = stats.kurtosis(dados)    # kurtosis

print(f"""
Estatísticas Descritivas:
  Média: {media:.2f}
  Mediana: {mediana:.2f}
  Desvio Padrão: {desvio_padrao:.2f}
  IQR: {iqr:.2f}
  Assimetria: {assimetria:.4f}
  Curtose: {curtose:.4f}
""")
```

### 3.2 Distribuições de Probabilidade

| Distribuição | Uso em ML | Parâmetros |
|-------------|-----------|------------|
| **Normal** | Pesos iniciais, erros | μ (média), σ (desvio) |
| **Bernoulli** | Classificação binária | p (probabilidade) |
| **Binomial** | N experimentos binários | n, p |
| **Poisson** | Contagem de eventos | λ (taxa) |
| **Uniforme** | Inicialização de pesos | a, b (limites) |
| **Exponencial** | Tempo entre eventos | λ (taxa) |

```python
from scipy import stats
import numpy as np

# Distribuição Normal
normal = stats.norm(loc=0, scale=1)  # μ=0, σ=1
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")       # ≈ 0.975
print(f"P(-1.96 < X < 1.96) = {normal.cdf(1.96) - normal.cdf(-1.96):.4f}")  # ≈ 0.95

# Amostrar da distribuição
amostras = normal.rvs(size=1000)
```

### 3.3 Teorema de Bayes

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

```python
# Exemplo: Teste diagnóstico
# P(Doença) = 1% (prevalência)
# P(Teste+ | Doença) = 99% (sensibilidade)
# P(Teste+ | Saudável) = 5% (falso positivo)

p_doenca = 0.01
p_teste_pos_dado_doenca = 0.99
p_teste_pos_dado_saudavel = 0.05

# P(Teste+) = P(T+|D)*P(D) + P(T+|S)*P(S)
p_teste_pos = (p_teste_pos_dado_doenca * p_doenca + 
               p_teste_pos_dado_saudavel * (1 - p_doenca))

# P(Doença | Teste+)
p_doenca_dado_teste_pos = (p_teste_pos_dado_doenca * p_doenca) / p_teste_pos

print(f"P(Doença | Teste+) = {p_doenca_dado_teste_pos:.2%}")
# ≈ 16.7% — mesmo com teste 99% sensível!
```

> **Aplicação em ML:** Naive Bayes, Bayesian Optimization, Probabilistic Programming.

### 3.4 Testes de Hipótese

```python
from scipy import stats
import numpy as np

# T-test: comparar duas médias
grupo_a = np.random.normal(50, 10, 100)  # controle
grupo_b = np.random.normal(55, 10, 100)  # tratamento

t_stat, p_value = stats.ttest_ind(grupo_a, grupo_b)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significativo (α=0.05)? {'Sim' if p_value < 0.05 else 'Não'}")

# Chi-square test: independência de variáveis categóricas
contingencia = np.array([[45, 55], [30, 70]])  # tabela de contingência
chi2, p_value, dof, expected = stats.chi2_contingency(contingencia)
print(f"\nChi²: {chi2:.4f}, P-value: {p_value:.4f}")

# A/B Testing
def ab_test(control, treatment, alpha=0.05):
    """Realiza A/B test entre controle e tratamento"""
    t_stat, p_value = stats.ttest_ind(control, treatment)
    
    effect_size = (np.mean(treatment) - np.mean(control)) / np.sqrt(
        (np.std(control)**2 + np.std(treatment)**2) / 2
    )  # Cohen's d
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "effect_size": effect_size,
        "control_mean": np.mean(control),
        "treatment_mean": np.mean(treatment),
        "lift": (np.mean(treatment) - np.mean(control)) / np.mean(control) * 100
    }

resultado = ab_test(grupo_a, grupo_b)
for k, v in resultado.items():
    print(f"  {k}: {v}")
```

### 3.5 Correlação

```python
import numpy as np
from scipy import stats

x = np.random.randn(100)
y = 2*x + np.random.randn(100) * 0.5

# Pearson (linear)
r_pearson, p_pearson = stats.pearsonr(x, y)
print(f"Pearson r: {r_pearson:.4f} (p={p_pearson:.4f})")

# Spearman (monotônica, não-paramétrica)
r_spearman, p_spearman = stats.spearmanr(x, y)
print(f"Spearman ρ: {r_spearman:.4f} (p={p_spearman:.4f})")

# Matriz de correlação
dados = np.column_stack([x, y, np.random.randn(100)])
corr_matrix = np.corrcoef(dados.T)
print(f"\nMatriz de Correlação:\n{corr_matrix}")
```

---

## 🏋️ Exercícios

1. **Implemente Gradient Descent** do zero para regressão logística
2. **Calcule manualmente** as derivadas parciais de $L = (y - (w_1x_1 + w_2x_2 + b))^2$
3. **Use PCA** (com eigendecomposition) para reduzir um dataset de 10 features para 2
4. **Realize um A/B test completo** com cálculo de power e sample size
5. **Implemente Naive Bayes** do zero usando o Teorema de Bayes

---

## 📝 Notas

> Adicione aqui suas anotações pessoais.
