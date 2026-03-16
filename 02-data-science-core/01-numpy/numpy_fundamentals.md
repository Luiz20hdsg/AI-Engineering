# 🔢 NumPy — Computação Numérica

> NumPy é a base de todo o ecossistema de Data Science em Python. Pandas, Scikit-learn, PyTorch — todos são construídos sobre NumPy.

---

## 1. Arrays — O Fundamento

```python
import numpy as np

# Criação de arrays
a = np.array([1, 2, 3, 4, 5])           # 1D
b = np.array([[1, 2, 3], [4, 5, 6]])    # 2D (matrix)
c = np.zeros((3, 4))                     # 3x4 de zeros
d = np.ones((2, 3))                      # 2x3 de uns
e = np.eye(4)                            # identidade 4x4
f = np.arange(0, 10, 0.5)               # [0, 0.5, 1.0, ..., 9.5]
g = np.linspace(0, 1, 100)              # 100 pontos entre 0 e 1
h = np.random.randn(3, 4)               # normal(0,1) shape 3x4

# Propriedades
print(f"Shape: {b.shape}")    # (2, 3)
print(f"Dtype: {b.dtype}")    # int64
print(f"Ndim: {b.ndim}")      # 2
print(f"Size: {b.size}")      # 6
```

## 2. Indexação e Slicing

```python
arr = np.arange(20).reshape(4, 5)
# [[ 0,  1,  2,  3,  4],
#  [ 5,  6,  7,  8,  9],
#  [10, 11, 12, 13, 14],
#  [15, 16, 17, 18, 19]]

# Básico
arr[0, 0]      # 0
arr[1, 3]      # 8
arr[2]         # [10, 11, 12, 13, 14] — linha inteira
arr[:, 1]      # [1, 6, 11, 16] — coluna inteira

# Slicing
arr[1:3, 2:4]  # [[7, 8], [12, 13]]
arr[::2]       # linhas pares: [[0,1,2,3,4], [10,11,12,13,14]]

# Boolean indexing (MUITO usado em data science)
mask = arr > 10
arr[mask]              # [11, 12, 13, 14, 15, 16, 17, 18, 19]
arr[arr % 2 == 0]     # todos os pares

# Fancy indexing
arr[[0, 2, 3], [1, 3, 4]]  # [1, 13, 19] — elementos específicos
```

## 3. Broadcasting

```python
# Broadcasting permite operações entre arrays de shapes diferentes
a = np.array([[1], [2], [3]])  # shape (3, 1)
b = np.array([10, 20, 30])     # shape (3,) → broadcast para (1, 3)

resultado = a + b
# [[11, 21, 31],
#  [12, 22, 32],
#  [13, 23, 33]]

# Normalização com broadcasting
dados = np.random.randn(1000, 5)  # 1000 amostras, 5 features
media = dados.mean(axis=0)         # shape (5,)
std = dados.std(axis=0)            # shape (5,)
dados_norm = (dados - media) / std  # broadcasting automático!
```

## 4. Operações Vetorizadas (NUNCA use loops!)

```python
# ❌ ERRADO (lento)
result = []
for i in range(len(a)):
    result.append(a[i] ** 2 + b[i])

# ✅ CERTO (rápido — vetorizado)
result = a ** 2 + b

# Comparação de performance
import time
n = 1_000_000
a = np.random.randn(n)

# Loop Python
start = time.time()
result_loop = [x**2 for x in a]
print(f"Loop: {time.time() - start:.4f}s")

# NumPy vetorizado
start = time.time()
result_numpy = a ** 2
print(f"NumPy: {time.time() - start:.4f}s")
# NumPy é tipicamente 50-100x mais rápido!

# Funções universais (ufuncs)
np.sqrt(a)
np.exp(a)
np.log(np.abs(a))
np.sin(a)
np.clip(a, -1, 1)  # limitar valores

# Aggregações
a.sum(), a.mean(), a.std(), a.min(), a.max()
a.argmin(), a.argmax()  # índice do min/max
np.percentile(a, [25, 50, 75])
```

## 5. Álgebra Linear com NumPy

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
v = np.array([1, 2])

# Operações
A @ B                    # multiplicação de matrizes
A @ v                    # matrix-vector
np.dot(A, B)             # equivalente a @
A.T                      # transposta
np.linalg.inv(A)         # inversa
np.linalg.det(A)         # determinante
np.linalg.eig(A)         # autovalores/vetores
np.linalg.svd(A)         # SVD
np.linalg.solve(A, v)    # resolver Ax = v
np.linalg.norm(v)        # norma L2
np.linalg.norm(v, ord=1) # norma L1
```

---

## 🏋️ Exercícios

1. Crie uma função que normaliza cada coluna de uma matriz usando vetorização
2. Implemente multiplicação de matrizes sem usar `@` ou `np.dot` (apenas para entender)
3. Use boolean indexing para filtrar outliers (> 3 desvios padrão)
4. Implemente softmax usando NumPy: $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$
