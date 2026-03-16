# 🐍 Python Avançado para Data Science

## 1. Estruturas de Dados Avançadas

### Collections Module

```python
from collections import Counter, defaultdict, namedtuple, deque, OrderedDict

# Counter - contar ocorrências
palavras = ['ml', 'ai', 'ml', 'data', 'ai', 'ml']
contagem = Counter(palavras)
print(contagem)  # Counter({'ml': 3, 'ai': 2, 'data': 1})
print(contagem.most_common(2))  # [('ml', 3), ('ai', 2)]

# defaultdict - dicionário com valor padrão
grupo_por_tipo = defaultdict(list)
dados = [('fruta', 'maçã'), ('vegetal', 'cenoura'), ('fruta', 'banana')]
for tipo, item in dados:
    grupo_por_tipo[tipo].append(item)
print(dict(grupo_por_tipo))  # {'fruta': ['maçã', 'banana'], 'vegetal': ['cenoura']}

# namedtuple - tupla com nomes
Ponto = namedtuple('Ponto', ['x', 'y', 'z'])
p = Ponto(1, 2, 3)
print(p.x, p.y, p.z)  # 1 2 3

# deque - fila de dupla entrada (O(1) nas pontas)
fila = deque([1, 2, 3])
fila.appendleft(0)  # [0, 1, 2, 3]
fila.append(4)       # [0, 1, 2, 3, 4]
fila.popleft()        # remove 0
```

### Itertools

```python
import itertools

# product - produto cartesiano
cores = ['red', 'blue']
tamanhos = ['S', 'M', 'L']
combinacoes = list(itertools.product(cores, tamanhos))
# [('red', 'S'), ('red', 'M'), ('red', 'L'), ('blue', 'S'), ...]

# chain - encadear iteráveis
lista1 = [1, 2, 3]
lista2 = [4, 5, 6]
todos = list(itertools.chain(lista1, lista2))  # [1, 2, 3, 4, 5, 6]

# groupby - agrupar por chave
dados = [('A', 1), ('A', 2), ('B', 3), ('B', 4)]
for chave, grupo in itertools.groupby(dados, key=lambda x: x[0]):
    print(chave, list(grupo))

# combinations e permutations
print(list(itertools.combinations([1, 2, 3], 2)))  # [(1,2), (1,3), (2,3)]
print(list(itertools.permutations([1, 2, 3], 2)))   # [(1,2), (1,3), (2,1), ...]

# islice - fatiar iteradores
numeros = itertools.count(0)  # 0, 1, 2, 3, ...
primeiros_10 = list(itertools.islice(numeros, 10))
```

---

## 2. Comprehensions Avançadas

```python
# List comprehension com condição
pares = [x for x in range(20) if x % 2 == 0]

# Nested comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Dict comprehension
quadrados = {x: x**2 for x in range(10)}

# Set comprehension
unicos = {palavra.lower() for palavra in ['Hello', 'World', 'hello', 'WORLD']}
# {'hello', 'world'}

# Comprehension com walrus operator (:=)
resultados = [y for x in range(10) if (y := x**2) > 20]
# [25, 36, 49, 64, 81]
```

---

## 3. Generators e Iterators

```python
# Generator function
def fibonacci(n):
    """Gera os primeiros n números de Fibonacci"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for num in fibonacci(10):
    print(num, end=' ')  # 0 1 1 2 3 5 8 13 21 34

# Generator expression (memory efficient)
soma_quadrados = sum(x**2 for x in range(1_000_000))

# Generator para processar arquivos grandes
def ler_arquivo_em_chunks(filepath, chunk_size=1024):
    with open(filepath, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Iterator Protocol
class Countdown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1
```

---

## 4. Decorators

```python
import functools
import time

# Decorator básico
def timer(func):
    """Mede o tempo de execução de uma função"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} executou em {elapsed:.4f}s")
        return result
    return wrapper

@timer
def treinar_modelo():
    time.sleep(1)
    return "modelo treinado"

# Decorator com parâmetros
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Tentativa {attempt} falhou: {e}. Retentando em {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def chamar_api():
    # simula chamada que pode falhar
    import random
    if random.random() < 0.7:
        raise ConnectionError("API indisponível")
    return {"status": "ok"}

# Decorator para cache/memoização
@functools.lru_cache(maxsize=128)
def fibonacci_memo(n):
    if n < 2:
        return n
    return fibonacci_memo(n-1) + fibonacci_memo(n-2)

# Class-based decorator
class ValidateInput:
    def __init__(self, validator):
        self.validator = validator
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.validator(*args, **kwargs):
                raise ValueError("Input inválido!")
            return func(*args, **kwargs)
        return wrapper

@ValidateInput(lambda x: x > 0)
def raiz_quadrada(x):
    return x ** 0.5
```

---

## 5. Context Managers

```python
from contextlib import contextmanager
import time

# Context Manager com classe
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"Tempo: {self.elapsed:.4f}s")
        return False  # não suprime exceções

with Timer() as t:
    # código que queremos medir
    sum(range(1_000_000))

# Context Manager com decorator
@contextmanager
def database_connection(db_url):
    """Gerencia conexão com banco de dados"""
    print(f"Conectando a {db_url}...")
    conn = {"url": db_url, "status": "connected"}  # simula conexão
    try:
        yield conn
    finally:
        print(f"Fechando conexão com {db_url}")
        conn["status"] = "closed"

with database_connection("postgresql://localhost/ml_db") as conn:
    print(f"Status: {conn['status']}")

# Context Manager para criar diretório temporário
@contextmanager
def temporary_directory():
    import tempfile
    import shutil
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)
```

---

## 6. Type Hints e Dataclasses

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Callable
from enum import Enum

# Type Hints avançados
def treinar(
    dados: List[Dict[str, float]],
    epochs: int = 100,
    lr: float = 0.01,
    callback: Optional[Callable[[int, float], None]] = None
) -> Tuple[float, List[float]]:
    """Treina modelo e retorna (acurácia, histórico_loss)"""
    ...

# Dataclass — substituto moderno para classes de dados
@dataclass
class ModelConfig:
    name: str
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    optimizer: str = "adam"
    
    def total_params_estimate(self) -> int:
        total = 0
        for i in range(1, len(self.layers)):
            total += self.layers[i-1] * self.layers[i]
        return total

config = ModelConfig(
    name="classificador_v1",
    learning_rate=0.0001,
    layers=[256, 128, 64, 10]
)
print(config)
print(f"Params estimados: {config.total_params_estimate()}")

# Dataclass com frozen (imutável)
@dataclass(frozen=True)
class Hyperparameters:
    lr: float
    batch_size: int
    dropout: float

# Enum para categorias
class ModelType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"

# Dataclass com validação
@dataclass
class TrainingRun:
    model_type: ModelType
    accuracy: float
    loss: float
    
    def __post_init__(self):
        if not 0 <= self.accuracy <= 1:
            raise ValueError(f"Accuracy deve estar entre 0 e 1, got {self.accuracy}")
        if self.loss < 0:
            raise ValueError(f"Loss deve ser >= 0, got {self.loss}")
```

---

## 7. OOP Avançado

```python
from abc import ABC, abstractmethod
from typing import List

# Abstract Base Class
class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self._is_trained = False
    
    @abstractmethod
    def fit(self, X, y):
        """Treina o modelo"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Faz previsões"""
        pass
    
    def evaluate(self, X, y):
        """Template method — usa predict internamente"""
        predictions = self.predict(X)
        return self._calculate_metric(predictions, y)
    
    @abstractmethod
    def _calculate_metric(self, predictions, y):
        pass

# Mixin
class LoggingMixin:
    def log(self, message: str):
        print(f"[{self.__class__.__name__}] {message}")

class SerializationMixin:
    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Modelo salvo em {path}")
    
    @classmethod
    def load(cls, path: str):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

# Classe concreta usando ABC + Mixins
class LinearRegression(BaseModel, LoggingMixin, SerializationMixin):
    def __init__(self, name: str = "LinearRegression"):
        super().__init__(name)
        self.weights = None
    
    def fit(self, X, y):
        self.log(f"Treinando com {len(X)} amostras")
        # Implementação simplificada
        self._is_trained = True
        self.log("Treinamento completo")
    
    def predict(self, X):
        if not self._is_trained:
            raise RuntimeError("Modelo não treinado!")
        return [0] * len(X)  # placeholder
    
    def _calculate_metric(self, predictions, y):
        # MSE simplificado
        return sum((p - t)**2 for p, t in zip(predictions, y)) / len(y)

# Property e descriptors
class Feature:
    def __init__(self, name: str):
        self._name = name
        self._values: List[float] = []
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0
    
    @property
    def std(self) -> float:
        if not self._values:
            return 0
        m = self.mean
        return (sum((x - m)**2 for x in self._values) / len(self._values)) ** 0.5
    
    def add_values(self, values: List[float]):
        self._values.extend(values)
    
    def normalize(self) -> List[float]:
        """Z-score normalization"""
        m, s = self.mean, self.std
        return [(x - m) / s if s != 0 else 0 for x in self._values]
```

---

## 8. Programação Funcional

```python
from functools import reduce, partial
from operator import add, mul

# Map, Filter, Reduce
numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

quadrados = list(map(lambda x: x**2, numeros))
pares = list(filter(lambda x: x % 2 == 0, numeros))
soma = reduce(add, numeros)
produto = reduce(mul, numeros)

# Partial — fixar argumentos
def potencia(base, expoente):
    return base ** expoente

quadrado = partial(potencia, expoente=2)
cubo = partial(potencia, expoente=3)

print(quadrado(5))  # 25
print(cubo(3))      # 27

# Pipeline funcional
from typing import Callable, List

def pipeline(*funcs: Callable) -> Callable:
    """Cria um pipeline de funções"""
    def apply(data):
        result = data
        for func in funcs:
            result = func(result)
        return result
    return apply

# Exemplo: pipeline de pré-processamento
remover_nulos = lambda dados: [x for x in dados if x is not None]
converter_float = lambda dados: [float(x) for x in dados]
normalizar = lambda dados: [(x - min(dados)) / (max(dados) - min(dados)) for x in dados]

preprocessar = pipeline(remover_nulos, converter_float, normalizar)

dados_brutos = [1, None, 3, 5, None, 8, 10]
dados_limpos = preprocessar(dados_brutos)
print(dados_limpos)  # [0.0, 0.222..., 0.444..., 0.777..., 1.0]
```

---

## 🏋️ Exercícios Práticos

1. **Crie um data loader** usando generators que leia um CSV grande linha por linha
2. **Implemente um decorator** `@validate_dataframe` que checa se um DataFrame tem as colunas esperadas
3. **Crie uma classe Pipeline** com ABC que permita encadear transformações
4. **Implemente cache com TTL** usando um decorator que expira após N segundos
5. **Crie um context manager** para logging de experimentos ML

---

## 📝 Notas

> Adicione aqui suas anotações pessoais conforme avança nos estudos.
