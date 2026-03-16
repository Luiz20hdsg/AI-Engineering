# ⚙️ MLOps — Guia Prático

---

## 1. O Que é MLOps?

```
MLOps = Machine Learning + DevOps + Data Engineering

Objetivo: Automatizar e monitorar todo o ciclo de vida de modelos de ML em produção.
```

### MLOps Maturity Model

| Nível | Descrição | Características |
|-------|-----------|----------------|
| **0** | Manual | Notebooks, sem automação, sem CI/CD |
| **1** | ML Pipeline | Pipeline automatizado, experiment tracking |
| **2** | CI/CD | Testes automatizados, deploy automatizado |
| **3** | Automated Retraining | Retraining com triggers, monitoring |
| **4** | Full MLOps | A/B testing, feature store, ML platform |

---

## 2. Experiment Tracking com MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Configurar MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # ou servidor remoto
mlflow.set_experiment("credit-fraud-detection")

# Treinar com tracking
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": 42,
    "class_weight": "balanced"
}

with mlflow.start_run(run_name="rf_baseline"):
    # Log de parâmetros
    mlflow.log_params(params)
    
    # Treinar
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Log de métricas
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob)
    })
    
    # Log do modelo
    mlflow.sklearn.log_model(model, "model")
    
    # Log de artefatos (gráficos, etc.)
    # mlflow.log_artifact("confusion_matrix.png")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# Model Registry
mlflow.register_model(
    f"runs:/{run_id}/model",
    "credit-fraud-model"
)
```

---

## 3. Docker para ML

### Dockerfile

```dockerfile
# Dockerfile para API de ML
FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro (cache de camadas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/latest
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/models
    deploy:
      resources:
        limits:
          memory: 2G

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root ./mlruns

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

---

## 4. Model Serving com FastAPI

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import time
from contextlib import asynccontextmanager

# Modelo global
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carregar modelo na inicialização"""
    global model
    model = joblib.load("models/pipeline.pkl")
    print("✅ Modelo carregado com sucesso")
    yield
    print("🔄 Shutting down...")

app = FastAPI(
    title="ML Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

# Schemas
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [25.0, 50000.0, 3.0, 1.0, 0.0]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    latency_ms: float

class BatchPredictionRequest(BaseModel):
    instances: List[List[float]]

class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    latency_ms: float

# Endpoints
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start = time.time()
    
    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0].max())
        
        latency = (time.time() - start) * 1000
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    start = time.time()
    
    X = np.array(request.instances)
    predictions = model.predict(X).tolist()
    probabilities = model.predict_proba(X).max(axis=1).tolist()
    
    latency = (time.time() - start) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        probabilities=probabilities,
        latency_ms=round(latency, 2)
    )
```

---

## 5. CI/CD com GitHub Actions

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      
      - name: Run data tests
        run: pytest tests/test_data.py -v
      
      - name: Run model tests
        run: pytest tests/test_model.py -v
      
      - name: Run API tests
        run: pytest tests/test_api.py -v

  train:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Train model
        run: python scripts/train.py
      
      - name: Evaluate model
        run: python scripts/evaluate.py
      
      - name: Upload model artifact
        uses: actions/upload-artifact@v4
        with:
          name: model
          path: models/

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: echo "Deploy model to production"
```

### Testes para ML

```python
# tests/test_data.py
import pytest
import pandas as pd
import numpy as np

def test_data_schema():
    """Verifica se o schema dos dados está correto"""
    df = pd.read_csv("data/processed/train.csv")
    expected_columns = ['feature_1', 'feature_2', 'target']
    assert all(col in df.columns for col in expected_columns)

def test_no_data_leakage():
    """Verifica se não há data leakage entre train e test"""
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    train_ids = set(train['id'])
    test_ids = set(test['id'])
    assert len(train_ids.intersection(test_ids)) == 0

def test_feature_ranges():
    """Verifica se features estão em ranges esperados"""
    df = pd.read_csv("data/processed/train.csv")
    assert df['idade'].between(0, 120).all()
    assert df['salario'].ge(0).all()

# tests/test_model.py
def test_model_accuracy():
    """Verifica se o modelo atinge accuracy mínima"""
    model = joblib.load("models/pipeline.pkl")
    X_test, y_test = load_test_data()
    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.8, f"Accuracy {accuracy} abaixo do mínimo (0.8)"

def test_model_inference_time():
    """Verifica se inferência é rápida o suficiente"""
    model = joblib.load("models/pipeline.pkl")
    X = np.random.randn(1, 10)
    
    import time
    start = time.time()
    model.predict(X)
    latency = (time.time() - start) * 1000
    
    assert latency < 100, f"Latência {latency}ms excede limite de 100ms"
```

---

## 6. Monitoring

```python
# monitoring/drift_detection.py
from scipy import stats
import numpy as np

class DataDriftDetector:
    """Detecta drift nos dados de produção vs treinamento"""
    
    def __init__(self, reference_data, threshold=0.05):
        self.reference = reference_data
        self.threshold = threshold
    
    def check_drift(self, production_data):
        """KS test para cada feature"""
        results = {}
        for col in self.reference.columns:
            stat, p_value = stats.ks_2samp(
                self.reference[col], 
                production_data[col]
            )
            results[col] = {
                'statistic': stat,
                'p_value': p_value,
                'drift_detected': p_value < self.threshold
            }
        
        drifted_features = [k for k, v in results.items() if v['drift_detected']]
        
        return {
            'features': results,
            'total_drifted': len(drifted_features),
            'drifted_features': drifted_features,
            'alert': len(drifted_features) > 0
        }

# Performance monitoring
class ModelMonitor:
    """Monitora performance do modelo em produção"""
    
    def __init__(self, baseline_metrics):
        self.baseline = baseline_metrics
        self.history = []
    
    def log_prediction(self, features, prediction, actual=None):
        self.history.append({
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'timestamp': pd.Timestamp.now()
        })
    
    def check_performance(self, window_size=1000):
        recent = self.history[-window_size:]
        actuals = [h for h in recent if h['actual'] is not None]
        
        if len(actuals) < 100:
            return {"status": "insufficient_data"}
        
        y_true = [h['actual'] for h in actuals]
        y_pred = [h['prediction'] for h in actuals]
        
        current_accuracy = sum(a == p for a, p in zip(y_true, y_pred)) / len(y_true)
        
        return {
            "current_accuracy": current_accuracy,
            "baseline_accuracy": self.baseline['accuracy'],
            "degradation": self.baseline['accuracy'] - current_accuracy,
            "alert": current_accuracy < self.baseline['accuracy'] * 0.95
        }
```

---

## 🏋️ Exercícios

1. **Setup MLflow** localmente e faça tracking de 5 experimentos diferentes
2. **Crie uma API** com FastAPI que serve um modelo treinado
3. **Dockerize** sua API e faça deploy local com Docker Compose
4. **Configure CI/CD** com GitHub Actions para seu projeto ML
5. **Implemente** data drift detection para um modelo em produção
6. **Crie um pipeline completo**: train → test → deploy → monitor

---

## 📝 Notas

> Adicione aqui suas anotações pessoais conforme avança nos estudos.
