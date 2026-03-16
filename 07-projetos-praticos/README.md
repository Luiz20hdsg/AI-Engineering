# 🚀 Módulo 7 — Projetos Práticos

> **Objetivo:** Construir um portfólio sólido com projetos end-to-end que demonstrem suas habilidades

---

## 📋 Projetos por Nível

### 🟢 Nível Iniciante (Módulos 1-2)

#### Projeto 1: EDA & Visualização — Análise de Dataset Real
- **Dataset:** Kaggle (ex: Titanic, House Prices, Netflix)
- **Skills:** Pandas, Matplotlib, Seaborn, Estatística
- **Deliverables:**
  - [ ] Notebook com EDA completa
  - [ ] Visualizações informativas
  - [ ] Relatório com insights
  - [ ] README documentado

#### Projeto 2: Dashboard de Dados
- **Stack:** Streamlit ou Plotly Dash
- **Skills:** Visualização interativa, deploy
- **Deliverables:**
  - [ ] Dashboard interativo
  - [ ] Deploy no Streamlit Cloud

---

### 🟡 Nível Intermediário (Módulos 3-4)

#### Projeto 3: ML Pipeline — Previsão de Churn
- **Dataset:** Telco Customer Churn (Kaggle)
- **Skills:** Feature Engineering, ML, Pipelines, Evaluation
- **Deliverables:**
  - [ ] Pipeline completo (preprocessing → training → evaluation)
  - [ ] Comparação de 5+ modelos
  - [ ] Hyperparameter tuning com Optuna
  - [ ] Feature importance analysis
  - [ ] Relatório final com métricas

#### Projeto 4: Deep Learning — Classificação de Imagens
- **Dataset:** CIFAR-10, Food101, ou custom dataset
- **Skills:** PyTorch, CNNs, Transfer Learning, Data Augmentation
- **Deliverables:**
  - [ ] Modelo CNN treinado
  - [ ] Comparação: treinar do zero vs transfer learning
  - [ ] Training curves e análise de erros
  - [ ] Inference pipeline

#### Projeto 5: NLP — Classificação de Texto
- **Dataset:** IMDB Reviews, News Classification
- **Skills:** Hugging Face, Transformers, Fine-tuning
- **Deliverables:**
  - [ ] Fine-tune BERT para classificação
  - [ ] Comparação com baseline (TF-IDF + ML clássico)
  - [ ] Error analysis

---

### 🔴 Nível Avançado (Módulos 5-6)

#### Projeto 6: MLOps — Modelo em Produção
- **Stack:** FastAPI, Docker, MLflow, GitHub Actions
- **Skills:** Model Serving, CI/CD, Monitoring
- **Deliverables:**
  - [ ] API REST para predições
  - [ ] Docker + Docker Compose
  - [ ] Experiment tracking com MLflow
  - [ ] CI/CD pipeline (GitHub Actions)
  - [ ] Testes automatizados
  - [ ] Documentação da API (Swagger)

#### Projeto 7: RAG System — Q&A sobre Documentos
- **Stack:** LangChain, ChromaDB/Pinecone, OpenAI/Claude
- **Skills:** RAG, Embeddings, Vector Search, Prompt Engineering
- **Deliverables:**
  - [ ] Sistema RAG funcional
  - [ ] Interface com Streamlit/Gradio
  - [ ] Evaluation com RAGAS
  - [ ] Comparação de chunking strategies
  - [ ] Hybrid search (dense + sparse)

#### Projeto 8: AI Agent — Assistente de Análise de Dados
- **Stack:** LangChain/LangGraph, OpenAI Function Calling
- **Skills:** Agents, Tool Use, Multi-step reasoning
- **Deliverables:**
  - [ ] Agent que analisa dados automaticamente
  - [ ] Tools: Python executor, SQL query, visualização
  - [ ] Memory de conversa
  - [ ] Interface conversacional

#### Projeto 9: Fine-tuning de LLM
- **Stack:** Hugging Face, PEFT/LoRA, Weights & Biases
- **Skills:** Fine-tuning, Dataset preparation, Evaluation
- **Deliverables:**
  - [ ] Dataset curado para fine-tuning
  - [ ] Modelo fine-tuned com LoRA
  - [ ] Comparação: base vs fine-tuned
  - [ ] Benchmark de performance

---

### ⭐ Projeto Capstone

#### Projeto 10: Plataforma End-to-End de ML/AI
Combine tudo que aprendeu em um projeto grande:

- **Ideia:** Plataforma que recebe dados, treina modelos, serve predições e responde perguntas
- **Componentes:**
  - [ ] Data pipeline (ingestão → processamento → storage)
  - [ ] ML training pipeline (Feature Eng → Training → Eval → Registry)
  - [ ] Model serving (API REST com FastAPI)
  - [ ] RAG system para documentação interna
  - [ ] Dashboard de monitoring
  - [ ] CI/CD completo
  - [ ] Docker Compose para orquestração

---

## 📁 Template de Projeto

```
projeto/
├── README.md              # Descrição, setup, resultados
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-feature-eng.ipynb
│   └── 03-modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── configs/
│   └── config.yaml
├── models/                # Modelos salvos
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 📝 Dicas para Portfólio

1. **README impecável** — é a primeira coisa que recrutadores veem
2. **Mostre resultados** — gráficos, métricas, comparações
3. **Documente decisões** — por que escolheu esse modelo/approach?
4. **Código limpo** — type hints, docstrings, testes
5. **Deploy** — mostre que sabe colocar em produção
6. **Diversifique** — NLP, CV, tabular, MLOps, AI Engineering
