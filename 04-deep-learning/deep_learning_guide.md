# 🧠 Deep Learning — Fundamentos e PyTorch

---

## 1. Neural Networks — Fundamentos

### 1.1 O Neurônio Artificial

$$z = \sum_{i=1}^n w_i x_i + b = \mathbf{w}^T\mathbf{x} + b$$
$$a = \sigma(z)$$

Onde $\sigma$ é a função de ativação.

### 1.2 Funções de Ativação

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

x = torch.linspace(-5, 5, 100)

# ReLU — a mais popular para hidden layers
relu = F.relu(x)                    # max(0, x)

# Sigmoid — para output de classificação binária
sigmoid = torch.sigmoid(x)          # 1 / (1 + e^(-x))

# Tanh — simétrica, centrada em zero
tanh = torch.tanh(x)                # (e^x - e^(-x)) / (e^x + e^(-x))

# GELU — usada em Transformers (BERT, GPT)
gelu = F.gelu(x)                    # x * Φ(x)

# Swish/SiLU — variante suave do ReLU
swish = F.silu(x)                   # x * sigmoid(x)

# Softmax — para output multi-class
logits = torch.tensor([2.0, 1.0, 0.5])
probs = F.softmax(logits, dim=0)     # soma = 1.0
```

### Quando usar qual ativação?

| Ativação | Onde Usar | Por quê |
|----------|----------|---------|
| **ReLU** | Hidden layers (padrão) | Simples, eficiente, resolve vanishing gradient |
| **GELU** | Transformers | Mais suave que ReLU, melhor para NLP |
| **Sigmoid** | Output binário | Mapeia para [0, 1] |
| **Softmax** | Output multi-class | Distribui probabilidade entre classes |
| **Tanh** | RNNs, normalizações | Output em [-1, 1] |

### 1.3 Loss Functions

```python
# Binary Cross-Entropy — classificação binária
loss_bce = F.binary_cross_entropy_with_logits(logits, targets)

# Cross-Entropy — classificação multi-class
loss_ce = F.cross_entropy(logits, class_indices)

# MSE — regressão
loss_mse = F.mse_loss(predictions, targets)

# L1 Loss (MAE) — regressão robusta a outliers
loss_l1 = F.l1_loss(predictions, targets)

# Focal Loss — para dados muito desbalanceados
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

---

## 2. PyTorch — Fundamentos

### 2.1 Tensors

```python
import torch

# Criação
x = torch.tensor([1.0, 2.0, 3.0])
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
rand = torch.randn(3, 4)           # normal(0, 1)
arange = torch.arange(0, 10, 0.5)

# Propriedades
print(x.shape, x.dtype, x.device)  # torch.Size([3]), torch.float32, cpu

# GPU / MPS (Apple Silicon)
device = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
x = x.to(device)

# Operações (idênticas a NumPy)
a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = a @ b                # matmul
d = a * 2                # element-wise
e = a.sum(dim=1)         # soma por linha
f = a.mean(dim=0)        # média por coluna

# Reshape
x = torch.randn(2, 3, 4)
x.view(6, 4)             # reshape (contíguo)
x.reshape(2, 12)         # reshape (flexível)
x.permute(2, 0, 1)       # transpor dimensões
x.unsqueeze(0)           # adicionar dimensão
x.squeeze()              # remover dimensões de tamanho 1
```

### 2.2 Autograd — Diferenciação Automática

```python
# Autograd calcula gradientes automaticamente!
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x[0]**2 + 3*x[1]      # y = x₀² + 3x₁
y.backward()                 # calcula gradientes
print(x.grad)               # tensor([4., 3.]) = [2*x₀, 3]

# No contexto de ML
w = torch.randn(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

X = torch.randn(100, 3)
y_true = torch.randn(100, 1)

# Forward pass
y_pred = X @ w + b
loss = ((y_pred - y_true) ** 2).mean()

# Backward pass — calcula todos os gradientes
loss.backward()

print(w.grad)  # ∂loss/∂w
print(b.grad)  # ∂loss/∂b

# Atualizar pesos (gradient descent manual)
with torch.no_grad():
    w -= 0.01 * w.grad
    b -= 0.01 * b.grad
    w.grad.zero_()  # IMPORTANTE: limpar gradientes
    b.grad.zero_()
```

### 2.3 Construindo um Modelo com nn.Module

```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Instanciar
model = NeuralNetwork(input_dim=10, hidden_dim=128, output_dim=2)
print(model)
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
```

### 2.4 Dataset e DataLoader

```python
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Criar datasets
train_dataset = TabularDataset(X_train, y_train)
test_dataset = TabularDataset(X_test, y_test)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### 2.5 Training Loop Completo

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(10, 128, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Training
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * X_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += X_batch.size(0)
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        total_loss += loss.item() * X_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += X_batch.size(0)
    
    return total_loss / total, correct / total

# Training loop
best_val_acc = 0
epochs = 50

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f}")

print(f"\nMelhor Val Acc: {best_val_acc:.4f}")

# Carregar melhor modelo
model.load_state_dict(torch.load('best_model.pt'))
```

---

## 3. Transformers — A Arquitetura que Mudou Tudo

### 3.1 Self-Attention (Conceito Central)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Calcular Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Pre-norm architecture (como no GPT-2+)
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x
```

### 3.2 Usando Hugging Face Transformers

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import load_dataset

# Carregar modelo pré-treinado
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenizar
text = "This is an example sentence for classification."
tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
print(tokens)

# Inferência
outputs = model(**tokens)
predictions = torch.softmax(outputs.logits, dim=-1)
print(f"Classe 0: {predictions[0][0]:.4f}")
print(f"Classe 1: {predictions[0][1]:.4f}")

# Fine-tuning com Trainer API
dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()
```

---

## 🏋️ Exercícios

1. **Implemente um MLP** do zero em PyTorch para classificar MNIST
2. **Implemente Self-Attention** do zero e visualize os attention weights
3. **Fine-tune BERT** para classificação de sentimentos
4. **Treine uma CNN** para classificação de imagens (CIFAR-10)
5. **Compare** training com e sem learning rate scheduling
6. **Implemente** mixed precision training e meça o speedup

---

## 📝 Notas

> Adicione aqui suas anotações pessoais conforme avança nos estudos.
