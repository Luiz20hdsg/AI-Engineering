# 🤖 AI Engineering — Guia Completo

---

## 1. LLMs — Fundamentos

### 1.1 Como LLMs Funcionam

```
Input Text → Tokenizer → Token IDs → Transformer Model → Logits → Sampling → Output Token
                                          ↑
                                    Bilhões de parâmetros
                                    treinados em trilhões de tokens
```

**Conceitos chave:**
- **Pre-training:** Modelo aprende a prever o próximo token (GPT) ou tokens mascarados (BERT)
- **Instruction tuning:** Fine-tune para seguir instruções
- **RLHF/DPO:** Alignment com preferências humanas
- **Context Window:** Quantidade máxima de tokens que o modelo pode processar

### 1.2 Usando APIs

```python
from openai import OpenAI
import anthropic

# ---- OpenAI ----
client = OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Você é um especialista em ML."},
        {"role": "user", "content": "Explique overfitting em 3 frases."}
    ],
    temperature=0.7,      # 0=determinístico, 1=criativo
    max_tokens=500,
    top_p=0.9,
)
print(response.choices[0].message.content)

# ---- Anthropic (Claude) ----
client = anthropic.Anthropic(api_key="your-key")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="Você é um especialista em Data Science.",
    messages=[
        {"role": "user", "content": "Qual a diferença entre Random Forest e XGBoost?"}
    ]
)
print(message.content[0].text)

# ---- Structured Output (JSON) ----
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extraia informações do texto e retorne em JSON."},
        {"role": "user", "content": "Alice tem 30 anos, mora em São Paulo e trabalha como engenheira de ML."}
    ],
    response_format={"type": "json_object"}
)
import json
data = json.loads(response.choices[0].message.content)
print(data)
# {"nome": "Alice", "idade": 30, "cidade": "São Paulo", "cargo": "engenheira de ML"}
```

---

## 2. Prompt Engineering

### 2.1 Técnicas Fundamentais

```python
# ---- Zero-Shot ----
prompt = "Classifique o sentimento como POSITIVO, NEGATIVO ou NEUTRO: 'O produto é excelente!'"

# ---- Few-Shot ----
prompt = """Classifique o sentimento:

Texto: "Adorei o produto!" → POSITIVO
Texto: "Péssima qualidade" → NEGATIVO  
Texto: "O produto é ok" → NEUTRO
Texto: "Melhor compra que já fiz!" → """

# ---- Chain-of-Thought (CoT) ----
prompt = """Resolva passo a passo:

Pergunta: Se tenho um dataset com 10.000 amostras, 95% são classe 0 e 5% são classe 1,
qual seria a accuracy de um modelo que sempre prediz classe 0?

Pensamento passo a passo:
1. Total de amostras: 10.000
2. Classe 0: 95% × 10.000 = 9.500 amostras
3. Se o modelo sempre prediz classe 0, acerta todas as 9.500
4. Accuracy = 9.500 / 10.000 = 95%

Resposta: 95%. Isso mostra por que accuracy é uma métrica ruim para dados desbalanceados."""

# ---- System Prompt Robusto ----
system_prompt = """Você é um engenheiro de ML senior com 10 anos de experiência.

## Diretrizes:
1. Sempre forneça código funcional em Python
2. Explique o raciocínio por trás de cada decisão
3. Mencione trade-offs e alternativas
4. Use type hints e docstrings
5. Se não souber algo, diga claramente

## Formato de resposta:
- Comece com um resumo de 1-2 frases
- Forneça o código com comentários
- Termine com "Próximos passos" sugeridos"""
```

### 2.2 Prompt Templates

```python
from string import Template

# Template reutilizável
analysis_template = Template("""
Analise o seguinte dataset e forneça insights:

**Dataset:** $dataset_name
**Objetivo:** $objective  
**Colunas principais:** $columns
**Tamanho:** $n_rows linhas × $n_cols colunas

Por favor:
1. Identifique possíveis problemas nos dados
2. Sugira features para criar
3. Recomende 3 modelos para testar
4. Proponha métricas de avaliação

Responda de forma estruturada.
""")

prompt = analysis_template.substitute(
    dataset_name="Credit Card Fraud",
    objective="Detectar transações fraudulentas",
    columns="amount, time, v1-v28, class",
    n_rows="284,807",
    n_cols="31"
)
```

---

## 3. RAG — Retrieval Augmented Generation

### 3.1 Arquitetura Completa

```
┌─────────────────────────────────────────────────────┐
│                  RAG Pipeline                         │
│                                                       │
│  ┌─────────┐    ┌──────────┐    ┌──────────────┐    │
│  │Documents│───→│ Chunking │───→│  Embeddings  │    │
│  └─────────┘    └──────────┘    └──────┬───────┘    │
│                                         │            │
│                                         ▼            │
│                                  ┌──────────────┐    │
│                                  │ Vector Store │    │
│                                  └──────┬───────┘    │
│                                         │            │
│  ┌─────────┐    ┌──────────┐           │            │
│  │  Query  │───→│ Embed    │───→ Search ┘            │
│  └─────────┘    │  Query   │                         │
│                  └──────────┘                         │
│                       │                               │
│                       ▼                               │
│              ┌─────────────────┐                      │
│              │  Retrieved Docs │                      │
│              └────────┬────────┘                      │
│                       │                               │
│              ┌────────▼────────┐                      │
│              │  LLM + Context  │──→ Response          │
│              └─────────────────┘                      │
└─────────────────────────────────────────────────────┘
```

### 3.2 Implementação Completa

```python
# ---- 1. Document Loading ----
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader,
    WebBaseLoader, CSVLoader
)

# PDF
loader = PyPDFLoader("documento.pdf")
docs = loader.load()

# Web
loader = WebBaseLoader("https://example.com/artigo")
docs = loader.load()

# ---- 2. Text Chunking ----
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # tamanho do chunk em caracteres
    chunk_overlap=200,        # overlap entre chunks
    separators=["\n\n", "\n", ". ", " ", ""],  # ordem de prioridade
    length_function=len
)

chunks = splitter.split_documents(docs)
print(f"Documentos: {len(docs)} → Chunks: {len(chunks)}")

# ---- 3. Embeddings ----
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Testar embedding
embedding = embeddings.embed_query("O que é machine learning?")
print(f"Dimensão do embedding: {len(embedding)}")  # 1536

# ---- 4. Vector Store ----
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Buscar documentos similares
results = vectorstore.similarity_search("Como funciona gradient descent?", k=5)
for doc in results:
    print(f"Score: {doc.metadata.get('score', 'N/A')}")
    print(f"Content: {doc.page_content[:200]}...")
    print("---")

# ---- 5. RAG Chain ----
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag_prompt = PromptTemplate(
    template="""Use o contexto abaixo para responder a pergunta. 
Se não souber a resposta baseado no contexto, diga "Não encontrei essa informação nos documentos".

Contexto:
{context}

Pergunta: {question}

Resposta:""",
    input_variables=["context", "question"]
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True
)

# Usar
result = rag_chain({"query": "Como funciona gradient descent?"})
print(result["result"])
print(f"\nFontes: {len(result['source_documents'])} documentos")
```

### 3.3 Estratégias Avançadas de RAG

```python
# ---- Hybrid Search (Dense + Sparse) ----
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 (sparse/keyword)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Dense (semantic)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Hybrid
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]  # peso para cada retriever
)

# ---- Reranking ----
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

reranker = CohereRerank(model="rerank-v3.5", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever
)

# ---- Multi-Query Retrieval ----
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
# Gera múltiplas versões da query para melhor retrieval
```

---

## 4. AI Agents

### 4.1 Conceito de Agents

```
Agent = LLM + Tools + Memory + Planning

Loop:
1. Recebe tarefa
2. Planeja próximo passo
3. Escolhe e usa uma Tool
4. Observa resultado
5. Decide se precisa mais passos
6. Retorna resposta final
```

### 4.2 Function Calling / Tool Use

```python
from openai import OpenAI
import json

client = OpenAI()

# Definir tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "buscar_dados_vendas",
            "description": "Busca dados de vendas por período e produto",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_inicio": {"type": "string", "description": "Data de início (YYYY-MM-DD)"},
                    "data_fim": {"type": "string", "description": "Data de fim (YYYY-MM-DD)"},
                    "produto": {"type": "string", "description": "Nome do produto (opcional)"}
                },
                "required": ["data_inicio", "data_fim"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_metricas",
            "description": "Calcula métricas estatísticas sobre dados numéricos",
            "parameters": {
                "type": "object",
                "properties": {
                    "dados": {"type": "array", "items": {"type": "number"}},
                    "metricas": {
                        "type": "array", 
                        "items": {"type": "string", "enum": ["media", "mediana", "desvio", "total"]}
                    }
                },
                "required": ["dados", "metricas"]
            }
        }
    }
]

# Implementação das tools
def buscar_dados_vendas(data_inicio, data_fim, produto=None):
    # Simula busca no banco de dados
    return {"vendas": [100, 150, 200, 180, 220], "periodo": f"{data_inicio} a {data_fim}"}

def calcular_metricas(dados, metricas):
    import numpy as np
    arr = np.array(dados)
    result = {}
    if "media" in metricas: result["media"] = float(arr.mean())
    if "mediana" in metricas: result["mediana"] = float(np.median(arr))
    if "desvio" in metricas: result["desvio"] = float(arr.std())
    if "total" in metricas: result["total"] = float(arr.sum())
    return result

# Agent loop
def run_agent(user_message):
    messages = [
        {"role": "system", "content": "Você é um analista de dados. Use as ferramentas disponíveis para responder perguntas."},
        {"role": "user", "content": user_message}
    ]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # Se não chamou nenhuma tool, é a resposta final
        if not message.tool_calls:
            return message.content
        
        # Executar tools chamadas
        messages.append(message)
        
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # Executar a função
            if func_name == "buscar_dados_vendas":
                result = buscar_dados_vendas(**args)
            elif func_name == "calcular_metricas":
                result = calcular_metricas(**args)
            
            # Adicionar resultado ao contexto
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

# Usar
resposta = run_agent("Qual foi a média de vendas em janeiro de 2024?")
print(resposta)
```

### 4.3 Multi-Agent Systems

```python
# Exemplo conceitual com CrewAI
from crewai import Agent, Task, Crew

# Definir agentes especializados
researcher = Agent(
    role="Data Researcher",
    goal="Encontrar e analisar dados relevantes",
    backstory="Expert em busca e análise de dados",
    tools=[search_tool, database_tool]
)

analyst = Agent(
    role="Data Analyst",
    goal="Analisar dados e gerar insights",
    backstory="Expert em análise estatística e ML",
    tools=[python_tool, viz_tool]
)

writer = Agent(
    role="Report Writer",
    goal="Criar relatórios claros e acionáveis",
    backstory="Expert em comunicação de dados",
    tools=[writing_tool]
)

# Definir tarefas
task1 = Task(
    description="Pesquise dados de vendas do último trimestre",
    agent=researcher
)

task2 = Task(
    description="Analise os dados e identifique tendências",
    agent=analyst
)

task3 = Task(
    description="Crie um relatório executivo com os insights",
    agent=writer
)

# Criar crew e executar
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[task1, task2, task3],
    verbose=True
)

result = crew.kickoff()
```

---

## 5. Fine-tuning de LLMs

### Quando fazer Fine-tuning?

```
Abordagem por custo-benefício:

1. Prompt Engineering (mais barato, mais rápido)
   └── Resolve 80% dos casos

2. Few-shot + RAG
   └── Quando precisa de conhecimento específico

3. Fine-tuning (mais caro, mais trabalho)
   └── Quando precisa de estilo/formato específico
   └── Quando precisa de performance consistente
   └── Quando quer reduzir custo de inferência (modelo menor)
```

### LoRA / QLoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# Carregar modelo base
model_name = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True  # QLoRA — quantização 4-bit
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configurar LoRA
lora_config = LoraConfig(
    r=16,                    # rank (menor = menos params, maior = mais expressivo)
    lora_alpha=32,           # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Aplicar LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 8,030,261,248 || trainable%: 0.0522%

# Dataset
dataset = load_dataset("json", data_files="training_data.jsonl")

# Training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=2048,
)

trainer.train()
model.save_pretrained("./fine-tuned-model")
```

---

## 6. RAG Evaluation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Preparar dados de avaliação
eval_data = {
    "question": ["O que é gradient descent?", "Como funciona BERT?"],
    "answer": ["Gradient descent é...", "BERT é um modelo que..."],
    "contexts": [["Chunk sobre gradient descent..."], ["Chunk sobre BERT..."]],
    "ground_truth": ["Gradient descent é um algoritmo...", "BERT usa masked language..."]
}

dataset = Dataset.from_dict(eval_data)

# Avaliar
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results)
# {
#   'faithfulness': 0.85,        # resposta fiel ao contexto?
#   'answer_relevancy': 0.90,    # resposta relevante à pergunta?
#   'context_precision': 0.80,   # contexto recuperado é preciso?
#   'context_recall': 0.75       # contexto cobre a resposta esperada?
# }
```

---

## 🏋️ Exercícios

1. **Construa um RAG** sobre a documentação de um framework (ex: PyTorch docs)
2. **Implemente um Agent** que pode executar código Python e buscar na web
3. **Compare** diferentes chunking strategies e seus efeitos no RAG
4. **Fine-tune** um modelo pequeno (ex: Llama 3.1 8B) com LoRA em um dataset custom
5. **Avalie** seu sistema RAG com RAGAS e itere para melhorar
6. **Crie um chatbot** com memória de conversa usando LangChain

---

## 📝 Notas

> Adicione aqui suas anotações pessoais conforme avança nos estudos.
