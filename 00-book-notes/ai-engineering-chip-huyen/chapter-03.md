# 📖 AI Engineering Book — Anotações de Leitura

> **Livro:** *AI Engineering* — Chip Huyen
> **Capítulo:** 3 — Evaluation Methodology

---

## Capítulo 3 — Evaluation Methodology

### Pontos-chave

- Quanto mais a IA é usada, maior a chance de **falhas catastróficas** (suicídio incentivado por chatbot, advogados submetendo provas alucinadas, Air Canada condenada por info falsa do chatbot)
- O **maior obstáculo** para colocar apps de IA no mundo real é a **avaliação** — pode consumir a maior parte do esforço de desenvolvimento
- Muita gente se contenta com "boca a boca" ou olhar os resultados no olho (*vibe check*) → cria risco e atrasa iteração
- Livro dedica **2 capítulos** à avaliação; este cobre métodos e limitações, o próximo cobre como montar o pipeline

### Por que avaliar foundation models é difícil

1. Quanto mais **inteligente** o modelo, mais difícil avaliá-lo (qualquer um vê erro de matemática de 1º ano, poucos veem erro de matemática de PhD)
2. Natureza **open-ended** quebra a abordagem tradicional de comparar com ground truth (há respostas corretas demais)
3. Modelos são **caixas-pretas** (providers não expõem detalhes)
4. Benchmarks **saturam rápido** (GLUE saturou em 1 ano → SuperGLUE; MMLU → MMLU-Pro)
5. Escopo expandiu: além de medir tarefas conhecidas, é preciso **descobrir novas capacidades**
- Investimento em avaliação ainda **fica atrás** do investimento em modelagem/treino

### Métricas de Language Modeling

- A performance do componente de language model correlaciona com a performance nas tarefas downstream
- **Entropia**: quanta informação, em média, um token carrega; menor entropia = mais previsível
- **Cross entropy**: quão difícil é para o modelo prever o próximo token no dataset; depende da entropia dos dados + divergência (KL) entre a distribuição aprendida (Q) e a real (P)
  - `H(P, Q) = H(P) + D_KL(P || Q)` — modelo perfeito → cross entropy = entropia dos dados
- **BPC** (bits-per-character) e **BPB** (bits-per-byte): variações que tornam a métrica comparável entre tokenizações diferentes; ligadas a **compressão** de texto
- **Perplexity (PPL)**: exponencial da cross entropy; mede a **incerteza** ao prever o próximo token (mais opções = maior perplexity)
  - ⚠️ unidade: *bit* (base 2) vs. *nat* (base e, usada em PyTorch/TF) → por isso muitos reportam perplexity em vez de cross entropy

#### Interpretação da perplexity

- Dados mais **estruturados** → menor perplexity (HTML é mais previsível que texto comum)
- **Vocabulário maior** → maior perplexity
- **Context length maior** → menor perplexity
- Usos: proxy da capacidade do modelo; detectar **contaminação de dados** (perplexity baixa em benchmark = benchmark provavelmente estava no treino); **deduplicação**; detectar texto anômalo
- ⚠️ Perplexity é proxy ruim para modelos **pós-treinados** (SFT/RLHF) — post-training "colapsa a entropia"

### Avaliação Exata

> Produz julgamento **sem ambiguidade** (≠ avaliação subjetiva, que depende de quem julga)

#### Functional Correctness

- Avalia se o sistema **faz o que deveria** (a métrica suprema)
- Automatizável para **código**: roda o código gerado contra **test cases** (execution accuracy)
- Benchmarks: HumanEval, MBPP, Spider, BIRD-SQL, WikiSQL
- Métrica **pass@k**: gera k amostras por problema; resolve se qualquer uma passa todos os testes; maior k = maior score

#### Similaridade com dados de referência

- Compara o output com **reference responses** (ground truths); gargalo é gerar dados de referência (humanos ou IA)
- 4 formas de medir similaridade:
  1. **Avaliador** (humano ou IA) julga se são iguais
  2. **Exact match**: output bate exatamente com a referência → só funciona para respostas curtas/exatas
  3. **Lexical similarity**: quanto os textos se **sobrepõem** (n-grams, edit distance/fuzzy matching); métricas BLEU, ROUGE, METEOR, TER, CIDEr
     - ⚠️ exige conjunto abrangente de referências; referências podem estar erradas; score alto ≠ resposta melhor (BLEU não otimiza correção funcional)
  4. **Semantic similarity**: quão próximos em **significado**; transforma texto em **embedding** e usa **cosine similarity**; métricas BERTScore, MoverScore
     - depende da qualidade do embedding; pode exigir compute não-trivial

#### Embeddings (intro)

- **Embedding** = vetor numérico que captura o **significado** dos dados (tamanho típico 100–10.000)
- Modelos: BERT, CLIP, Sentence Transformers, APIs (OpenAI, Cohere)
- Bom embedding = textos mais parecidos têm embeddings mais próximos
- **Joint/multimodal embedding space**: CLIP mapeia texto e imagem no mesmo espaço (permite busca de imagem por texto); ULIP, ImageBind

### AI as a Judge

- Usar IA para avaliar IA — um dos métodos **mais comuns** em produção (LangChain 2023: 58% das avaliações)
- **Vantagens**: rápido, barato, flexível (qualquer critério), funciona sem dados de referência, e **explica** a decisão
- Forte correlação com humanos em alguns casos (Zheng et al. 2023: GPT-4 ↔ humanos = 85%, maior que entre humanos = 81%)

#### Como usar

1. Avaliar a qualidade de uma resposta sozinha (nota 1–5)
2. Comparar resposta gerada vs. referência (True/False)
3. Comparar **duas respostas** e dizer qual é melhor (gera preference data)
- Critérios **não são padronizados** — "relevância" do Azure ≠ "relevância" do MLflow; dependem do modelo + prompt do juiz
- Prompt do juiz deve explicar: **tarefa**, **critérios**, **sistema de pontuação** (+ exemplos ajudam)
- Juízes funcionam melhor com **classificação** que com nota numérica; discreto > contínuo; faixa típica 1–5
- ⚠️ Um juiz = modelo **+ prompt** (+ sampling params); mudar qualquer um = juiz diferente

#### Limitações

- **Inconsistência**: probabilístico, mesmo prompt pode dar notas diferentes (incluir exemplos aumenta consistência de 65%→77.5%, mas encarece)
- **Ambiguidade de critérios**: mesma "faithfulness" tem prompts e escalas diferentes entre ferramentas → scores não comparáveis; **não confie em juiz cujo modelo+prompt você não vê**
- **Custo e latência**: usar GPT-4 para gerar E avaliar dobra/quadruplica as chamadas; mitigar com modelos mais fracos ou **spot-checking** (avaliar um subconjunto)
- **Vieses**: **self-bias** (favorece as próprias respostas), **first-position bias** (favorece a 1ª opção — humanos têm recency bias, o oposto), **verbosity bias** (favorece respostas mais longas mesmo com erros)

#### Que modelos podem ser juízes

- Juiz pode ser **mais forte, mais fraco ou igual** ao modelo avaliado
- **Mais forte**: melhores julgamentos, mas o modelo mais forte fica sem juiz; usado para avaliar subconjunto (custo/latência)
- **Self-evaluation/self-critique**: bom para sanity check e para o modelo revisar a própria resposta
- **Mais fraco**: julgar é mais fácil que gerar → direção promissora dos **juízes pequenos e especializados**:
  - **Reward model**: (prompt, response) → nota (ex: Cappy, 360M params)
  - **Reference-based judge**: compara com referência (ex: BLEURT, Prometheus)
  - **Preference model**: (prompt, resp 1, resp 2) → qual é melhor (ex: PandaLM, JudgeLM)

### Ranking com Avaliação Comparativa

- Muitas vezes você quer um **ranking** de modelos, não scores absolutos
- **Pointwise**: avalia cada modelo isolado e ranqueia por score
- **Comparative**: avalia modelos **uns contra os outros** (qual é melhor) → mais fácil para qualidade subjetiva
- Usada pela primeira vez em IA pela Anthropic (2021); alimenta o **LMSYS Chatbot Arena**
- Cada comparação = **match**; **win rate** de A sobre B = % de vitórias de A
- Algoritmos de rating (de esportes/games): **Elo**, **Bradley–Terry**, **TrueSkill** (Chatbot Arena trocou Elo→Bradley-Terry por sensibilidade à ordem)
- Ranking é um problema **preditivo**: prever resultados futuros de matches; não há ground truth do ranking "correto"
- ⚠️ Nem toda pergunta deve ser decidida por **preferência** — muitas devem ser por **correção** (preferência pode dar sinais errados); só funciona com votantes que **entendem** do assunto
- ≠ A/B testing (lá o usuário vê um modelo por vez; aqui vê vários ao mesmo tempo)

#### Desafios

- **Escalabilidade**: nº de pares cresce **quadraticamente**; mitigado por **transitividade** (A>B, B>C ⟹ A>C — mas pode não valer para preferência humana) e por melhores algoritmos de matching
- **Falta de padronização/controle de qualidade**: crowdsourcing (Chatbot Arena) capta sinais amplos e é difícil de fraudar, mas prompts são simples demais ("hello" = 0.55% dos prompts), votantes podem não fact-checkar; alternativas = avaliadores treinados (caro, ex: Scale) ou avaliar dentro do produto
- **Comparativo → absoluto**: dizer que B > A não diz **quão bom** B é nem se é **bom o suficiente**; 1% de win rate pode dar boost grande ou mínimo; dificulta análise custo-benefício

#### Futuro

- Comparar é mais fácil que dar nota (sobretudo quando modelos superam humanos)
- Captura o que importa (**preferência humana**) e **não satura** como benchmarks
- Difícil de fraudar → leaderboards comparativos são bem confiáveis
- Complementa benchmarks (offline) e A/B testing (online)

---

> *Resumo:* Capítulo cobriu por que avaliar foundation models é difícil, métricas de language modeling (entropia, cross entropy, perplexity), avaliação exata (functional correctness + similaridade), AI as a judge (subjetiva, com vieses) e avaliação comparativa (ranking via preferência). O próximo capítulo mostra como montar um pipeline de avaliação confiável.
