# 📖 AI Engineering Book — Anotações de Leitura

> **Livro:** *AI Engineering* — Chip Huyen
> **Capítulo:** 2 — Understanding Foundation Models

---

## Capítulo 2 — Understanding Foundation Models

### Pontos-chave

- Não precisa saber treinar um modelo para usá-lo, mas entender o **alto nível** ajuda a escolher e adaptar o modelo
- Diferenças entre foundation models vêm de 3 decisões: **dados de treino**, **arquitetura + tamanho**, e **post-training** (alinhamento com preferências humanas)
- Treino dividido em **pre-training** (torna o modelo capaz) e **post-training** (torna seguro e fácil de usar)
- **Sampling** é subestimado: explica alucinações e inconsistências, e a estratégia certa melhora bastante o desempenho

### Training Data

- "Um modelo é tão bom quanto os dados em que foi treinado" — sem vietnamita no treino, não traduz para vietnamita
- **Common Crawl** = fonte comum (2–3 bilhões de páginas/mês), mas qualidade duvidosa (clickbait, fake news); C4 é um subconjunto "limpo" do Google
- Muitas empresas **pararam de divulgar** as fontes de treino para evitar escrutínio e concorrência
- Mais dados nem sempre = melhor: pouca data de **alta qualidade** pode superar muita data de baixa qualidade

#### Modelos multilíngues

- Inglês domina a internet (~46% do Common Crawl, 8× mais que o russo)
- Línguas **low-resource** (pouco representadas) → modelos têm pior desempenho (ex: GPT-4 muito melhor em inglês que em telugu/birmanês no MMLU)
- Traduzir tudo para inglês e voltar **não é ideal**: precisa de modelo que entenda a língua e há perda de informação
- Tokenização é **menos eficiente** em algumas línguas → birmanês usa ~10× mais tokens que inglês → mais lento e caro

#### Modelos por domínio

- Modelos gerais (GPT, Gemini, Llama) vão bem em muitos domínios, mas mal em **tarefas especializadas** nunca vistas no treino
- Exemplos: AlphaFold (proteínas), BioNeMo (biomolecular), Med-PaLM2 (médico) — exigem datasets curados específicos

### Modeling — Arquitetura

- **Transformer** (Vaswani et al., 2017) domina, baseado no **attention mechanism**
- Veio resolver limitações do **seq2seq** (RNN encoder-decoder): (1) decoder usava só o estado final do input, (2) processamento sequencial lento
- Attention permite ao modelo **pesar a importância** de cada token de input ao gerar cada output
- Inferência em 2 passos: **prefill** (processa input em paralelo) e **decode** (gera 1 token por vez, sequencial)
- Attention usa 3 vetores: **Query (Q)** = estado atual, **Key (K)** = "número da página", **Value (V)** = "conteúdo da página"
- Quanto maior a sequência, mais K/V para computar e guardar → por isso é difícil estender o **context length**
- Attention é quase sempre **multi-head** (múltiplas cabeças olham grupos diferentes de tokens em paralelo)
- **Bloco transformer** = módulo de attention (4 matrizes: Q, K, V, output projection) + módulo MLP (camadas lineares + ativação não-linear como ReLU/GELU)
- Funções de ativação simples funcionam melhor — basta quebrar a linearidade; as sofisticadas gastam compute à toa

#### Outras arquiteturas

- Transformer é "pegajoso" (desde 2017); difícil superá-lo porque já foi muito otimizado
- Alternativas ganhando tração: **RWKV** (RNN paralelizável), **SSMs/state space models** (bom em sequências longas): S4, H3, **Mamba** (escala linear vs. quadrática), **Jamba** (híbrido transformer+Mamba)

### Model Size

- Mais parâmetros = mais capacidade de aprender (em geral)
- Modelos de geração mais nova superam os antigos do mesmo tamanho (ex: Llama 3-8B > Llama 2-70B no MMLU)
- **Sparse models** (muitos parâmetros zero) e **Mixture-of-Experts (MoE)** = só um subconjunto de "experts" ativo por token (ex: Mixtral 8x7B = 46.7B params mas só 12.9B ativos por token)
- Tamanho do dataset medido melhor em **número de tokens** (não amostras); LLMs atuais treinam com trilhões de tokens (Llama 3 = 15T)
- 3 números sinalizam a escala: **nº de parâmetros** (capacidade), **nº de tokens de treino** (quanto aprendeu), **nº de FLOPs** (custo de treino)
- ⚠️ FLOPs (operações) ≠ FLOP/s (operações por segundo)

#### Scaling laws

- **Chinchilla scaling law** (DeepMind, 2022): para treino compute-ótimo, nº de tokens ≈ **20× o tamanho do modelo**
- Modelo e dataset devem escalar igualmente (dobrar um = dobrar o outro)
- Qualidade não é tudo: Llama escolheu modelos menores (subótimos) por serem mais fáceis e baratos → maior adoção
- **Gargalos do scaling**: (1) vamos ficar sem **dados** da internet, (2) **eletricidade** (data centers podem chegar a 4–20% da energia global até 2030)
- Custo para a mesma performance cai com o tempo, mas custo para **melhorar** performance continua alto (last mile)

### Post-Training

- Pre-trained model tem 2 problemas: otimizado para **completar texto** (não conversar) e pode gerar saídas ruins/ofensivas
- 2 passos: (1) **Supervised Finetuning (SFT)** com dados de instrução de qualidade, (2) **Preference Finetuning** para alinhar com preferência humana
- Analogia: pre-training = ler para adquirir conhecimento; post-training = aprender a usar esse conhecimento
- Post-training usa pouco compute (InstructGPT: 2% post-training, 98% pre-training)

#### SFT

- Usa **demonstration data** = pares (prompt, response); o modelo clona o comportamento desejado
- Labelers de alta qualidade são caros (InstructGPT: ~90% com diploma; 1 par pode levar 30 min)

#### Preference Finetuning (RLHF)

- Objetivo: fazer o modelo se comportar segundo preferência humana (meta ambiciosa — preferência universal talvez não exista)
- **RLHF** = (1) treinar um **reward model** que dá nota às respostas, (2) otimizar o modelo para maximizar essa nota (geralmente com **PPO**)
- Reward model é treinado com **comparison data** (prompt, resposta_vencedora, resposta_perdedora) — comparar é mais fácil/confiável que dar nota absoluta
- Alternativas mais novas: **DPO** (mais simples, usado no Llama 3), **RLAIF**
- Alguns pulam o RL e usam só o reward model (estratégia **best of N**)

### Sampling

- Modelo gera output **amostrando** de uma distribuição de probabilidades → IA é **probabilística**
- Logits → softmax → probabilidades; **logprobs** (log das probs) evitam underflow

#### Estratégias

- **Greedy** (sempre o mais provável) → bom para classificação, mas chato para geração de texto
- **Temperature**: divide os logits; maior temp → mais criativo (menos coerente), menor → mais consistente (mais chato); ~0.7 recomendado para criatividade
- **Top-k**: softmax só nos k maiores logits (reduz computação)
- **Top-p (nucleus)**: considera os tokens até a prob acumulada atingir p (ex: 0.9–0.95); seleção dinâmica
- **Stopping condition**: nº fixo de tokens ou stop tokens (controla latência/custo)

#### Test Time Compute

- Gerar **múltiplas respostas** e escolher a melhor (best of N, beam search) aumenta a chance de boa resposta
- Selecionar: maior logprob médio, reward model/verifier, ou heurística da aplicação
- OpenAI: verifier deu boost equivalente a aumentar o modelo **30×**
- Aumentar amostras melhora performance, mas com limite (OpenAI: até ~400; depois cai)
- Pegar a resposta **mais frequente** funciona bem para tarefas com resposta exata (self-consistency)

#### Structured Outputs

- Necessário quando: (1) a tarefa exige formato estruturado (ex: text-to-SQL, semantic parsing), (2) o output alimenta outra aplicação (ex: JSON)
- Abordagens (do "band-aid" ao "tratamento intensivo"): **prompting**, **post-processing**, **test time compute**, **constrained sampling** (filtra logits inválidos via gramática), **finetuning** (mais eficaz e geral)

### A Natureza Probabilística da IA

- Mesma pergunta → respostas diferentes (oposto de determinístico)
- Causa **inconsistência** (saídas diferentes para mesmo/parecido input) e **alucinação** (resposta sem base factual)
- Ótimo para tarefas **criativas**, problemático para o resto

#### Inconsistência

- Mitigar: cache da resposta, fixar variáveis de sampling (temperature, top-p, top-k, seed) — mas sem garantia 100% (hardware também influencia)

#### Alucinação

- 2 hipóteses: (1) **self-delusion** — o modelo não distingue o que recebeu do que gerou, e "bola de neve" a partir de uma suposição errada (snowballing); (2) **mismatch de conhecimento** — SFT ensina o modelo a imitar respostas que usam conhecimento que o labeler tem mas o modelo não → ensina a alucinar
- Mitigações: RL que diferencia input de output, prompts ("se não souber, diga não sei"), respostas concisas, verificação de fontes

---

> *Resumo:* Capítulo cobriu as decisões de design de um foundation model — dados de treino, arquitetura (transformer), tamanho (scaling laws), post-training (SFT + RLHF) e sampling (a fonte da natureza probabilística da IA).
