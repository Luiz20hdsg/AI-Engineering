# 📖 AI Engineering Book — Anotações de Leitura

> **Livro:** *AI Engineering* — Chip Huyen  
> **Início:** 16 de março de 2026

---

## Capítulo 1 — Introduction to Building AI Applications with Foundation Models

### Pontos-chave

- **Escala** é a palavra que define AI pós-2020 — modelos tão grandes que consomem parcela significativa da eletricidade mundial
- Escala trouxe 2 consequências: (1) modelos mais poderosos com mais aplicações, (2) só poucas orgs conseguem treinar → surgiu **model as a service**
- Demanda por AI apps **subiu**, barreira de entrada **caiu** → nasceu **AI Engineering** como disciplina
- **AI Engineering** = construir aplicações em cima de modelos já prontos (não treinar do zero)
- Princípios de ML em produção continuam os mesmos, mas foundation models trazem possibilidades e desafios novos

### Language Models

- **Language model** = codifica informação estatística sobre uma ou mais línguas (qual palavra é mais provável em dado contexto)
- Conceito vem de longe: Shannon (1951) já modelava inglês com estatística — conceito de **entropia** ainda é usado hoje
- Unidade básica = **token** (pode ser caractere, palavra ou parte de palavra, ex: "can't" → `can` + `'t`)
- **Tokenização** = processo de quebrar texto em tokens; no GPT-4, ~100 tokens ≈ 75 palavras
- **Vocabulário** = conjunto de todos os tokens que o modelo conhece (Mixtral: 32K, GPT-4: ~100K)
- Tokens > palavras porque: (1) capturam partes com significado (cook+ing), (2) vocabulário menor = mais eficiente, (3) lidam com palavras desconhecidas

### Dois tipos de Language Model

- **Masked LM (ex: BERT)** — preenche o buraco usando contexto de antes E depois → bom para classificação, sentiment analysis, debugging de código
- **Autoregressive LM (ex: GPT)** — prediz o **próximo token** usando apenas tokens anteriores → modelo dominante para geração de texto
- No livro, "language model" = autoregressive por padrão

### Self-supervision

- LLMs só conseguiram escalar graças a **self-supervision** — aprendem a partir dos próprios dados sem precisar de labels manuais

---

> *Parei em:* Fim da seção "From Language Models to Large Language Models" (Figura 1-2)
