# Trabalho P4 — Implementação de Rede Neural Artificial

**Tema (Parte 1):** Detecção de Discurso de Ódio em redes sociais
**Framework:** TensorFlow / Keras 3 (TF 2.21)
**Arquitetura escolhida:** MLP (Multilayer Perceptron / Rede Neural Feedforward)
**Tarefa:** Classificação binária — tóxico vs. não-tóxico
**Dataset:** ToLD-BR (21.000 tweets em português, rotulados em 6 categorias de toxicidade)
**Script:** [`nn_keras.py`](nn_keras.py)
**Artefatos:** [`outputs_p4/`](outputs_p4/)

---

## 1. Por que MLP?

Na Parte 1 do trabalho descrevemos três famílias de redes neurais aplicáveis ao problema: MLP, RNN/LSTM e Transformers (BERT). A MLP foi escolhida por três motivos:

1. **Coerência com o pseudo-algoritmo apresentado na Parte 1** (Coleta → Pré-processamento → Vetorização TF-IDF → Rede com camadas de entrada/oculta/saída → Treinamento → Avaliação).
2. **Reaproveita a representação TF-IDF** já utilizada nos modelos clássicos do projeto (Naive Bayes, Regressão Logística, SVM Linear), permitindo comparação direta.
3. **Tarefa simples**, como pedido pelo enunciado — sem a complexidade extra de embeddings sequenciais ou atenção.

---

## 2. Pré-processamento

Aplicado em `nn_keras.py → preprocessar()`:

- Lowercase
- Remoção de URLs (`http...`, `www...`)
- Remoção de menções (`@usuario`)
- Remoção do token `rt` (retweet)
- Remoção de caracteres não-alfanuméricos (mantendo acentos `à-ú`)
- Colapso de múltiplos espaços

O rótulo binário `toxic` é definido como o **OR** das 6 categorias originais
(`homophobia`, `obscene`, `insult`, `racism`, `misogyny`, `xenophobia`): se qualquer uma for ≥ 1, a amostra é tóxica.

| Classe         | Amostras |
|----------------|---------:|
| Não-tóxico (0) |  11.745  |
| Tóxico (1)     |   9.255  |
| **Total**      |  21.000  |

Divisão estratificada **60/20/20** → 12.600 treino, 4.200 validação, 4.200 teste.

---

## 3. Parâmetros do modelo (itens i–v do enunciado)

### (i) Dimensão do vetor de entrada

**`input_dim = 5000`** — cada tweet é representado por um vetor TF-IDF de 5.000 dimensões.

```python
TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
```

O `TfidfVectorizer` seleciona os **5.000 termos mais frequentes** do corpus (incluindo unigramas e bigramas, com `min_df=2` para descartar termos raros), e cada documento é projetado nesse espaço.

### (ii) O que representa cada posição do vetor (features)

Cada uma das 5.000 posições corresponde a um **termo (palavra ou bigrama) do vocabulário** aprendido no corpus. O valor armazenado é o **TF-IDF** desse termo no documento:

```
TF-IDF(t, d) = TF(t, d) × log( N / DF(t) )
```

onde:
- `TF(t, d)` = frequência do termo `t` no documento `d`,
- `DF(t)` = número de documentos do corpus que contêm `t`,
- `N` = total de documentos.

Termos comuns em todo o corpus (alta `DF`) ganham peso baixo; termos discriminativos (frequentes em poucos documentos) ganham peso alto. Vetor é esparso por natureza — a maioria das posições é zero para cada tweet.

### (iii) Pesos, camadas e quantidade de neurônios

A rede tem **3 camadas Dense** (totalmente conectadas) intercaladas por **Dropout**:

| Camada       | Tipo          | Entrada → Saída | Parâmetros |
|--------------|---------------|-----------------|-----------:|
| `entrada_tfidf` | Input      | (5000,)         |          0 |
| `oculta_1`   | Dense + ReLU  | 5000 → 256      |  1.280.256 |
| `dropout_1`  | Dropout 0.4   | 256 → 256       |          0 |
| `oculta_2`   | Dense + ReLU  | 256 → 64        |     16.448 |
| `dropout_2`  | Dropout 0.3   | 64 → 64         |          0 |
| `saida`      | Dense + Sigmoid | 64 → 1        |         65 |
| **Total**    |               |                 | **1.296.769** |

Cada Dense aprende uma matriz **W** (pesos) de tamanho `(entradas × neurônios)` e um vetor **b** (bias) de tamanho `(neurônios)`:

- `oculta_1`: W₁ ∈ ℝ^(5000×256), b₁ ∈ ℝ^256 → 5000×256 + 256 = 1.280.256 parâmetros
- `oculta_2`: W₂ ∈ ℝ^(256×64),  b₂ ∈ ℝ^64  → 256×64 + 64 = 16.448 parâmetros
- `saida`:    W₃ ∈ ℝ^(64×1),    b₃ ∈ ℝ^1   → 64 + 1 = 65 parâmetros

Os pesos são inicializados pelo Keras com **Glorot Uniform** (default para Dense) e ajustados pelo otimizador **Adam** (`lr=1e-3`) minimizando **Binary Crossentropy** via backpropagation.

### (iv) Funções de ativação

| Camada     | Ativação | Justificativa |
|------------|----------|---------------|
| Camadas ocultas (`oculta_1`, `oculta_2`) | **ReLU** | `ReLU(x) = max(0, x)`. É a função padrão para camadas ocultas em MLPs modernas — barata de computar, evita o problema do gradiente esvanecente das funções saturantes (sigmoid/tanh) e induz esparsidade, o que casa bem com a natureza esparsa do TF-IDF. |
| Camada de saída (`saida`) | **Sigmoid** | `σ(x) = 1/(1+e⁻ˣ)`, que mapeia o logit para `[0, 1]`, interpretável como **probabilidade da classe positiva (tóxico)**. É a escolha canônica para classificação binária quando a perda é Binary Crossentropy. |

### (v) Número total de camadas e neurônios por camada

- **Camadas com pesos:** 3 (duas ocultas + uma de saída).
- **Camadas totais (contando Dropout):** 5.
- **Neurônios por camada:** entrada **5000** → oculta_1 **256** → oculta_2 **64** → saída **1**.
- **Total de parâmetros treináveis:** 1.296.769.

Adicionalmente:
- **Otimizador:** Adam, learning rate = 1e-3.
- **Perda:** Binary Crossentropy.
- **Métrica:** accuracy.
- **Batch size:** 128.
- **Épocas máximas:** 25 (com `EarlyStopping` em `val_loss`, paciência 4, `restore_best_weights=True`).
- **Class weights:** `{0: 0.894, 1: 1.135}` para compensar o leve desbalanceamento (mais não-tóxicos).
- **Regularização:** Dropout (0.4 e 0.3) nas camadas ocultas.

---

## 4. Visualização dos resultados

### (i) Evolução da perda e acurácia

![Curvas de treinamento](outputs_p4/treinamento_curvas.png)

| Época | loss (treino) | val_loss | accuracy (treino) | val_accuracy |
|------:|--------------:|---------:|------------------:|-------------:|
| 1     | 0,636         | **0,563** | 0,668             | **0,715**    |
| 2     | 0,465         | 0,578     | 0,783             | 0,727        |
| 3     | 0,363         | 0,654     | 0,844             | 0,718        |
| 4     | 0,274         | 0,760     | 0,892             | 0,711        |
| 5     | 0,201         | 0,892     | 0,924             | 0,710        |

Fonte: `outputs_p4/history.json`.

### (ii) Comportamento treino vs. validação

Os gráficos evidenciam um caso clássico de **overfitting**:

- A perda de treino **cai monotonicamente** (de 0,636 para 0,201), e a acurácia de treino **sobe** (66,8% → 92,4%).
- A perda de validação **atinge mínimo já na época 1** (0,563) e cresce em seguida, enquanto a acurácia de validação **estagna em ~71–72%**.
- O `EarlyStopping` (paciência 4) parou o treino na **época 5** e restaurou os pesos da época 1 (melhor `val_loss`).

Ou seja: 1.296.769 parâmetros são muito para 12.600 exemplos de treino — a rede memoriza rapidamente.

### Avaliação no teste (pesos da época 1)

| Métrica                | Valor    |
|------------------------|---------:|
| **Test loss**          | 0,5508   |
| **Test accuracy**      | **0,7269** |
| Precisão (não-tóxico)  | 0,7606   |
| Recall (não-tóxico)    | 0,7467   |
| F1 (não-tóxico)        | 0,7536   |
| Precisão (tóxico)      | 0,6859   |
| Recall (tóxico)        | 0,7018   |
| F1 (tóxico)            | 0,6937   |
| F1 macro               | 0,7237   |

#### Matriz de confusão (4.200 amostras de teste)

![Matriz de confusão](outputs_p4/matriz_confusao.png)

|                        | Pred. Não-tóxico | Pred. Tóxico |
|------------------------|-----------------:|-------------:|
| **Real Não-tóxico**    |             1754 |          595 |
| **Real Tóxico**        |              552 |         1299 |

- **Falsos positivos:** 595 (não-tóxicos classificados como tóxicos).
- **Falsos negativos:** 552 (tóxicos passando como não-tóxicos).

Fontes brutas: `outputs_p4/metricas_teste.json`, `outputs_p4/history.json`, `outputs_p4/model_summary.txt`.

---

## 5. Conclusão

### (i) A arquitetura foi adequada ao tipo de dado?

**Parcialmente.** A MLP funciona como **baseline neural** sobre TF-IDF, mas o experimento mostra duas limitações estruturais:

1. **TF-IDF é uma representação bag-of-words** — ignora a ordem das palavras. Isso é especialmente prejudicial para discurso de ódio, onde **negação, sarcasmo e construção sintática** alteram completamente o sentido (compare *"você não é burro"* vs. *"você é burro"*: mesma sacola de palavras, sentidos opostos).
2. **Capacidade vs. dados:** com ~1,3 M parâmetros para ~12 k exemplos, a MLP **decora** o treino e generaliza pouco — o `EarlyStopping` precisou intervir já na 5ª época. A acurácia de validação trava em ~72%, no mesmo patamar dos modelos clássicos do projeto (LogReg, SVM Linear). Ou seja: **a MLP sobre TF-IDF não traz ganho qualitativo em relação a um linear bem regularizado** — confirmando o ponto teórico de Goodfellow et al. (2016) de que modelos lineares já capturam quase tudo o que um bag-of-words tem a oferecer.

### (ii) Próximos passos para melhorar o modelo

| # | Direção | Por quê |
|---|---------|---------|
| 1 | **Trocar TF-IDF por Embeddings** (`Embedding` aprendido ou pré-treinado, e.g. fastText/Glove pt-br) | Preserva semântica e generaliza para palavras OOV (out-of-vocabulary). |
| 2 | **Adotar arquitetura sequencial (BiLSTM ou Conv1D)** | Captura ordem das palavras, negações e n-gramas locais — diretamente alinhado com o que a Parte 1 cita sobre LSTM/RNN. |
| 3 | **Fine-tuning de um Transformer pré-treinado** (`BERTimbau` ou `XLM-R`) | Estado-da-arte para PLN em português. Geralmente +10–15 pontos de F1 sobre baselines. |
| 4 | **Regularização e capacidade menor na MLP atual** (Dense menores, L2, mais Dropout) | Atacaria o overfitting direto, mas com teto baixo dado o limite do bag-of-words. |
| 5 | **Tratamento de desbalanceamento por categoria** (classificação multi-rótulo separada por tipo de ódio) | Hoje colapsamos 6 categorias em 1; recuperar o detalhe melhora a aplicação real e permite usar `class_weight` por categoria. |
| 6 | **Aumentar e diversificar o corpus** combinando ToLD-BR + HateBR + HateBRXplain | Mais dados endereçam diretamente o overfitting observado. |
| 7 | **Calibração de limiar** (threshold ≠ 0.5) | Otimizar o ponto de corte para o F1 da classe minoritária ou para o recall, dependendo do custo de falsos negativos. |
| 8 | **Análise de erros** — inspecionar os 552 falsos negativos | Provavelmente revelará casos de ironia / código ofensivo implícito, justificando o salto para modelos contextuais. |

A combinação **#2 + #6** já daria um salto significativo dentro do escopo de "redes neurais clássicas"; **#3** (Transformer) é o caminho natural se o grupo quiser explorar deep learning estado-da-arte para o próximo trabalho.

---

## 6. Como reproduzir

```bash
# (na pasta do projeto, com Python 3.10–3.12)
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
python nn_keras.py
```

Saídas geradas em `outputs_p4/`:
- `treinamento_curvas.png` — loss e accuracy por época (treino vs. validação).
- `matriz_confusao.png` — matriz de confusão no conjunto de teste.
- `history.json` — histórico bruto do treinamento.
- `metricas_teste.json` — métricas finais de teste e classification report.
- `model_summary.txt` — `model.summary()` do Keras.

Modelo treinado e vectorizer salvos em `models/`:
- `mlp_tfidf.keras`
- `tfidf_vectorizer_mlp.pkl`
