"""
Trabalho P4 - Rede Neural Artificial em TensorFlow/Keras
=========================================================
MLP (Multilayer Perceptron) para detecção binária de discurso de ódio
em português, treinada sobre o dataset ToLD-BR com features TF-IDF.

Saídas geradas em outputs_p4/:
  - treinamento_curvas.png   (loss e accuracy por época, treino vs validação)
  - matriz_confusao.png      (avaliação no conjunto de teste)
  - history.json             (histórico bruto do treinamento)
  - metricas_teste.json      (loss, accuracy, classification report, etc.)
  - model_summary.txt        (model.summary() do Keras)
"""

import os
# Reduz ruído de logs do TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import io
import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # backend sem display (roda em script)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

# =================== CONFIGURAÇÃO ===================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Bases_de_dados" / "ToLD-BR.csv"
OUT_DIR = BASE_DIR / "outputs_p4"
MODELS_DIR = BASE_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CATEGORIAS = ["homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]

MAX_FEATURES = 5000          # tamanho do vocabulário TF-IDF (dimensão do vetor de entrada)
NGRAM_RANGE = (1, 2)         # unigramas + bigramas
EPOCHS = 25
BATCH_SIZE = 128
LR = 1e-3

# =================== PRÉ-PROCESSAMENTO ===================
_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_RT_RE = re.compile(r"\brt\b", flags=re.IGNORECASE)
_NONWORD_RE = re.compile(r"[^a-zà-ú0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def preprocessar(texto: str) -> str:
    """Lowercase + remove URLs/menções/RT/pontuação. Mantém leve por opção
    didática (a MLP aprende padrões lexicais; não usamos lematização aqui
    para evitar a dependência pesada do spaCy nesse script)."""
    if not isinstance(texto, str):
        return ""
    t = texto.lower().strip()
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _RT_RE.sub(" ", t)
    t = _NONWORD_RE.sub(" ", t)
    t = _MULTISPACE_RE.sub(" ", t).strip()
    return t


# =================== CARGA DE DADOS ===================
print(f"[1/6] Carregando dataset: {DATA_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)
df = pd.read_csv(DATA_PATH)
print(f"      Total de amostras: {len(df)}")

# Rótulo binário: "tóxico" se QUALQUER categoria for >= 1
df["toxic"] = (df[CATEGORIAS] >= 1).any(axis=1).astype(int)
df["text_clean"] = df["text"].apply(preprocessar)

n_pos = int((df["toxic"] == 1).sum())
n_neg = int((df["toxic"] == 0).sum())
print(f"      Distribuição -> não-tóxico: {n_neg} | tóxico: {n_pos}")

# =================== VETORIZAÇÃO TF-IDF ===================
print(f"[2/6] Vetorizando com TF-IDF (max_features={MAX_FEATURES}, ngram={NGRAM_RANGE})...")
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    min_df=2,
)
X = vectorizer.fit_transform(df["text_clean"]).toarray().astype(np.float32)
y = df["toxic"].values.astype(np.float32)
print(f"      Shape X: {X.shape}  (amostras, features)")

# =================== SPLITS ===================
print("[3/6] Dividindo treino/validação/teste (60/20/20 estratificado)...")
X_tmp, X_te, y_tmp, y_te = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=SEED, stratify=y_tmp
)  # 0.25 * 0.8 = 0.2
print(f"      Treino: {X_tr.shape[0]} | Validação: {X_va.shape[0]} | Teste: {X_te.shape[0]}")

# =================== MODELO MLP ===================
INPUT_DIM = X.shape[1]
print(f"[4/6] Construindo MLP (input_dim={INPUT_DIM})...")

model = models.Sequential(
    [
        layers.Input(shape=(INPUT_DIM,), name="entrada_tfidf"),
        layers.Dense(256, activation="relu", name="oculta_1"),
        layers.Dropout(0.4, name="dropout_1"),
        layers.Dense(64, activation="relu", name="oculta_2"),
        layers.Dropout(0.3, name="dropout_2"),
        layers.Dense(1, activation="sigmoid", name="saida"),
    ],
    name="MLP_Discurso_Odio",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Salvar resumo do modelo num arquivo texto também
buf = io.StringIO()
model.summary(print_fn=lambda s: buf.write(s + "\n"))
summary_str = buf.getvalue()
print(summary_str)
(OUT_DIR / "model_summary.txt").write_text(summary_str, encoding="utf-8")

# =================== TREINAMENTO ===================
# Pesos por classe para compensar desbalanceamento
total_tr = len(y_tr)
n_pos_tr = float((y_tr == 1).sum())
n_neg_tr = float((y_tr == 0).sum())
class_weight = {
    0: total_tr / (2.0 * n_neg_tr),
    1: total_tr / (2.0 * n_pos_tr),
}
print(f"[5/6] class_weight = {class_weight}")

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
)

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=2,
)

# =================== AVALIAÇÃO ===================
print("[6/6] Avaliando no conjunto de teste...")
test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
print(f"      test_loss={test_loss:.4f}  |  test_accuracy={test_acc:.4f}")

y_proba = model.predict(X_te, verbose=0).ravel()
y_pred = (y_proba >= 0.5).astype(int)

report_str = classification_report(
    y_te, y_pred, target_names=["nao_toxico", "toxico"], digits=4
)
report_dict = classification_report(
    y_te, y_pred, target_names=["nao_toxico", "toxico"], digits=4, output_dict=True
)
cm = confusion_matrix(y_te, y_pred)
print("\n" + report_str)
print(f"Matriz de confusão:\n{cm}")

# =================== GRÁFICOS ===================
h = history.history
epochs_x = range(1, len(h["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(epochs_x, h["loss"], "o-", label="Treino")
axes[0].plot(epochs_x, h["val_loss"], "s-", label="Validação")
axes[0].set_title("Evolução da Perda (Binary Crossentropy)")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_x, h["accuracy"], "o-", label="Treino")
axes[1].plot(epochs_x, h["val_accuracy"], "s-", label="Validação")
axes[1].set_title("Evolução da Acurácia")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Acurácia")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
curvas_path = OUT_DIR / "treinamento_curvas.png"
plt.savefig(curvas_path, dpi=120)
plt.close(fig)
print(f"      Gráfico de curvas salvo em: {curvas_path}")

# Matriz de confusão
fig2, ax = plt.subplots(figsize=(5, 4.3))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["não-tóxico", "tóxico"])
ax.set_yticklabels(["não-tóxico", "tóxico"])
ax.set_xlabel("Predito")
ax.set_ylabel("Real")
ax.set_title("Matriz de Confusão (Teste)")
vmax = cm.max()
for i in range(2):
    for j in range(2):
        ax.text(
            j, i, int(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > vmax / 2 else "black",
            fontsize=12,
        )
plt.colorbar(im, ax=ax)
plt.tight_layout()
cm_path = OUT_DIR / "matriz_confusao.png"
plt.savefig(cm_path, dpi=120)
plt.close(fig2)
print(f"      Matriz de confusão salva em: {cm_path}")

# =================== PERSISTÊNCIA ===================
model_path = MODELS_DIR / "mlp_tfidf.keras"
vec_path = MODELS_DIR / "tfidf_vectorizer_mlp.pkl"
model.save(model_path)
joblib.dump(vectorizer, vec_path)
print(f"      Modelo salvo em: {model_path}")
print(f"      Vectorizer salvo em: {vec_path}")

# Histórico e métricas em JSON
hist_serializable = {k: [float(v) for v in vals] for k, vals in h.items()}
(OUT_DIR / "history.json").write_text(
    json.dumps(hist_serializable, indent=2), encoding="utf-8"
)

metricas = {
    "input_dim": int(INPUT_DIM),
    "n_train": int(len(y_tr)),
    "n_val": int(len(y_va)),
    "n_test": int(len(y_te)),
    "class_weight": class_weight,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "confusion_matrix": cm.tolist(),
    "classification_report": report_dict,
    "epochs_run": len(h["loss"]),
    "early_stop_epoch": int(early_stop.stopped_epoch) if early_stop.stopped_epoch else None,
}
(OUT_DIR / "metricas_teste.json").write_text(
    json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
)

# Contagem total de parâmetros treináveis (útil pro relatório)
trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
print(f"\nTotal de parâmetros treináveis: {trainable_params:,}")

print("\n[OK] Concluido. Veja a pasta outputs_p4/ para os artefatos do relatorio.")
