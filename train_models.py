import pandas as pd
import spacy
import joblib
import os
import re
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ================== CONFIGURAÇÃO (caminhos automáticos) ==================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODELS_DIR.mkdir(exist_ok=True)

categorias = ['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia']

# Carregar spaCy
print("Carregando spaCy (pt_core_news_sm)...")
nlp = spacy.load("pt_core_news_sm")

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_RT_RE = re.compile(r"\brt\b", flags=re.IGNORECASE)

def preprocessar_texto(texto):
    if not isinstance(texto, str) or not texto.strip():
        return ""
    t = texto.lower().strip()
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _RT_RE.sub(" ", t)
    doc = nlp(t)
    tokens = [token.lemma_ for token in doc
              if not token.is_stop and not token.is_punct and not token.is_space
              and len(token.lemma_) > 2]
    return " ".join(tokens)

# ================== CARREGAR DADOS (MongoDB) ==================
from db_manager import HateSpeechDB

print("Conectando ao MongoDB e carregando dados...")
db = HateSpeechDB()
df_train = db.get_split("train", source="ToLD-BR")
df_test  = db.get_split("test",  source="ToLD-BR")
df = pd.concat([df_train, df_test], ignore_index=True)
print(f"Dataset carregado do MongoDB com {len(df)} exemplos.")

# Converter rótulos para binário (0 = não tóxico, 1 = tóxico)
for cat in categorias:
    df[cat] = (df[cat] >= 1).astype(int)

print(f"Dataset carregado com {len(df)} exemplos.")

print("Pré-processando os textos com spaCy...")
df['text_clean'] = df['text'].apply(preprocessar_texto)

# ================== TREINAMENTO ==================
print("\nIniciando treinamento dos modelos...\n")

metrics_all = {}

for categoria in categorias:
    print(f"Treinando modelos para: **{categoria.upper()}**")

    X = df['text_clean']
    y = df[categoria]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    modelos = {
        'Naive Bayes': ComplementNB(),  # melhor para classes desbalanceadas
        'Regressão Logística': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'SVM Linear': LinearSVC(class_weight='balanced', dual=False, max_iter=2000)
    }

    metrics_cat = {}

    for nome, clf in modelos.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)),
            ('clf', clf)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        print(f"   ✓ {nome:20} → Acc: {acc:.4f} | F1-macro: {f1_macro:.4f} | F1(tóxico): {f1_pos:.4f}")

        filename = f"{nome.lower().replace(' ', '_')}_{categoria}.pkl"
        joblib.dump(pipeline, MODELS_DIR / filename)

        metrics_cat[nome] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_toxico": f1_pos,
            "precision_toxico": report.get("1", {}).get("precision", 0.0),
            "recall_toxico": report.get("1", {}).get("recall", 0.0),
            "confusion_matrix": cm,
            "classification_report": report,
        }

    metrics_all[categoria] = metrics_cat
    print(f"   ✅ Concluído: {categoria}\n")

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_all, f, ensure_ascii=False, indent=2)

print(f"🎉 Modelos salvos em '{MODELS_DIR}'")
print(f"📊 Métricas salvas em '{METRICS_PATH}'")
print("Agora você pode rodar o app Streamlit: streamlit run app.py")
