import streamlit as st
import pandas as pd
import numpy as np
import joblib
import spacy
import re
import json
import unicodedata
from pathlib import Path

# ================== CAMINHOS AUTOMÁTICOS ==================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Configuração da página
st.set_page_config(
    page_title="Detecção de Discurso de Ódio + Ofensa",
    page_icon="🛡️",
    layout="wide"
)

# ================== CARREGAMENTO SPAcy ==================
@st.cache_resource
def carregar_spacy():
    import spacy
    try:
        return spacy.load("pt_core_news_sm")
    except Exception:
        import pt_core_news_sm
        return pt_core_news_sm.load()

nlp = carregar_spacy()

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_RT_RE = re.compile(r"\brt\b", flags=re.IGNORECASE)

def preprocessar_texto(texto):
    if not texto or not isinstance(texto, str):
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

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def _normalize(s: str) -> str:
    return _strip_accents(s.lower())

# ================== DETECTOR DE LINGUAGEM OFENSIVA ==================
# { termo_normalizado: peso }
PALAVRAS_OFENSIVAS = {
    # Palavrões fortes
    "porra": 2, "caralho": 3, "foda": 2, "foder": 3, "fodido": 3, "fodendo": 3,
    "merda": 2, "bosta": 2, "cu": 2, "arrombado": 3, "arrombada": 3,
    "puta": 3, "puto": 2, "putaria": 2, "cacete": 2, "pica": 1, "rola": 1,
    "boquete": 2, "punheta": 2, "piranha": 2, "vagabunda": 3, "vagabundo": 3,
    "buceta": 3,
    # Xingamentos
    "idiota": 2, "imbecil": 2, "babaca": 2, "otario": 2, "otaria": 2,
    "burro": 1, "desgracado": 2, "safado": 2, "canalha": 2,
    "lixo": 1, "inutil": 1, "verme": 2, "corno": 2, "corna": 2,
}

EXPRESSOES_OFENSIVAS = {
    "filho da puta": 5, "filha da puta": 5, "vai tomar no cu": 5,
    "vai se foder": 5, "vai se fuder": 5, "puta que pariu": 4,
    "pau no cu": 4, "seu merda": 3, "seu lixo": 3, "seu babaca": 3,
    "seu idiota": 3, "sua puta": 4, "sua vadia": 4,
}

def detectar_linguagem_ofensiva(texto):
    if not texto:
        return [], 0, "Baixo"

    texto_norm = _normalize(texto)
    tokens = set(re.findall(r"\w+", texto_norm))

    ofensas_encontradas = []
    score = 0

    for palavra, peso in PALAVRAS_OFENSIVAS.items():
        if palavra in tokens:
            ofensas_encontradas.append(palavra)
            score += peso

    for expr, peso in EXPRESSOES_OFENSIVAS.items():
        if expr in texto_norm:
            ofensas_encontradas.append(expr)
            score += peso

    nivel = "Baixo" if score <= 3 else "Médio" if score <= 8 else "Alto"
    return list(dict.fromkeys(ofensas_encontradas)), score, nivel

# ================== CARREGAMENTO DOS MODELOS ==================
@st.cache_resource
def carregar_modelos():
    if not MODELS_DIR.exists():
        st.error(f"🚨 Pasta '{MODELS_DIR}' não encontrada! Rode primeiro `python train_models.py`.")
        st.stop()

    categorias = ['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia']
    modelos_dict = {}

    for cat in categorias:
        modelos_cat = {}
        for nome in ['Naive Bayes', 'Regressão Logística', 'SVM Linear']:
            arquivo = MODELS_DIR / f"{nome.lower().replace(' ', '_')}_{cat}.pkl"
            if arquivo.exists():
                modelos_cat[nome] = joblib.load(arquivo)
            else:
                st.warning(f"Modelo não encontrado: {arquivo.name}")
        modelos_dict[cat] = modelos_cat

    return modelos_dict

@st.cache_resource
def carregar_metricas():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

modelos_dict = carregar_modelos()
metricas = carregar_metricas()

# ================== HELPERS DE CLASSIFICAÇÃO ==================
def _prob_from_pipeline(pipeline, texto_clean):
    clf = pipeline.named_steps['clf']
    if hasattr(clf, "predict_proba"):
        return float(pipeline.predict_proba([texto_clean])[0][1])
    if hasattr(clf, "decision_function"):
        dec = float(pipeline.decision_function([texto_clean])[0])
        return float(1.0 / (1.0 + np.exp(-dec)))
    return None

def classificar_texto(texto_clean):
    """Retorna dict {categoria: {modelo: (pred, prob)}} + voto majoritário por categoria."""
    resultados = {}
    votos_por_categoria = {}
    for categoria, mods in modelos_dict.items():
        resultados[categoria] = {}
        votos = 0
        for nome, pipeline in mods.items():
            try:
                pred = int(pipeline.predict([texto_clean])[0])
                prob = _prob_from_pipeline(pipeline, texto_clean)
                resultados[categoria][nome] = (pred, prob)
                votos += pred
            except Exception:
                resultados[categoria][nome] = (None, None)
        votos_por_categoria[categoria] = votos  # 0..3
    return resultados, votos_por_categoria

# ================== DETECÇÃO DE IDEAÇÃO SUICIDA ==================
ALTO_RISCO = [
    "quero me matar", "vou me matar", "me matar", "me suicidar", "quero morrer",
    "vou morrer", "quero acabar com tudo", "acabar com tudo", "vou me entregar",
    "eu preferia estar morto", "eu queria estar morto", "preferia nao ter nascido",
    "queria nao existir", "dormir e nunca mais acordar", "esta e minha ultima mensagem",
    "adeus para sempre", "nao vejo sentido na vida", "nao vale a pena viver",
]
RISCO_MODERADO = [
    "nao quero viver", "nao aguento mais viver", "nao aguento mais", "nao suporto mais",
    "sou um peso", "sou um fardo", "melhor sem mim", "os outros vao ser mais felizes sem mim",
    "quero desaparecer", "vou deixar voces em paz", "nao tenho mais esperanca",
    "nada vai melhorar", "nao aguento essa dor", "quero sumir",
]
RISCO_LEVE = ["nao queria estar mais aqui", "nao quero mais estar aqui"]
CONTEXTOS_SOCIAIS = ["festa", "trabalho", "faculdade", "escola", "reuniao", "aqui"]

def avaliar_risco_suicida(texto):
    t = _normalize(texto)
    if any(e in t for e in ALTO_RISCO):
        return "ALTO"
    if any(e in t for e in RISCO_MODERADO):
        return "MODERADO"
    if any(e in t for e in RISCO_LEVE):
        if any(ctx in t for ctx in CONTEXTOS_SOCIAIS):
            return "BAIXO"
        return "MODERADO"
    return "NENHUM"

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("Menu")
    modo = st.radio(
        "Escolha o modo:",
        options=["Modo Análise", "Modo Chat", "Desempenho dos Modelos"],
        horizontal=False
    )

# ================== MODO ANÁLISE ==================
if modo == "Modo Análise":
    st.title("🛡️ Detecção de Discurso de Ódio + Linguagem Ofensiva")
    st.markdown("**ToLD-BR + Detector de Palavrões e Xingamentos**")

    col1, col2 = st.columns([3, 1])

    with col1:
        texto_novo = st.text_area(
            "Digite o texto para análise:",
            height=160,
            placeholder="Ex: Quero me matar... ou Vai se foder seu idiota!"
        )

    with col2:
        st.markdown("### 📌 Dicas")
        st.info(
            "• Textos mais longos e naturais têm melhor análise  \n"
            "• Palavrões ativam o detector de ofensa  \n"
            "• Frases suicidas recebem alerta especial"
        )

    if st.button("🔎 Analisar Texto", type="primary", use_container_width=True):
        if not texto_novo.strip():
            st.warning("⚠️ Por favor, digite um texto para análise.")
        else:
            with st.spinner("Processando o texto..."):
                # Ideação suicida
                nivel_suicida = avaliar_risco_suicida(texto_novo)
                if nivel_suicida in ("ALTO", "MODERADO"):
                    st.error("🚨 **ALERTA DE IDEAÇÃO SUICIDA DETECTADA**")
                    st.markdown("**Esta frase indica possível risco de suicídio ou automutilação.**")
                    st.info(
                        "**CVV - Centro de Valorização da Vida**\n\n"
                        "**Ligue 188** (24h, gratuito e sigiloso)\n\n"
                        "Ou acesse: [cvv.org.br](https://cvv.org.br)"
                    )

                # Linguagem ofensiva
                ofensas, score_ofensa, nivel_ofensa = detectar_linguagem_ofensiva(texto_novo)
                if ofensas:
                    st.warning(f"**Linguagem Ofensiva Detectada** — Nível: **{nivel_ofensa}** (Score: {score_ofensa})")
                    st.caption(f"Palavras/expressões encontradas: **{', '.join(ofensas)}**")

                # Modelos
                texto_clean = preprocessar_texto(texto_novo)
                resultados, votos = classificar_texto(texto_clean)

                linhas = []
                toxicas_majoritarias = 0
                for categoria, mods in resultados.items():
                    row = {"Categoria": categoria.replace("_", " ").title()}
                    for nome, (pred, prob) in mods.items():
                        if pred is None:
                            row[nome] = "❌ Erro"
                            continue
                        label = "Tóxico" if pred == 1 else "Não tóxico"
                        cor = "🔴" if pred == 1 else "🟢"
                        row[nome] = f"{cor} {label} ({prob:.1%})" if prob is not None else f"{cor} {label}"
                    if votos[categoria] >= 2:
                        toxicas_majoritarias += 1
                    linhas.append(row)

                st.success("✅ Análise concluída!")
                st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

                st.markdown("### Resumo da Análise")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Categorias analisadas", len(linhas))
                with c2: st.metric("Categorias tóxicas (≥2 modelos)", toxicas_majoritarias,
                                   delta_color="inverse" if toxicas_majoritarias > 0 else "normal")
                with c3: st.metric("Score de Ofensa", score_ofensa)
                with c4:
                    if toxicas_majoritarias >= 3 or score_ofensa >= 10:
                        risco = "Alto"
                    elif toxicas_majoritarias >= 1 or score_ofensa >= 5:
                        risco = "Médio"
                    else:
                        risco = "Baixo"
                    st.metric("Risco Geral Estimado", risco)

                st.info("**Legenda**\n• 🔴 Tóxico = indícios de discurso de ódio\n• 🟢 Não tóxico = sem indícios claros")

# ================== MODO CHAT ==================
elif modo == "Modo Chat":
    st.title("Chat com Proteção contra Discurso de Ódio")
    st.markdown("Escreva o que quiser. Se eu detectar ódio ou ofensa forte, vou te perguntar antes de continuar.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "pending" not in st.session_state:
        # guarda mensagem aguardando confirmação: {"content": str, "motivo": str}
        st.session_state.pending = None

    # Histórico
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Confirmação pendente (se houver)
    if st.session_state.pending is not None:
        pend = st.session_state.pending
        with st.chat_message("assistant"):
            st.warning("**Atenção**: Detectei possível discurso de ódio ou linguagem ofensiva forte.")
            st.caption(f"Motivo: {pend['motivo']}")
            st.write("Você realmente deseja enviar esta mensagem?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Sim, enviar mesmo assim", key="confirm_yes"):
                    st.session_state.chat_messages.append({"role": "user", "content": pend["content"]})
                    st.session_state.chat_messages.append({"role": "assistant", "content": "✅ Mensagem enviada."})
                    st.session_state.pending = None
                    st.rerun()
            with c2:
                if st.button("❌ Não, quero editar", key="confirm_no"):
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": "❌ Mensagem descartada. Você pode reescrever."}
                    )
                    st.session_state.pending = None
                    st.rerun()

    # Input
    prompt = st.chat_input("Digite sua mensagem...")
    if prompt and st.session_state.pending is None:
        # Alerta de ideação suicida (tem prioridade, não bloqueia envio)
        nivel = avaliar_risco_suicida(prompt)

        # Ofensa + ódio (voto majoritário entre as 3 classificações por categoria)
        ofensas, score_ofensa, _ = detectar_linguagem_ofensiva(prompt)
        texto_clean = preprocessar_texto(prompt)
        _, votos = classificar_texto(texto_clean)
        categorias_toxicas = [c for c, v in votos.items() if v >= 2]
        toxico_detectado = len(categorias_toxicas) > 0

        # Se risco alto de suicídio, não pergunta — envia e exibe alerta CVV
        if nivel in ("ALTO", "MODERADO"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            titulo = "🚨 **ALERTA SÉRIO DE IDEAÇÃO SUICIDA**" if nivel == "ALTO" \
                     else "⚠️ **Possível sofrimento emocional detectado**"
            aviso = (
                f"{titulo}\n\n"
                "**Centro de Valorização da Vida (CVV)**  \n"
                "📞 **Ligue 188** — 24h, gratuito e sigiloso  \n"
                "🌐 [cvv.org.br](https://cvv.org.br)"
            )
            st.session_state.chat_messages.append({"role": "assistant", "content": aviso})
            st.rerun()
        elif nivel == "BAIXO":
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "😕 Parece que você não está confortável nessa situação. Se quiser desabafar, estou aqui."
            })
            st.rerun()
        elif toxico_detectado or score_ofensa >= 8:
            motivos = []
            if categorias_toxicas:
                motivos.append("categorias detectadas: " + ", ".join(categorias_toxicas))
            if score_ofensa >= 8:
                motivos.append(f"score de ofensa {score_ofensa}")
            st.session_state.pending = {"content": prompt, "motivo": "; ".join(motivos)}
            st.rerun()
        else:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "✅ Mensagem analisada. Não identifiquei conteúdo tóxico."
            })
            st.rerun()

# ================== DESEMPENHO DOS MODELOS ==================
else:
    st.title("Desempenho dos Modelos")
    if not metricas:
        st.info("Métricas ainda não geradas. Rode `python train_models.py` para criar `models/metrics.json`.")
    else:
        linhas = []
        for cat, mods in metricas.items():
            for nome, m in mods.items():
                linhas.append({
                    "Categoria": cat,
                    "Modelo": nome,
                    "Acurácia": f"{m['accuracy']:.4f}",
                    "F1-macro": f"{m['f1_macro']:.4f}",
                    "F1 (tóxico)": f"{m['f1_toxico']:.4f}",
                    "Precisão (tóxico)": f"{m['precision_toxico']:.4f}",
                    "Recall (tóxico)": f"{m['recall_toxico']:.4f}",
                })
        st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

        st.caption(
            "F1-macro trata as classes igualmente (adequado a dados desbalanceados). "
            "Precisão alta = poucos falsos positivos. Recall alto = poucos textos tóxicos passam batido."
        )

# Rodapé
st.divider()
st.caption("Desenvolvido com spaCy, scikit-learn e Streamlit | Dataset ToLD-BR + Detector de Ofensa")
