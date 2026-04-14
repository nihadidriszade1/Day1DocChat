"""
Sənədlərinlə Danış — PDF RAG Chatbot
Local Embeddings + Groq API (llama-3.3-70b) — Tamamilə Pulsuz
"""

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import requests

# ─────────────────────────────────────────────
#  SESSION STATE — İLK OLARAQ
# ─────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "conversation":    None,
        "chat_history":    [],
        "processing_done": False,
        "doc_count":       0,
        "chunk_count":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sənədlərimlə Danış",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');
:root{--bg:#0d0f14;--surface:#151920;--border:#232836;--accent:#6c8fff;--accent2:#a78bfa;--text:#e8ecf4;--muted:#6b7280;--user-bg:#1a2035;--bot-bg:#131720;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg)!important;color:var(--text)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"]>div:first-child{padding:2rem 1.5rem;}
.sidebar-logo{font-family:'Syne',sans-serif;font-weight:500;font-size:1.4rem;letter-spacing:-.03em;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.25rem;}
.sidebar-sub{font-size:.72rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:2rem;}
[data-testid="stTextInput"] input{background:var(--bg)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:10px!important;}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;border:none!important;color:#fff!important;font-family:'Syne',sans-serif!important;font-weight:600!important;border-radius:10px!important;padding:.6rem 1.4rem!important;width:100%;}
.stButton>button:hover{opacity:.88;}
.chat-wrapper{display:flex;flex-direction:column;gap:1.25rem;padding:.5rem 0;}
.msg{display:flex;gap:.85rem;animation:fadeUp .3s ease both;}
.msg.user{flex-direction:row-reverse;}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;margin-top:4px;}
.avatar.bot{background:linear-gradient(135deg,#6c8fff22,#a78bfa22);border:1px solid var(--border);}
.avatar.user{background:linear-gradient(135deg,#a78bfa33,#6c8fff33);border:1px solid var(--border);}
.bubble{max-width:72%;padding:.85rem 1.1rem;border-radius:16px;font-size:.91rem;line-height:1.65;white-space:pre-wrap;}
.bubble.bot{background:var(--bot-bg);border:1px solid var(--border);border-top-left-radius:4px;}
.bubble.user{background:var(--user-bg);border:1px solid #2a3454;border-top-right-radius:4px;}
.main-header{font-family:'Syne',sans-serif;font-size:2rem;font-weight:500;letter-spacing:-.04em;background:linear-gradient(135deg,var(--text) 30%,var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.2rem;}
.main-sub{font-size:.82rem;color:var(--muted);letter-spacing:.06em;margin-bottom:1.8rem;}
.status-badge{display:inline-flex;align-items:center;gap:.4rem;font-size:.74rem;font-weight:500;padding:.3rem .75rem;border-radius:100px;letter-spacing:.04em;}
.badge-ready{background:#34d39915;color:#34d399;border:1px solid #34d39933;}
.badge-waiting{background:#fbbf2415;color:#fbbf24;border:1px solid #fbbf2433;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOCAL EMBEDDING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Embedding modeli yüklənir...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = load_embedding_model()

    def embed_documents(self, texts: list) -> list:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


# ─────────────────────────────────────────────
#  GROQ CHAT  (OpenAI-compatible, tamamilə pulsuz)
# ─────────────────────────────────────────────
def groq_chat(api_key: str, history: list, question: str, context: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    messages = [
        {
            "role": "system",
            "content": """Sən yüklənmiş sənədlərdəki məlumatları əsas götürərək sualları cavablandıran AI assistentsən.
Təlimatlar:
- Cavabı yalnız verilmiş kontekstə əsasla.
- Kontekstdə məlumat yoxdursa, bunu açıq bildir.
- Cavabı aydın, strukturlu Azərbaycan dilində ver.

Kontekst:
""" + context,
        }
    ]

    for msg in history[-6:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"],
        })

    messages.append({"role": "user", "content": question})

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        timeout=60,
    )

    if not resp.ok:
        err = resp.json().get("error", {}).get("message", resp.text)
        raise Exception(f"Groq API xətası: {err}")

    return resp.json()["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────
#  MODULE 1 — PDF TEXT
# ─────────────────────────────────────────────
def get_pdf_text(pdf_files: list) -> str:
    if not pdf_files:
        raise ValueError("Heç bir PDF seçilməyib.")
    all_text = []
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            if reader.is_encrypted:
                st.warning(f"⚠️ '{pdf_file.name}' şifrəlidir.")
                continue
            for page in reader.pages:
                try:
                    t = page.extract_text()
                    if t and t.strip():
                        all_text.append(t)
                except Exception:
                    continue
        except Exception as e:
            st.error(f"❌ '{pdf_file.name}': {e}")
    combined = "\n\n".join(all_text)
    if not combined.strip():
        raise ValueError("PDF-lərdən mətn çıxarıla bilmədi.")
    return combined


# ─────────────────────────────────────────────
#  MODULE 2 — CHUNKING
# ─────────────────────────────────────────────
def get_text_chunks(raw_text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    if not chunks:
        raise ValueError("Mətn parçalanarkən boş nəticə.")
    return chunks


# ─────────────────────────────────────────────
#  MODULE 3 — VECTOR STORE
# ─────────────────────────────────────────────
def get_vectorstore(text_chunks: list) -> FAISS:
    if not text_chunks:
        raise ValueError("Parçalar boşdur.")
    try:
        embeddings = LocalEmbeddings()
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        raise Exception(f"Vektor bazası xətası: {e}")


# ─────────────────────────────────────────────
#  MODULE 4 — CONVERSATION CHAIN
# ─────────────────────────────────────────────
def get_conversation_chain(vectorstore: FAISS, api_key: str):
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    def chain(question: str, history: list) -> str:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        return groq_chat(api_key, history, question, context)
    return chain


# ─────────────────────────────────────────────
#  MODULE 5 — HANDLE INPUT
# ─────────────────────────────────────────────
def handle_user_input(user_question: str) -> None:
    if not st.session_state.get("conversation"):
        st.error("❌ Əvvəlcə PDF yükləyin.")
        return

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question,
    })

    with st.spinner("Düşünürəm..."):
        try:
            answer = st.session_state.conversation(
                question=user_question,
                history=st.session_state.chat_history[:-1],
            )
            if not answer or not str(answer).strip():
                answer = "Cavab alına bilmədi. Yenidən cəhd edin."
        except Exception as e:
            answer = f"❌ Xəta: {str(e)}"

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": str(answer),
    })


# ─────────────────────────────────────────────
#  RENDER CHAT
# ─────────────────────────────────────────────
def render_chat() -> None:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center;padding:4rem 0;color:#6b7280;">
            <div style="font-size:2.5rem;margin-bottom:1rem;">📄</div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;color:#9ca3af;">
                PDF yükləyin, sual verin</div>
            <div style="font-size:.8rem;margin-top:.4rem;">
                Sənədiniz haqqında hər şeyi soruşa bilərsiniz</div>
        </div>""", unsafe_allow_html=True)
        return

    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        css  = "bot"  if msg["role"] == "assistant" else "user"
        icon = "🤖" if msg["role"] == "assistant" else "🧑"
        st.markdown(f"""
        <div class="msg {css}">
            <div class="avatar {css}">{icon}</div>
            <div class="bubble {css}">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main() -> None:

    with st.sidebar:
        st.markdown('<div class="sidebar-logo">📄 DocChat</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">Local Embed · Groq LLaMA 3.3</div>',
                    unsafe_allow_html=True)

        st.markdown("**🔑 Groq API Key**")
        st.markdown(
            '<div style="font-size:.72rem;color:#6b7280;margin-bottom:.5rem;">'
            '🔗 <a href="https://console.groq.com" target="_blank" '
            'style="color:#6c8fff;">console.groq.com</a> — pulsuz</div>',
            unsafe_allow_html=True)
        api_key = st.text_input("API Key", type="password",
                                placeholder="gsk_...", label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📂 PDF Faylları**")
        pdf_files = st.file_uploader("PDF", type=["pdf"],
                                     accept_multiple_files=True,
                                     label_visibility="collapsed")
        if pdf_files:
            st.markdown(
                f'<div style="font-size:.78rem;color:#6b7280;margin:.4rem 0 1rem;">'
                f'✅ {len(pdf_files)} fayl seçildi</div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⚡ Sənədləri İşlə", use_container_width=True):
            if not api_key:
                st.error("❌ Groq API Key daxil edin.")
            elif not pdf_files:
                st.error("❌ PDF seçin.")
            else:
                try:
                    with st.spinner("📖 PDF oxunur..."):
                        raw = get_pdf_text(pdf_files)
                    with st.spinner("✂️ Parçalanır..."):
                        chunks = get_text_chunks(raw)
                    with st.spinner("🧠 Lokal embedding yaradılır..."):
                        vs = get_vectorstore(chunks)
                    with st.spinner("🔗 Zəncir qurulur..."):
                        st.session_state.conversation = get_conversation_chain(vs, api_key)
                    st.session_state.update({
                        "processing_done": True,
                        "doc_count":       len(pdf_files),
                        "chunk_count":     len(chunks),
                        "chat_history":    [],
                    })
                    st.success(f"✅ Hazırdır! {len(chunks)} parça indeksləndi.")
                except Exception as e:
                    st.error(f"❌ {e}")

        if st.session_state.processing_done:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:.78rem;color:#6b7280;line-height:1.8;">
                📑 <b style="color:#e8ecf4">{st.session_state.doc_count}</b> sənəd<br>
                🧩 <b style="color:#e8ecf4">{st.session_state.chunk_count}</b> parça<br>
                🔒 all-MiniLM-L6-v2 (lokal)<br>
                ⚡ llama-3.3-70b (Groq)
            </div>""", unsafe_allow_html=True)

        st.markdown("<br><hr>", unsafe_allow_html=True)
        if st.button("🗑️ Sıfırla", use_container_width=True):
            st.session_state.update({
                "chat_history": [], "conversation": None, "processing_done": False
            })
            st.rerun()

    # ── MAIN PANEL ────────────────────────────
    st.markdown('<div class="main-header">Sənədlərinlə Danış</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Local Embeddings · LLaMA 3.3 70B · FAISS</div>',
                unsafe_allow_html=True)

    if st.session_state.processing_done:
        st.markdown(
            '<span class="status-badge badge-ready">● Hazırdır — sual verə bilərsiniz</span>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="status-badge badge-waiting">◌ Sənəd gözlənilir</span>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    render_chat()
    st.markdown("<br>", unsafe_allow_html=True)

    q = st.chat_input(
        "Sənəd haqqında sualınızı yazın...",
        disabled=not st.session_state.processing_done,
    )
    if q:
        handle_user_input(q)
        st.rerun()


if __name__ == "__main__":
    main()