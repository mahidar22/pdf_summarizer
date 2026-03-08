import os
import uuid
import shutil
import streamlit as st
from dotenv import load_dotenv

# ── Create folders ──
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/vector_db", exist_ok=True)

load_dotenv()

# ══════════════════════════════════════
# PAGE CONFIG — MUST BE FIRST ST CALL
# ══════════════════════════════════════
st.set_page_config(
    page_title="AI PDF Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════
# SAFE IMPORTS — Won't crash the app
# ══════════════════════════════════════
IMPORT_ERRORS = []

try:
    import pdfplumber
except ImportError:
    IMPORT_ERRORS.append("pdfplumber")

try:
    from langchain_groq import ChatGroq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False
    IMPORT_ERRORS.append("langchain-groq")

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
    LANGCHAIN_OK = True
except ImportError:
    LANGCHAIN_OK = False
    IMPORT_ERRORS.append("langchain-core")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    SPLITTER_OK = True
except ImportError:
    SPLITTER_OK = False
    IMPORT_ERRORS.append("langchain-text-splitters")

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False
    IMPORT_ERRORS.append("chromadb or langchain-community")

HF_OK = False
HFEmbeddings = None
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HFEmbeddings = HuggingFaceEmbeddings
    HF_OK = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as HFE2
        HFEmbeddings = HFE2
        HF_OK = True
    except ImportError:
        IMPORT_ERRORS.append("sentence-transformers")

try:
    from langdetect import detect as detect_lang
except ImportError:
    detect_lang = None


# ══════════════════════════════════════
# CSS STYLES
# ══════════════════════════════════════
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .sub-header { text-align: center; color: #888; margin-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1.2rem;
        text-align: center; border: 1px solid #e9ecef; margin: 0.3rem;
    }
    .section-card {
        background: #ffffff; border-left: 4px solid #667eea;
        padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .source-badge {
        background: #e8f0fe; color: #1967d2; padding: 2px 8px;
        border-radius: 12px; font-size: 0.8rem; margin-right: 4px;
    }
    .importance-must { color: #d93025; font-weight: bold; }
    .importance-rec { color: #f9ab00; font-weight: bold; }
    .importance-opt { color: #34a853; font-weight: bold; }
    .free-tag {
        background: linear-gradient(90deg, #34a853, #0d652d);
        color: white; padding: 4px 14px; border-radius: 15px;
        font-size: 0.85rem; font-weight: bold; display: inline-block;
    }
    .setup-box {
        background: #f0f7ff; border: 1px solid #c2dbff;
        border-radius: 10px; padding: 1.2rem; margin: 1rem 0;
    }
    .key-loaded {
        background: #e6f4ea; border: 1px solid #34a853;
        border-radius: 8px; padding: 0.5rem; text-align: center;
    }
    .error-box {
        background: #fce8e6; border: 1px solid #d93025;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
    [data-testid="stSidebar"] { min-width: 300px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════
def init_state():
    defaults = {
        "doc_data": None, "processed": False, "rag_ready": False,
        "summary": None, "detailed_summary": None,
        "key_points": None, "key_points_priority": None,
        "reading_guide": None, "messages": [], "chat_pairs": [],
        "suggestions": [], "current_file": None, "vectorstore": None,
        "llm": None, "session_id": str(uuid.uuid4())[:8],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════
# PDF EXTRACTION (built-in)
# ══════════════════════════════════════
def extract_pdf(file_path):
    pages = []
    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({
                    "page_number": i + 1,
                    "text": text.strip(),
                    "word_count": len(text.split()),
                })
            full_text = "\n\n".join(p["text"] for p in pages if p["text"])
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
    return pages, full_text


# ══════════════════════════════════════
# LLM SETUP (built-in)
# ══════════════════════════════════════
def get_llm(api_key, model):
    if not GROQ_OK:
        st.error("langchain-groq not installed!")
        return None
    try:
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=0.3,
        )
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None


# ══════════════════════════════════════
# VECTOR STORE (built-in)
# ══════════════════════════════════════
def build_vectorstore(pages):
    if not SPLITTER_OK or not CHROMA_OK or not HF_OK:
        missing = []
        if not SPLITTER_OK: missing.append("langchain-text-splitters")
        if not CHROMA_OK: missing.append("chromadb")
        if not HF_OK: missing.append("sentence-transformers")
        st.error(f"Missing packages: {', '.join(missing)}")
        return None

    documents = []
    for p in pages:
        if p["text"]:
            documents.append(Document(
                page_content=p["text"],
                metadata={"page_number": p["page_number"], "source": f"Page {p['page_number']}"},
            ))

    if not documents:
        st.error("No text found in document!")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    persist_dir = "data/vector_db"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = HFEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def query_rag(question, vectorstore, llm, chat_history=None):
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(
        f"[Page {d.metadata.get('page_number', '?')}]: {d.page_content}"
        for d in docs
    )

    history = ""
    if chat_history:
        history = "Previous conversation:\n"
        history += "\n".join(f"Q: {q}\nA: {a}" for q, a in chat_history[-2:])
        history += "\n\n"

    prompt = f"""{history}Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find this in the document."
Include page numbers as [Page X].

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)

    sources = []
    seen = set()
    for d in docs:
        pg = d.metadata.get("page_number", 0)
        if pg not in seen:
            sources.append({"page": pg, "text": d.page_content[:300]})
            seen.add(pg)

    return response.content.strip(), sorted(sources, key=lambda x: x["page"])


# ══════════════════════════════════════
# HELPER FUNCTIONS (built-in)
# ══════════════════════════════════════
def generate_summary(text, llm, mode="concise"):
    if mode == "concise":
        prompt = f"Write a concise summary (150-300 words) of this document. Focus on main ideas.\n\n{text[:7000]}\n\nSUMMARY:"
    else:
        prompt = f"Write a detailed summary of this document. Cover all topics and conclusions.\n\n{text[:7000]}\n\nDETAILED SUMMARY:"
    response = llm.invoke(prompt)
    return response.content.strip()


def extract_key_points(text, llm):
    prompt = f"""Extract 5-15 key points from this document.
Each point: one clear sentence. Most important first.

Document:
{text[:7000]}

Key points as numbered list:"""
    response = llm.invoke(prompt)
    points = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if line:
            cleaned = line.lstrip("0123456789.-•*) ").strip()
            if cleaned and len(cleaned) > 5:
                points.append(cleaned)
    return points


def extract_priority_points(text, llm):
    prompt = f"""Extract key points with priority levels.
Use: CRITICAL (must know), IMPORTANT (should know), NICE-TO-KNOW (supplementary)

Document:
{text[:7000]}

Format:
1. CRITICAL: point here
2. IMPORTANT: point here
3. NICE-TO-KNOW: point here"""
    response = llm.invoke(prompt)
    results = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if not line: continue
        for i, ch in enumerate(line):
            if ch == "." and i < 4:
                line = line[i+1:].strip()
                break
        priority = "NICE-TO-KNOW"
        upper = line.upper()
        if "CRITICAL" in upper:
            priority = "CRITICAL"
            line = line.split(":", 1)[-1].strip() if ":" in line else line
        elif "IMPORTANT" in upper:
            priority = "IMPORTANT"
            line = line.split(":", 1)[-1].strip() if ":" in line else line
        elif "NICE" in upper:
            priority = "NICE-TO-KNOW"
            line = line.split(":", 1)[-1].strip() if ":" in line else line
        for e in ["🔴", "🟡", "🟢", "CRITICAL", "IMPORTANT", "NICE-TO-KNOW"]:
            line = line.replace(e, "").strip()
        if line and len(line) > 5:
            results.append({"point": line, "priority": priority})
    return results


def generate_reading_guide(text, pages, llm):
    total_words = sum(p["word_count"] for p in pages)
    reading_time = round(total_words / 220, 1)

    prompt = f"""Analyze this document structure and provide:
1. A reading strategy (2-3 sentences)
2. List of sections with importance (MUST-READ / RECOMMENDED / OPTIONAL)
3. Sections to skip if short on time

Document:
{text[:5000]}

Format:
STRATEGY: <your strategy>
SECTIONS:
1. Title | Description | Importance
SKIP: <section names>"""

    response = llm.invoke(prompt)

    return {
        "strategy": response.content.strip(),
        "total_time": reading_time,
        "total_words": total_words,
        "num_pages": len(pages),
    }


def get_suggestions(text, llm):
    try:
        prompt = f"Based on this document, suggest 5 questions a reader might ask. One per line.\n\n{text[:2000]}\n\nQuestions:"
        response = llm.invoke(prompt)
        questions = [q.strip().lstrip("0123456789.-) ") for q in response.content.strip().split("\n") if q.strip()]
        return questions[:5]
    except Exception:
        return ["What is the main topic?", "What are the key findings?", "Summarize the conclusion"]


def save_file(uploaded_file):
    path = os.path.join("data/uploads", uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def download_text():
    parts = []
    if st.session_state.summary:
        parts.append("=== SUMMARY ===\n" + st.session_state.summary)
    if st.session_state.key_points:
        parts.append("\n\n=== KEY POINTS ===\n" + "\n".join(f"• {p}" for p in st.session_state.key_points))
    return "\n".join(parts) if parts else "No content yet."


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 AI PDF Summarizer")
    st.markdown('<span class="free-tag">🆓 100% FREE</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Show import errors if any
    if IMPORT_ERRORS:
        st.warning(f"⚠️ Missing: {', '.join(IMPORT_ERRORS)}")

    # ── Provider ──
    provider = st.selectbox(
        "🤖 AI Provider",
        ["Groq (FREE ⭐)", "Google Gemini (FREE)", "OpenAI (Paid)"],
        index=0,
    )

    api_key = ""
    model_choice = ""

    if provider == "Groq (FREE ⭐)":
        saved = os.getenv("GROQ_API_KEY", "")
        if saved:
            api_key = saved
            st.markdown('<div class="key-loaded">✅ API Key loaded automatically</div>', unsafe_allow_html=True)
        else:
            api_key = st.text_input("🔑 Groq API Key", type="password",
                                     help="Get FREE key: https://console.groq.com/keys")
            if not api_key:
                st.markdown("""
                <div class="setup-box">
                <b>Get FREE key:</b><br>
                1. <a href="https://console.groq.com/keys" target="_blank">console.groq.com/keys</a><br>
                2. Sign up with Google<br>
                3. Create API Key<br>
                4. Paste above
                </div>
                """, unsafe_allow_html=True)

        model_choice = st.selectbox("🧠 Model", [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ], index=0)

    elif provider == "Google Gemini (FREE)":
        saved = os.getenv("GOOGLE_API_KEY", "")
        if saved:
            api_key = saved
            st.markdown('<div class="key-loaded">✅ Key loaded</div>', unsafe_allow_html=True)
        else:
            api_key = st.text_input("🔑 Gemini Key", type="password")
        model_choice = st.selectbox("🧠 Model", ["gemini-2.0-flash", "gemini-1.5-flash"], index=0)

    else:
        saved = os.getenv("OPENAI_API_KEY", "")
        if saved:
            api_key = saved
            st.markdown('<div class="key-loaded">✅ Key loaded</div>', unsafe_allow_html=True)
        else:
            api_key = st.text_input("🔑 OpenAI Key", type="password")
        model_choice = st.selectbox("🧠 Model", ["gpt-4o-mini", "gpt-4o"], index=0)

    st.markdown("---")

    # ── Upload ──
    uploaded_file = st.file_uploader("📁 Upload PDF", type=["pdf", "png", "jpg", "jpeg"])

    # ── Process ──
    if uploaded_file and api_key:
        # Reset on new file
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.processed = False
            st.session_state.rag_ready = False
            st.session_state.summary = None
            st.session_state.detailed_summary = None
            st.session_state.key_points = None
            st.session_state.key_points_priority = None
            st.session_state.reading_guide = None
            st.session_state.messages = []
            st.session_state.chat_pairs = []
            st.session_state.suggestions = []
            st.session_state.vectorstore = None
            st.session_state.llm = None
            st.session_state.current_file = uploaded_file.name

        if st.button("🚀 Process Document", use_container_width=True, type="primary"):
            file_path = save_file(uploaded_file)

            # Extract PDF
            with st.spinner("📖 Extracting text…"):
                pages, full_text = extract_pdf(file_path)

            if not full_text:
                st.error("No text found in PDF!")
            else:
                st.session_state.doc_data = {
                    "pages": pages,
                    "full_text": full_text,
                    "file_name": uploaded_file.name,
                    "num_pages": len(pages),
                    "total_words": sum(p["word_count"] for p in pages),
                }
                st.session_state.processed = True

                # Setup LLM
                llm = get_llm(api_key, model_choice)
                st.session_state.llm = llm

                # Build Vector Store
                with st.spinner("🧠 Building search index…"):
                    try:
                        vs = build_vectorstore(pages)
                        if vs:
                            st.session_state.vectorstore = vs
                            st.session_state.rag_ready = True

                            # Get suggestions
                            if llm:
                                try:
                                    st.session_state.suggestions = get_suggestions(full_text, llm)
                                except Exception:
                                    pass

                            st.success("✅ Ready! Upload processed successfully.")
                        else:
                            st.warning("Vector store failed, but summary/key points will still work!")
                    except Exception as e:
                        st.warning(f"Search index failed: {e}")

    elif uploaded_file and not api_key:
        st.warning("⚠️ Enter your API key above!")

    # ── Doc Info ──
    if st.session_state.processed and st.session_state.doc_data:
        dd = st.session_state.doc_data
        st.markdown("---")
        st.markdown("### 📊 Document Info")
        st.markdown(f"**File:** {dd['file_name']}")
        st.markdown(f"**Pages:** {dd['num_pages']}")
        st.markdown(f"**Words:** {dd['total_words']:,}")
        st.markdown(f"**Read time:** ~{round(dd['total_words'] / 220, 1)} min")
        st.markdown("---")
        st.download_button("📥 Download Summary", data=download_text(),
                           file_name="summary.txt", mime="text/plain", use_container_width=True)


# ══════════════════════════════════════════
# MAIN CONTENT AREA
# ══════════════════════════════════════════

# ── Welcome Screen (no file uploaded) ──
if not uploaded_file or not api_key:
    st.markdown('<div class="main-header">📄 AI-Powered PDF Summarizer</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload any PDF → Get summaries, key points, reading guide & AI chat — 100% FREE!</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-card">📝<br><b>Smart Summary</b><br>Concise & Detailed</div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card">🎯<br><b>Key Points</b><br>Prioritized</div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-card">📖<br><b>Reading Guide</b><br>Time Estimates</div>', unsafe_allow_html=True)
    c4.markdown('<div class="metric-card">💬<br><b>AI Chat</b><br>Ask Anything</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="setup-box">
    <h3>🚀 Get Started in 2 Minutes</h3>
    <ol>
    <li>Get your FREE API key → <a href="https://console.groq.com/keys" target="_blank"><b>console.groq.com/keys</b></a></li>
    <li>Paste the key in the sidebar (left side)</li>
    <li>Upload your PDF file</li>
    <li>Click <b>🚀 Process Document</b></li>
    <li>Explore all 4 tabs!</li>
    </ol>
    <p><b>✅ No payment needed • No credit card • Works forever</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Show import status
    if IMPORT_ERRORS:
        st.markdown("---")
        st.markdown("### ⚠️ Missing Packages")
        st.code(f"pip install {' '.join(IMPORT_ERRORS)}", language="bash")
    
    st.stop()

# ── Not processed yet ──
if not st.session_state.processed:
    st.info("👈 Click **🚀 Process Document** in the sidebar to begin.")
    st.stop()


# ══════════════════════════════════════════
# 4 TABS
# ══════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🎯 Key Points", "📖 Reading Guide", "💬 Chat"])
dd = st.session_state.doc_data
llm = st.session_state.llm

# ═══════════════════
# TAB 1: SUMMARY
# ═══════════════════
with tab1:
    st.header("📝 Document Summary")

    if not llm:
        st.error("LLM not available. Check your API key.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Generate Concise Summary", use_container_width=True, type="primary"):
                with st.spinner("Generating…"):
                    try:
                        st.session_state.summary = generate_summary(dd["full_text"], llm, "concise")
                    except Exception as e:
                        st.error(f"Error: {e}")
        with c2:
            if st.button("Generate Detailed Summary", use_container_width=True):
                with st.spinner("Generating…"):
                    try:
                        st.session_state.detailed_summary = generate_summary(dd["full_text"], llm, "detailed")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.summary:
            st.subheader("📋 Concise Summary")
            st.markdown(st.session_state.summary)

        if st.session_state.detailed_summary:
            st.subheader("📄 Detailed Summary")
            st.markdown(st.session_state.detailed_summary)


# ═══════════════════
# TAB 2: KEY POINTS
# ═══════════════════
with tab2:
    st.header("🎯 Key Points")

    if not llm:
        st.error("LLM not available.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Extract Key Points", use_container_width=True, type="primary"):
                with st.spinner("Extracting…"):
                    try:
                        st.session_state.key_points = extract_key_points(dd["full_text"], llm)
                    except Exception as e:
                        st.error(f"Error: {e}")
        with c2:
            if st.button("Extract with Priority", use_container_width=True):
                with st.spinner("Extracting…"):
                    try:
                        st.session_state.key_points_priority = extract_priority_points(dd["full_text"], llm)
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.key_points:
            st.subheader("Key Points")
            for i, p in enumerate(st.session_state.key_points, 1):
                st.markdown(f"**{i}.** {p}")

        if st.session_state.key_points_priority:
            st.subheader("Prioritized Key Points")
            for item in st.session_state.key_points_priority:
                pr = item["priority"]
                emoji = {"CRITICAL": "🔴", "IMPORTANT": "🟡", "NICE-TO-KNOW": "🟢"}.get(pr, "⚪")
                css = {"CRITICAL": "importance-must", "IMPORTANT": "importance-rec", "NICE-TO-KNOW": "importance-opt"}.get(pr, "")
                st.markdown(f'{emoji} <span class="{css}">[{pr}]</span> {item["point"]}', unsafe_allow_html=True)


# ═══════════════════
# TAB 3: READING GUIDE
# ═══════════════════
with tab3:
    st.header("📖 Reading Guide")

    if not llm:
        st.error("LLM not available.")
    else:
        if st.button("Generate Reading Guide", use_container_width=True, type="primary"):
            with st.spinner("Analyzing…"):
                try:
                    st.session_state.reading_guide = generate_reading_guide(dd["full_text"], dd["pages"], llm)
                except Exception as e:
                    st.error(f"Error: {e}")

        guide = st.session_state.reading_guide
        if guide:
            m1, m2, m3 = st.columns(3)
            m1.metric("📄 Pages", guide["num_pages"])
            m2.metric("📝 Words", f"{guide['total_words']:,}")
            m3.metric("⏱️ Reading Time", f"{guide['total_time']} min")

            st.subheader("📋 Reading Strategy & Analysis")
            st.markdown(guide["strategy"])


# ═══════════════════
# TAB 4: CHAT
# ═══════════════════
with tab4:
    st.header("💬 Chat with Your Document")

    if not st.session_state.rag_ready or not st.session_state.vectorstore:
        st.warning("⚠️ Search index not available. Process a document first.")
        st.info("If you already processed, the vector store may have failed. Summary and Key Points tabs should still work!")
    elif not llm:
        st.error("LLM not available. Check API key.")
    else:
        # Suggested questions
        if st.session_state.suggestions:
            st.markdown("**💡 Suggested questions:**")
            cols = st.columns(min(len(st.session_state.suggestions), 3))
            for idx, q in enumerate(st.session_state.suggestions[:3]):
                if cols[idx].button(q, key=f"sq_{idx}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    answer, sources = query_rag(q, st.session_state.vectorstore, llm, st.session_state.chat_pairs)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    st.session_state.chat_pairs.append((q, answer))
                    st.rerun()

        st.markdown("---")

        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📄 Sources & Page References"):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<span class="source-badge">Page {src["page"]}</span> {src["text"][:200]}…',
                                unsafe_allow_html=True,
                            )

        # Chat input
        if prompt := st.chat_input("Ask anything about the document…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer, sources = query_rag(
                            prompt, st.session_state.vectorstore, llm, st.session_state.chat_pairs
                        )
                        st.markdown(answer)
                        if sources:
                            with st.expander("📄 Sources"):
                                for src in sources:
                                    st.markdown(
                                        f'<span class="source-badge">Page {src["page"]}</span> {src["text"][:200]}…',
                                        unsafe_allow_html=True,
                                    )
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                        st.session_state.chat_pairs.append((prompt, answer))
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Clear chat
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.chat_pairs = []
                st.rerun()