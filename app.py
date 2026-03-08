"""
AI PDF Summarizer — Deployed on Render
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv

# ── Create directories (needed for cloud deployment) ──
for folder in ["data/uploads", "data/vector_db", ".streamlit"]:
    os.makedirs(folder, exist_ok=True)

load_dotenv()

# ... rest of your app.py stays the same ...


"""
AI PDF Summarizer — FREE with Groq API
No paid subscription needed!
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv

os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/vector_db", exist_ok=True)
load_dotenv()

from modules.pdf_extractor import PDFExtractor
from modules.summarizer import Summarizer
from modules.key_points import KeyPointsExtractor
from modules.reading_guide import ReadingGuideGenerator
from modules.rag_pipeline import RAGPipeline
from modules.chatbot import PDFChatbot

OCR_AVAILABLE = False
try:
    from modules.ocr_engine import OCREngine
    OCR_AVAILABLE = True
except ImportError:
    pass

st.set_page_config(page_title="AI PDF Summarizer", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

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
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        text-align: center; border: 1px solid #e9ecef;
    }
    .section-card {
        background: #ffffff; border-left: 4px solid #667eea;
        padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .source-badge {
        background: #e8f0fe; color: #1967d2; padding: 2px 8px;
        border-radius: 12px; font-size: 0.8rem;
    }
    .importance-must { color: #d93025; font-weight: bold; }
    .importance-rec { color: #f9ab00; font-weight: bold; }
    .importance-opt { color: #34a853; font-weight: bold; }
    .free-tag {
        background: linear-gradient(90deg, #34a853, #0d652d);
        color: white; padding: 3px 10px; border-radius: 15px;
        font-size: 0.8rem; font-weight: bold;
    }
    .setup-box {
        background: #f0f7ff; border: 1px solid #c2dbff;
        border-radius: 10px; padding: 1.2rem; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "document_data": None, "file_processed": False, "rag_built": False,
        "summary": None, "detailed_summary": None, "key_points": None,
        "key_points_priority": None, "reading_guide": None, "chat_messages": [],
        "chatbot": None, "session_id": str(uuid.uuid4())[:8],
        "suggested_questions": [], "current_file": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


def save_uploaded_file(f) -> str:
    path = os.path.join("data/uploads", f.name)
    with open(path, "wb") as out:
        out.write(f.getbuffer())
    return path


def build_download_text() -> str:
    parts = []
    if st.session_state.summary:
        parts.append("=== SUMMARY ===\n" + st.session_state.summary)
    if st.session_state.key_points:
        parts.append("\n\n=== KEY POINTS ===\n" + "\n".join(f"• {p}" for p in st.session_state.key_points))
    if st.session_state.reading_guide:
        rg = st.session_state.reading_guide
        parts.append(f"\n\n=== READING GUIDE ===\nTime: {rg.total_reading_time_min} min\n{rg.reading_strategy}")
    return "\n".join(parts) if parts else "No content yet."


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 AI PDF Summarizer")
    st.markdown('<span class="free-tag">🆓 FREE</span>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Provider Selection ──
    provider = st.selectbox(
        "🤖 AI Provider",
        ["Groq (FREE ⭐ Recommended)", "Google Gemini (FREE)", "OpenAI (Paid)"],
        index=0,
    )

    api_key = ""
    model_choice = ""

    if provider == "Groq (FREE ⭐ Recommended)":
        api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="FREE key from https://console.groq.com/keys",
        )
        model_choice = st.selectbox(
            "🧠 Model",
            [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            index=0,
            help="llama-3.1-8b = fastest, 70b = smartest",
        )
        if not api_key:
            st.markdown("""
            <div class="setup-box">
            <b>Get FREE Key (30 seconds):</b><br>
            1. Go to <a href="https://console.groq.com/keys" target="_blank">console.groq.com/keys</a><br>
            2. Sign up with Google<br>
            3. Click "Create API Key"<br>
            4. Paste it above<br><br>
            <b>✅ Free forever • No credit card</b>
            </div>
            """, unsafe_allow_html=True)

    elif provider == "Google Gemini (FREE)":
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
        )
        model_choice = st.selectbox("🧠 Model", ["gemini-2.0-flash", "gemini-1.5-flash"], index=0)
        if not api_key:
            st.markdown("🔗 [Get FREE key](https://aistudio.google.com/apikey)")

    else:
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )
        model_choice = st.selectbox("🧠 Model", ["gpt-4o-mini", "gpt-4o"], index=0)

    st.markdown("---")

    # ── File Upload ──
    uploaded_file = st.file_uploader(
        "📁 Upload PDF",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Drag & drop or click to upload",
    )

    # ── Process Button ──
    if uploaded_file and api_key:
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.file_processed = False
            st.session_state.rag_built = False
            st.session_state.summary = None
            st.session_state.detailed_summary = None
            st.session_state.key_points = None
            st.session_state.key_points_priority = None
            st.session_state.reading_guide = None
            st.session_state.chat_messages = []
            st.session_state.chatbot = None
            st.session_state.suggested_questions = []
            st.session_state.current_file = uploaded_file.name

        if st.button("🚀 Process Document", use_container_width=True, type="primary"):
            file_path = save_uploaded_file(uploaded_file)

            # Extract text
            with st.spinner("📖 Extracting text…"):
                extractor = PDFExtractor(file_path)
                doc_data = extractor.extract()

            # OCR if needed
            if doc_data.scanned_pages and OCR_AVAILABLE:
                with st.spinner("🔍 Running OCR…"):
                    try:
                        ocr = OCREngine()
                        ocr_results = ocr.extract_from_pdf(file_path, pages=doc_data.scanned_pages)
                        for page in doc_data.pages:
                            if page.page_number in ocr_results:
                                t = ocr_results[page.page_number]
                                if len(t) > len(page.text):
                                    page.text = t
                                    page.word_count = len(t.split())
                        doc_data.full_text = "\n\n".join(p.text for p in doc_data.pages if p.text)
                        doc_data.total_words = sum(p.word_count for p in doc_data.pages)
                    except Exception:
                        pass

            st.session_state.document_data = doc_data
            st.session_state.file_processed = True

            # Build RAG
            with st.spinner("🧠 Building search index…"):
                try:
                    rag = RAGPipeline(api_key=api_key, model=model_choice, persist_dir="data/vector_db")
                    pages_rag = [{"page_number": p.page_number, "text": p.text} for p in doc_data.pages]
                    num_chunks = rag.build(pages_rag)
                    st.session_state.rag_built = True

                    chatbot = PDFChatbot(rag_pipeline=rag, session_id=st.session_state.session_id)
                    st.session_state.chatbot = chatbot

                    try:
                        st.session_state.suggested_questions = chatbot.get_suggested_questions(doc_data.full_text)
                    except Exception:
                        pass

                    st.success(f"✅ Ready! {num_chunks} chunks indexed.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ── Document Info ──
    if st.session_state.file_processed and st.session_state.document_data:
        dd = st.session_state.document_data
        st.markdown("---")
        st.markdown("### 📊 Document Info")
        st.markdown(f"**File:** {dd.file_name}")
        st.markdown(f"**Pages:** {dd.num_pages}")
        st.markdown(f"**Words:** {dd.total_words:,}")
        st.markdown(f"**Read time:** ~{round(dd.total_words / 220, 1)} min")
        st.markdown("---")
        st.download_button("📥 Download Summary", data=build_download_text(),
                           file_name="summary.txt", mime="text/plain", use_container_width=True)


# ══════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════
if not uploaded_file or not api_key:
    st.markdown('<div class="main-header">📄 AI-Powered PDF Summarizer</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload any PDF → Get summaries, key points, reading guide & chat — 100% FREE!</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-card">📝<br><b>Smart Summary</b><br>Concise & Detailed</div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card">🎯<br><b>Key Points</b><br>Prioritized</div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-card">📖<br><b>Reading Guide</b><br>Time Estimates</div>', unsafe_allow_html=True)
    c4.markdown('<div class="metric-card">💬<br><b>AI Chat</b><br>Ask Anything</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="setup-box">
    <h3>🚀 Quick Start (2 minutes)</h3>
    <ol>
    <li>Get FREE API key → <a href="https://console.groq.com/keys" target="_blank"><b>console.groq.com/keys</b></a></li>
    <li>Paste the key in the sidebar</li>
    <li>Upload your PDF</li>
    <li>Click "Process Document"</li>
    </ol>
    <b>That's it! No payment, no credit card needed.</b>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not st.session_state.file_processed:
    st.info("👈 Click **🚀 Process Document** in the sidebar.")
    st.stop()


# ── Tabs ──
tab_summary, tab_keypoints, tab_guide, tab_chat = st.tabs(
    ["📝 Summary", "🎯 Key Points", "📖 Reading Guide", "💬 Chat"]
)
dd = st.session_state.document_data


# ═══════════════════════════
# TAB 1: SUMMARY
# ═══════════════════════════
with tab_summary:
    st.header("📝 Document Summary")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate Concise Summary", use_container_width=True, type="primary"):
            with st.spinner("Generating summary…"):
                try:
                    s = Summarizer(api_key=api_key, model=model_choice)
                    st.session_state.summary = s.summarize(dd.full_text, mode="concise")
                except Exception as e:
                    st.error(f"Failed: {e}")

    with c2:
        if st.button("Generate Detailed Summary", use_container_width=True):
            with st.spinner("Generating detailed summary…"):
                try:
                    s = Summarizer(api_key=api_key, model=model_choice)
                    st.session_state.detailed_summary = s.summarize(dd.full_text, mode="detailed")
                except Exception as e:
                    st.error(f"Failed: {e}")

    if st.session_state.summary:
        st.subheader("📋 Concise Summary")
        st.markdown(st.session_state.summary)

    if st.session_state.detailed_summary:
        st.subheader("📄 Detailed Summary")
        st.markdown(st.session_state.detailed_summary)

    st.markdown("---")
    if st.button("📑 Generate Section-wise Summary"):
        with st.spinner("Analyzing sections…"):
            try:
                rg_gen = ReadingGuideGenerator(api_key=api_key, model=model_choice)
                pdata = [{"page_number": p.page_number, "text": p.text, "word_count": p.word_count} for p in dd.pages]
                guide = rg_gen.generate(pdata)
                st.session_state.reading_guide = guide

                sec_data = []
                for sec in guide.sections:
                    sec_text = "\n".join(p.text for p in dd.pages if sec.start_page <= p.page_number <= sec.end_page)
                    sec_data.append({"title": sec.title, "text": sec_text})

                sm = Summarizer(api_key=api_key, model=model_choice)
                for ss in sm.summarize_by_sections(sec_data):
                    st.markdown(f'<div class="section-card"><b>{ss["title"]}</b><br>{ss["summary"]}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed: {e}")


# ═══════════════════════════
# TAB 2: KEY POINTS
# ═══════════════════════════
with tab_keypoints:
    st.header("🎯 Key Points")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Extract Key Points", use_container_width=True, type="primary"):
            with st.spinner("Extracting…"):
                try:
                    kp = KeyPointsExtractor(api_key=api_key, model=model_choice)
                    st.session_state.key_points = kp.extract(dd.full_text)
                except Exception as e:
                    st.error(f"Failed: {e}")

    with c2:
        if st.button("Extract with Priority Levels", use_container_width=True):
            with st.spinner("Extracting with priorities…"):
                try:
                    kp = KeyPointsExtractor(api_key=api_key, model=model_choice)
                    st.session_state.key_points_priority = kp.extract_with_priority(dd.full_text)
                except Exception as e:
                    st.error(f"Failed: {e}")

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


# ═══════════════════════════
# TAB 3: READING GUIDE
# ═══════════════════════════
with tab_guide:
    st.header("📖 Reading Guide")

    if st.button("Generate Reading Guide", use_container_width=True, type="primary"):
        with st.spinner("Analyzing structure…"):
            try:
                rg_gen = ReadingGuideGenerator(api_key=api_key, model=model_choice)
                pdata = [{"page_number": p.page_number, "text": p.text, "word_count": p.word_count} for p in dd.pages]
                st.session_state.reading_guide = rg_gen.generate(pdata)
            except Exception as e:
                st.error(f"Failed: {e}")

    guide = st.session_state.reading_guide
    if guide:
        m1, m2, m3 = st.columns(3)
        m1.metric("📄 Sections", len(guide.sections))
        m2.metric("⏱️ Reading Time", f"{guide.total_reading_time_min} min")
        m3.metric("🔴 Must-Read", sum(1 for s in guide.sections if "MUST" in s.importance.upper()))

        st.subheader("📋 Reading Strategy")
        st.info(guide.reading_strategy)

        if guide.skip_if_short:
            st.subheader("⏩ Skip If Short on Time")
            st.warning(", ".join(guide.skip_if_short))

        st.subheader("📑 Section Breakdown")
        for sec in guide.sections:
            imp = sec.importance.upper()
            icon = "🔴" if "MUST" in imp else "🟡" if "RECOMMEND" in imp else "🟢"
            css = "importance-must" if "MUST" in imp else "importance-rec" if "RECOMMEND" in imp else "importance-opt"
            st.markdown(
                f'<div class="section-card">{icon} <b>{sec.title}</b> '
                f'<span class="{css}">[{sec.importance}]</span><br>'
                f'Pages {sec.start_page}–{sec.end_page} · {sec.word_count:,} words · ~{sec.reading_time_min} min<br>'
                f'<i>{sec.description}</i></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════
# TAB 4: CHAT
# ═══════════════════════════
with tab_chat:
    st.header("💬 Chat with Your Document")

    if not st.session_state.rag_built:
        st.warning("Process a document first.")
        st.stop()

    # Suggested questions
    if st.session_state.suggested_questions:
        st.markdown("**💡 Suggested questions:**")
        cols = st.columns(min(len(st.session_state.suggested_questions), 3))
        for idx, q in enumerate(st.session_state.suggested_questions[:3]):
            if cols[idx].button(q, key=f"sq_{idx}", use_container_width=True):
                st.session_state.chat_messages.append({"role": "user", "content": q})
                resp = st.session_state.chatbot.ask(q)
                st.session_state.chat_messages.append({
                    "role": "assistant", "content": resp.content, "sources": resp.sources
                })
                st.rerun()

    st.markdown("---")

    # Chat history
    for msg in st.session_state.chat_messages:
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
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = st.session_state.chatbot.ask(prompt)
                st.markdown(resp.content)
                if resp.sources:
                    with st.expander("📄 Sources & Page References"):
                        for src in resp.sources:
                            st.markdown(
                                f'<span class="source-badge">Page {src["page"]}</span> {src["text"][:200]}…',
                                unsafe_allow_html=True,
                            )

        st.session_state.chat_messages.append({
            "role": "assistant", "content": resp.content, "sources": resp.sources
        })

    # Clear chat
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_history()
            st.rerun()