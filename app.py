import streamlit as st
import os

st.set_page_config(page_title="AI PDF Summarizer", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# Create folders
os.makedirs("data/uploads", exist_ok=True)

# ── Safe imports ──
errors = []

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False
    errors.append("pdfplumber")

try:
    from langchain_groq import ChatGroq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False
    errors.append("langchain-groq")

# ── CSS ──
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem; font-weight: 700;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; padding: 1rem 0;
}
.card {
    background: #f8f9fa; border-radius: 10px; padding: 1rem;
    text-align: center; border: 1px solid #e9ecef; margin: 0.3rem 0;
}
.section-card {
    background: white; border-left: 4px solid #667eea;
    padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.source-badge {
    background: #e8f0fe; color: #1967d2; padding: 2px 8px;
    border-radius: 12px; font-size: 0.8rem;
}
.green-box {
    background: #e6f4ea; border: 1px solid #34a853;
    border-radius: 8px; padding: 0.7rem; text-align: center;
}
.blue-box {
    background: #f0f7ff; border: 1px solid #c2dbff;
    border-radius: 10px; padding: 1.2rem; margin: 1rem 0;
}
[data-testid="stSidebar"] { min-width: 300px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.pages = []
    st.session_state.full_text = ""
    st.session_state.file_name = ""
    st.session_state.summary = None
    st.session_state.detailed_summary = None
    st.session_state.key_points = None
    st.session_state.priority_points = None
    st.session_state.guide = None
    st.session_state.messages = []
    st.session_state.chat_pairs = []
    st.session_state.current_file = None


# ══════════════════════════════
# FUNCTIONS
# ══════════════════════════════
def extract_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page_number": i + 1, "text": text.strip(), "words": len(text.split())})
    full = "\n\n".join(p["text"] for p in pages if p["text"])
    return pages, full


def ask_llm(llm, prompt):
    response = llm.invoke(prompt)
    return response.content.strip()


def make_summary(llm, text, mode="concise"):
    if mode == "concise":
        p = f"Write a concise summary (150-300 words). Main ideas and conclusions only.\n\n{text[:7000]}\n\nSUMMARY:"
    else:
        p = f"Write a detailed, thorough summary. Cover all topics.\n\n{text[:7000]}\n\nDETAILED SUMMARY:"
    return ask_llm(llm, p)


def make_key_points(llm, text):
    p = f"Extract 5-15 most important key points. One sentence each. Numbered list.\n\n{text[:7000]}\n\nKEY POINTS:"
    result = ask_llm(llm, p)
    points = []
    for line in result.split("\n"):
        line = line.strip().lstrip("0123456789.-•*) ").strip()
        if line and len(line) > 5:
            points.append(line)
    return points


def make_priority_points(llm, text):
    p = f"""Extract key points with priority:
- CRITICAL: must know
- IMPORTANT: should know
- NICE-TO-KNOW: extra info

Document:
{text[:7000]}

Format each as: PRIORITY: point
"""
    result = ask_llm(llm, p)
    items = []
    for line in result.split("\n"):
        line = line.strip().lstrip("0123456789.-•*) ").strip()
        if not line or len(line) < 5:
            continue
        priority = "NICE-TO-KNOW"
        upper = line.upper()
        if "CRITICAL" in upper:
            priority = "CRITICAL"
        elif "IMPORTANT" in upper:
            priority = "IMPORTANT"
        clean = line
        for tag in ["CRITICAL:", "IMPORTANT:", "NICE-TO-KNOW:", "CRITICAL", "IMPORTANT", "NICE-TO-KNOW"]:
            clean = clean.replace(tag, "").replace(tag.lower(), "").replace(tag.title(), "")
        clean = clean.strip().strip(":").strip()
        if clean and len(clean) > 5:
            items.append({"point": clean, "priority": priority})
    return items


def make_guide(llm, text, pages):
    words = sum(p["words"] for p in pages)
    time_min = round(words / 220, 1)
    p = f"""Analyze this document and provide:
1. Reading strategy (how to approach it)
2. Key sections with importance (MUST-READ / RECOMMENDED / OPTIONAL)
3. What to skip if short on time
4. Estimated reading time per section

Document ({len(pages)} pages, {words} words):
{text[:5000]}

Provide a complete reading guide:"""
    result = ask_llm(llm, p)
    return {"analysis": result, "words": words, "time": time_min, "pages": len(pages)}


def chat_answer(llm, question, pages, history):
    # Simple search: find pages with matching words
    q_words = set(question.lower().split())
    scored = []
    for p in pages:
        overlap = len(q_words.intersection(set(p["text"].lower().split())))
        scored.append((overlap, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:4]

    context = "\n\n".join(f"[Page {p['page_number']}]: {p['text'][:500]}" for _, p in top)

    hist = ""
    if history:
        hist = "Previous Q&A:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in history[-2:]) + "\n\n"

    prompt = f"""{hist}Answer using ONLY this context. Say [Page X] for references.
If not found, say "I couldn't find this in the document."

Context:
{context}

Question: {question}

Answer:"""

    answer = ask_llm(llm, prompt)
    sources = [{"page": p["page_number"], "text": p["text"][:200]} for _, p in top[:3]]
    return answer, sources


# ══════════════════════════════
# SIDEBAR
# ══════════════════════════════
with st.sidebar:
    st.markdown("## 📄 AI PDF Summarizer")
    st.markdown("🆓 **100% FREE**")
    st.markdown("---")

    if errors:
        st.error(f"Missing: {', '.join(errors)}")
        st.code(f"pip install {' '.join(errors)}")

    # API Key
    st.markdown("### 🔑 API Key")
    saved_key = os.getenv("GROQ_API_KEY", "")
    if saved_key:
        api_key = saved_key
        st.markdown('<div class="green-box">✅ Key loaded from environment</div>', unsafe_allow_html=True)
    else:
        api_key = st.text_input("Groq API Key", type="password")
        if not api_key:
            st.info("Get FREE key: [console.groq.com/keys](https://console.groq.com/keys)")

    # Model
    model = st.selectbox("🧠 Model", [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])

    st.markdown("---")

    # Upload
    st.markdown("### 📁 Upload PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    # Process
    if uploaded and api_key and GROQ_OK and PDF_OK:
        if st.session_state.current_file != uploaded.name:
            st.session_state.processed = False
            st.session_state.summary = None
            st.session_state.detailed_summary = None
            st.session_state.key_points = None
            st.session_state.priority_points = None
            st.session_state.guide = None
            st.session_state.messages = []
            st.session_state.chat_pairs = []
            st.session_state.current_file = uploaded.name

        if st.button("🚀 Process Document", use_container_width=True, type="primary"):
            path = os.path.join("data/uploads", uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner("📖 Reading PDF..."):
                pages, full_text = extract_pdf(path)

            if full_text:
                st.session_state.pages = pages
                st.session_state.full_text = full_text
                st.session_state.file_name = uploaded.name
                st.session_state.processed = True
                st.success(f"✅ Done! {len(pages)} pages, {sum(p['words'] for p in pages):,} words")
            else:
                st.error("❌ No text found in PDF!")

    # Doc info
    if st.session_state.processed:
        st.markdown("---")
        st.markdown("### 📊 Info")
        st.markdown(f"**{st.session_state.file_name}**")
        st.markdown(f"Pages: {len(st.session_state.pages)}")
        words = sum(p["words"] for p in st.session_state.pages)
        st.markdown(f"Words: {words:,}")
        st.markdown(f"~{round(words/220, 1)} min read")


# ══════════════════════════════
# MAIN AREA
# ══════════════════════════════

# Welcome screen
if not uploaded or not api_key:
    st.markdown('<div class="main-header">📄 AI-Powered PDF Summarizer</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="card">📝<br><b>Summary</b><br>Concise & Detailed</div>', unsafe_allow_html=True)
    c2.markdown('<div class="card">🎯<br><b>Key Points</b><br>Prioritized</div>', unsafe_allow_html=True)
    c3.markdown('<div class="card">📖<br><b>Reading Guide</b><br>Time Estimates</div>', unsafe_allow_html=True)
    c4.markdown('<div class="card">💬<br><b>AI Chat</b><br>Ask Anything</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="blue-box">
    <h3>🚀 Quick Start</h3>
    <ol>
    <li>Get FREE API key → <a href="https://console.groq.com/keys" target="_blank"><b>console.groq.com/keys</b></a></li>
    <li>Paste key in sidebar (left side) or set as environment variable</li>
    <li>Upload your PDF</li>
    <li>Click Process Document</li>
    </ol>
    <p><b>✅ Free forever • No credit card needed</b></p>
    </div>
    """, unsafe_allow_html=True)

    if errors:
        st.error(f"⚠️ Missing packages: {', '.join(errors)}")

    st.stop()

if not st.session_state.processed:
    st.info("👈 Upload a PDF and click **🚀 Process Document**")
    st.stop()

# ══════════════════════════════
# TABS
# ══════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🎯 Key Points", "📖 Reading Guide", "💬 Chat"])

# Create LLM
try:
    llm = ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.3)
except Exception as e:
    st.error(f"Cannot connect to AI: {e}")
    st.stop()

# ── TAB 1: SUMMARY ──
with tab1:
    st.header("📝 Document Summary")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✨ Concise Summary", use_container_width=True, type="primary"):
            with st.spinner("Generating..."):
                try:
                    st.session_state.summary = make_summary(llm, st.session_state.full_text, "concise")
                except Exception as e:
                    st.error(f"Error: {e}")
    with c2:
        if st.button("📄 Detailed Summary", use_container_width=True):
            with st.spinner("Generating..."):
                try:
                    st.session_state.detailed_summary = make_summary(llm, st.session_state.full_text, "detailed")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.summary:
        st.subheader("Concise Summary")
        st.markdown(st.session_state.summary)
        st.download_button("📥 Download", st.session_state.summary, "summary.txt")

    if st.session_state.detailed_summary:
        st.subheader("Detailed Summary")
        st.markdown(st.session_state.detailed_summary)


# ── TAB 2: KEY POINTS ──
with tab2:
    st.header("🎯 Key Points")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎯 Extract Key Points", use_container_width=True, type="primary"):
            with st.spinner("Extracting..."):
                try:
                    st.session_state.key_points = make_key_points(llm, st.session_state.full_text)
                except Exception as e:
                    st.error(f"Error: {e}")
    with c2:
        if st.button("🏷️ With Priority Levels", use_container_width=True):
            with st.spinner("Extracting..."):
                try:
                    st.session_state.priority_points = make_priority_points(llm, st.session_state.full_text)
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.key_points:
        for i, p in enumerate(st.session_state.key_points, 1):
            st.markdown(f"**{i}.** {p}")

    if st.session_state.priority_points:
        st.markdown("---")
        st.subheader("Prioritized")
        for item in st.session_state.priority_points:
            pr = item["priority"]
            emoji = {"CRITICAL": "🔴", "IMPORTANT": "🟡", "NICE-TO-KNOW": "🟢"}.get(pr, "⚪")
            st.markdown(f"{emoji} **[{pr}]** {item['point']}")


# ── TAB 3: READING GUIDE ──
with tab3:
    st.header("📖 Reading Guide")

    if st.button("📖 Generate Reading Guide", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            try:
                st.session_state.guide = make_guide(llm, st.session_state.full_text, st.session_state.pages)
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.guide:
        g = st.session_state.guide
        c1, c2, c3 = st.columns(3)
        c1.metric("📄 Pages", g["pages"])
        c2.metric("📝 Words", f"{g['words']:,}")
        c3.metric("⏱️ Read Time", f"{g['time']} min")

        st.markdown("---")
        st.markdown(g["analysis"])


# ── TAB 4: CHAT ──
with tab4:
    st.header("💬 Chat with Your Document")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for s in msg["sources"]:
                        st.markdown(f'<span class="source-badge">Page {s["page"]}</span> {s["text"][:150]}...', unsafe_allow_html=True)

    # Input
    if question := st.chat_input("Ask anything about the document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = chat_answer(llm, question, st.session_state.pages, st.session_state.chat_pairs)
                    st.markdown(answer)
                    if sources:
                        with st.expander("📄 Sources"):
                            for s in sources:
                                st.markdown(f'<span class="source-badge">Page {s["page"]}</span> {s["text"][:150]}...', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    st.session_state.chat_pairs.append((question, answer))
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_pairs = []
            st.rerun()