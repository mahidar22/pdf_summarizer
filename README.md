# 📄 AI-Powered PDF Summarizer with OCR, Smart Chatbot & RAG Pipeline

An advanced AI assistant that extracts text from any PDF (digital **or** scanned),
generates summaries, key points, a smart reading guide, and lets you
**chat** with the document using Retrieval-Augmented Generation (RAG).

---

## ✨ Features

| Feature | Description |
|---|---|
| 📝 Smart Summary | Concise, detailed, and section-wise summaries |
| 🎯 Key Points | Prioritized (Critical / Important / Nice-to-know) |
| 📖 Reading Guide | Section analysis, reading time, must-read flags |
| 💬 AI Chat | Q&A over the PDF with page references |
| 🔍 OCR | Scanned PDF / image support via Tesseract |
| 🧠 RAG Pipeline | ChromaDB + Sentence-Transformers + GPT |
| 🌐 Multi-language | Detect and summarize in any language |
| 📥 Download | Export summary as `.txt` |

---

## 🗂️ Project Structure

```
pdf-summarizer-advanced/
├── app.py                   # Streamlit application
├── requirements.txt
├── .env                     # API keys
├── modules/
│   ├── __init__.py
│   ├── pdf_extractor.py     # Digital PDF text extraction
│   ├── ocr_engine.py        # Tesseract OCR for scanned PDFs
│   ├── summarizer.py        # LLM summarization
│   ├── key_points.py        # Key points extraction
│   ├── reading_guide.py     # Reading guide generator
│   ├── vector_store.py      # ChromaDB operations
│   ├── rag_pipeline.py      # RAG pipeline
│   ├── chatbot.py           # Chat logic + memory
│   └── database.py          # SQLite chat history
├── data/
│   ├── uploads/             # Uploaded PDFs
│   └── vector_db/           # ChromaDB storage
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & enter the project

```bash
git clone <repo-url>
cd pdf-summarizer-advanced
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system dependencies

**Tesseract OCR** (needed only for scanned PDFs):

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler

# Windows → download installer from:
# https://github.com/tesseract-ocr/tesseract/wiki
```

### 5. Set your API key

Edit `.env`:

```
OPENAI_API_KEY=sk-your-key-here
```

Or enter it directly in the app sidebar.

### 6. Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| PDF Extraction | pdfplumber |
| OCR | pytesseract + pdf2image |
| LLM | OpenAI GPT-4o-mini / GPT-4o |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB |
| Orchestration | LangChain |
| Chat History | SQLite |

---

## 📸 Screenshots

> Upload a PDF → Process → explore the four tabs:
> **Summary** · **Key Points** · **Reading Guide** · **Chat**

---

## 🔧 Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `DEFAULT_MODEL` | `gpt-4o-mini` | LLM model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `TESSERACT_PATH` | `/usr/bin/tesseract` | Path to Tesseract binary |

---

## 📝 License

MIT