"""
Summarization Module - Uses Groq (FREE, no installation needed)
"""

from typing import List, Dict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Import providers ──
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langdetect import detect as detect_language
except ImportError:
    detect_language = None


# ── Prompts ──
CONCISE_PROMPT = PromptTemplate(
    template="Write a concise summary (150-300 words) of this document. Focus on main ideas and conclusions.\n\n{text}\n\nCONCISE SUMMARY:",
    input_variables=["text"],
)

DETAILED_PROMPT = PromptTemplate(
    template="Write a detailed summary of this document. Cover all topics, arguments, and conclusions.\n\n{text}\n\nDETAILED SUMMARY:",
    input_variables=["text"],
)

MAP_PROMPT = PromptTemplate(
    template="Write a concise summary of this section:\n\n{text}\n\nSECTION SUMMARY:",
    input_variables=["text"],
)

COMBINE_PROMPT = PromptTemplate(
    template="Combine these section summaries into one coherent summary:\n\n{text}\n\nFINAL SUMMARY:",
    input_variables=["text"],
)

SECTION_PROMPT_TEMPLATE = "Summarize this section titled '{section_title}' in 2-4 sentences:\n\n{text}\n\nSECTION SUMMARY:"

MULTILANG_PROMPT = PromptTemplate(
    template="This document is in {language}. Summarize in {output_language}.\n\n{text}\n\nSUMMARY:",
    input_variables=["text", "language", "output_language"],
)


def create_llm(api_key: str, model: str = "auto", temperature: float = 0.3):
    """
    Create LLM based on API key.
    gsk_... → Groq (FREE)
    AIza... → Gemini (FREE)
    sk-...  → OpenAI (PAID)
    """

    # GROQ (FREE)
    if api_key.startswith("gsk_") and GROQ_AVAILABLE:
        groq_models = {
            "llama-3.1-8b-instant": "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768": "mixtral-8x7b-32768",
            "gemma2-9b-it": "gemma2-9b-it",
        }
        if model in groq_models:
            model_name = model
        else:
            model_name = "llama-3.1-8b-instant"
        
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )

    # GEMINI (FREE)
    if api_key.startswith("AIza") and GEMINI_AVAILABLE:
        model_name = model if "gemini" in model else "gemini-2.0-flash"
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )

    # OPENAI (PAID)
    if api_key.startswith("sk-") and OPENAI_AVAILABLE:
        model_name = model if "gpt" in model else "gpt-4o-mini"
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )

    # Try Groq as default
    if GROQ_AVAILABLE:
        return ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=temperature,
        )

    raise ValueError(
        "Cannot create LLM!\n"
        "Get FREE Groq key: https://console.groq.com/keys\n"
        "Key should start with gsk_"
    )


class Summarizer:
    CHUNK_SIZE = 3500
    CHUNK_OVERLAP = 300

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.3):
        self.llm = create_llm(api_key, model, temperature)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP,
        )

    def _text_to_docs(self, text: str) -> List[Document]:
        chunks = self.splitter.split_text(text)
        return [Document(page_content=c) for c in chunks]

    def summarize(self, text: str, mode: str = "concise") -> str:
        docs = self._text_to_docs(text)

        if len(docs) == 1:
            prompt = CONCISE_PROMPT if mode == "concise" else DETAILED_PROMPT
            response = self.llm.invoke(prompt.format(text=text[:7000]))
            return response.content.strip()

        chunk_summaries = []
        for doc in docs[:6]:
            response = self.llm.invoke(MAP_PROMPT.format(text=doc.page_content))
            chunk_summaries.append(response.content.strip())

        combined = "\n\n".join(chunk_summaries)
        response = self.llm.invoke(COMBINE_PROMPT.format(text=combined[:7000]))
        return response.content.strip()

    def summarize_by_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        results = []
        for sec in sections:
            title = sec.get("title", "Untitled")
            text = sec.get("text", "")
            if len(text.strip()) < 30:
                results.append({"title": title, "summary": "(Too short.)"})
                continue
            prompt = SECTION_PROMPT_TEMPLATE.format(section_title=title, text=text[:5000])
            response = self.llm.invoke(prompt)
            results.append({"title": title, "summary": response.content.strip()})
        return results

    def summarize_multilanguage(self, text: str, output_language: str = "English") -> str:
        input_lang = "Unknown"
        if detect_language:
            try:
                input_lang = detect_language(text[:2000])
            except Exception:
                pass
        summary = self.summarize(text, mode="concise")
        response = self.llm.invoke(MULTILANG_PROMPT.format(
            text=summary, language=input_lang, output_language=output_language))
        return response.content.strip()

    @staticmethod
    def detect_doc_language(text: str) -> str:
        if detect_language is None:
            return "Unknown"
        try:
            return detect_language(text[:3000])
        except Exception:
            return "Unknown"