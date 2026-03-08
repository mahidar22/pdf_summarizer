"""
AI-Powered PDF Summarizer Modules
"""

# Import one by one so we see exactly which file fails
from .pdf_extractor import PDFExtractor
from .database import ChatDatabase

# These use langchain - import with error handling
try:
    from .ocr_engine import OCREngine
except ImportError:
    OCREngine = None

from .summarizer import Summarizer, create_llm
from .key_points import KeyPointsExtractor
from .reading_guide import ReadingGuideGenerator
from .vector_store import VectorStoreManager
from .rag_pipeline import RAGPipeline
from .chatbot import PDFChatbot
