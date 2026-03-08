"""
Vector Store Module - Fixed Embedding Imports
"""

import os
import shutil
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Try importing ChromaDB ──
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# ── Try importing HuggingFace Embeddings (multiple paths) ──
HF_AVAILABLE = False
HFEmbeddings = None

# Try path 1: langchain-huggingface (newest)
try:
    from langchain_huggingface import HuggingFaceEmbeddings as HFE1
    HFEmbeddings = HFE1
    HF_AVAILABLE = True
except ImportError:
    pass

# Try path 2: langchain-community
if not HF_AVAILABLE:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as HFE2
        HFEmbeddings = HFE2
        HF_AVAILABLE = True
    except ImportError:
        pass

# Try path 3: direct sentence-transformers wrapper
if not HF_AVAILABLE:
    try:
        from sentence_transformers import SentenceTransformer
        
        class ManualHFEmbeddings:
            """Manual wrapper around sentence-transformers."""
            def __init__(self, model_name="all-MiniLM-L6-v2", **kwargs):
                self.model = SentenceTransformer(model_name)
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                return embeddings.tolist()
            
            def embed_query(self, text: str) -> List[float]:
                embedding = self.model.encode([text], normalize_embeddings=True)
                return embedding[0].tolist()
        
        HFEmbeddings = ManualHFEmbeddings
        HF_AVAILABLE = True
    except ImportError:
        pass

# ── Try importing OpenAI Embeddings ──
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMB_AVAILABLE = True
except ImportError:
    OPENAI_EMB_AVAILABLE = False


class VectorStoreManager:
    DEFAULT_PERSIST_DIR = "data/vector_db"
    DEFAULT_COLLECTION = "pdf_documents"

    def __init__(
        self,
        embedding_type: str = "huggingface",
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_api_key: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or self.DEFAULT_PERSIST_DIR
        os.makedirs(self.persist_directory, exist_ok=True)

        # Try OpenAI embeddings first if requested
        if embedding_type == "openai" and openai_api_key and OPENAI_EMB_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Try HuggingFace embeddings
        elif HF_AVAILABLE and HFEmbeddings is not None:
            try:
                self.embeddings = HFEmbeddings(
                    model_name=embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except TypeError:
                # ManualHFEmbeddings doesn't accept all kwargs
                self.embeddings = HFEmbeddings(model_name=embedding_model)
        
        else:
            raise ImportError(
                "No embedding backend available.\n"
                "Run these commands:\n"
                "  pip install sentence-transformers\n"
                "  pip install langchain-huggingface\n"
                "  pip install langchain-community"
            )

        self.vectorstore = None

    def create_from_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        self.clear()

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.DEFAULT_COLLECTION,
        )
        return self.vectorstore

    def load_existing(self):
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.DEFAULT_COLLECTION,
            )
            return self.vectorstore
        return None

    def query(self, query_text: str, k: int = 5) -> List[Document]:
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        return self.vectorstore.similarity_search(query_text, k=k)

    def query_with_scores(self, query_text: str, k: int = 5) -> List[tuple]:
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        return self.vectorstore.similarity_search_with_score(query_text, k=k)

    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        kwargs = search_kwargs or {"k": 5}
        return self.vectorstore.as_retriever(search_kwargs=kwargs)

    def clear(self):
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vectorstore = None

    def document_count(self) -> int:
        if self.vectorstore is None:
            return 0
        try:
            collection = self.vectorstore._collection
            return collection.count()
        except Exception:
            return 0