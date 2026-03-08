"""
RAG Pipeline Module - Gemini Compatible
"""

from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vector_store import VectorStoreManager
from .summarizer import create_llm


QA_PROMPT = PromptTemplate(
    template="""You are a helpful AI assistant answering questions about a PDF document.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I couldn't find this information in the document."

Always mention the page number(s) where you found the information using the format [Page X].

Context:
{context}

Question: {question}

Helpful Answer (with page references):""",
    input_variables=["context", "question"],
)


class RAGPipeline:
    """Full RAG pipeline: chunk → embed → store → retrieve → generate."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        embedding_type: str = "huggingface",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_dir: str = "data/vector_db",
    ):
        self.api_key = api_key
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.llm = create_llm(api_key, model, temperature=0.2)

        self.vector_store_manager = VectorStoreManager(
            embedding_type=embedding_type,
            embedding_model=embedding_model,
            openai_api_key=api_key if embedding_type == "openai" else None,
            persist_directory=persist_dir,
        )

        self._built = False

    def build(self, pages_data: List[Dict]) -> int:
        documents: List[Document] = []
        for page in pages_data:
            text = page.get("text", "").strip()
            if not text:
                continue
            doc = Document(
                page_content=text,
                metadata={
                    "page_number": page.get("page_number", 0),
                    "source": f"Page {page.get('page_number', 0)}",
                },
            )
            documents.append(doc)

        if not documents:
            raise ValueError("No text content found in the document.")

        self.vector_store_manager.create_from_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self._built = True
        return self.vector_store_manager.document_count()

    def query(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict:
        if not self._built:
            raise RuntimeError("Pipeline not built. Call build() first.")

        # Retrieve relevant chunks
        docs = self.vector_store_manager.query(question, k=5)

        # Build context from retrieved docs
        context_parts = []
        for doc in docs:
            page_num = doc.metadata.get("page_number", "?")
            context_parts.append(f"[Page {page_num}]: {doc.page_content}")
        
        context = "\n\n".join(context_parts)

        # Handle conversation history for follow-up questions
        history_text = ""
        if chat_history:
            history_parts = []
            for human, ai in chat_history[-3:]:  # Last 3 exchanges
                history_parts.append(f"Human: {human}\nAssistant: {ai}")
            history_text = "\n".join(history_parts)

        # Build prompt
        if history_text:
            full_prompt = (
                f"Previous conversation:\n{history_text}\n\n"
                f"{QA_PROMPT.format(context=context, question=question)}"
            )
        else:
            full_prompt = QA_PROMPT.format(context=context, question=question)

        # Generate answer
        response = self.llm.invoke(full_prompt)

        # Extract sources
        sources = []
        seen_pages = set()
        for doc in docs:
            page = doc.metadata.get("page_number", 0)
            if page not in seen_pages:
                sources.append(
                    {
                        "page": page,
                        "text": doc.page_content[:300] + "..."
                        if len(doc.page_content) > 300
                        else doc.page_content,
                    }
                )
                seen_pages.add(page)

        return {
            "answer": response.content.strip(),
            "sources": sorted(sources, key=lambda x: x["page"]),
        }

    def get_relevant_chunks(self, question: str, k: int = 5) -> List[Dict]:
        docs = self.vector_store_manager.query(question, k=k)
        return [
            {
                "page": d.metadata.get("page_number", 0),
                "text": d.page_content,
            }
            for d in docs
        ]

    @property
    def is_built(self) -> bool:
        return self._built

    def clear(self):
        self.vector_store_manager.clear()
        self._built = False