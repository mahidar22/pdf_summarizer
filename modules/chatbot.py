"""
Chatbot Module - Gemini Compatible
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .rag_pipeline import RAGPipeline
from .database import ChatDatabase
from .summarizer import create_llm


@dataclass
class ChatMessage:
    role: str
    content: str
    sources: List[Dict] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PDFChatbot:
    """Interactive chatbot for Q&A over PDF content."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        session_id: str = "default",
        db_path: str = "data/chat_history.db",
        use_db: bool = True,
    ):
        self.rag = rag_pipeline
        self.session_id = session_id
        self.conversation_history: List[ChatMessage] = []
        self._chat_pairs: List[Tuple[str, str]] = []

        self.db: Optional[ChatDatabase] = None
        if use_db:
            try:
                self.db = ChatDatabase(db_path)
            except Exception:
                self.db = None

    def ask(self, question: str) -> ChatMessage:
        if not self.rag.is_built:
            return ChatMessage(
                role="assistant",
                content="⚠️ The document hasn't been processed yet. Please upload and process a PDF first.",
            )

        user_msg = ChatMessage(role="user", content=question)
        self.conversation_history.append(user_msg)

        try:
            result = self.rag.query(question, chat_history=self._chat_pairs)

            answer = result["answer"]
            sources = result["sources"]

            assistant_msg = ChatMessage(
                role="assistant", content=answer, sources=sources
            )
            self.conversation_history.append(assistant_msg)

            self._chat_pairs.append((question, answer))

            if self.db:
                self.db.save_message(self.session_id, "user", question)
                self.db.save_message(self.session_id, "assistant", answer)

            return assistant_msg

        except Exception as e:
            error_msg = ChatMessage(
                role="assistant",
                content=f"❌ Error generating answer: {str(e)}",
            )
            self.conversation_history.append(error_msg)
            return error_msg

    def get_history(self) -> List[ChatMessage]:
        return self.conversation_history

    def clear_history(self):
        self.conversation_history = []
        self._chat_pairs = []
        if self.db:
            self.db.clear_session(self.session_id)

    def get_suggested_questions(self, document_text: str) -> List[str]:
        try:
            llm = self.rag.llm
            prompt = (
                "Based on this document excerpt, suggest 5 interesting questions "
                "a reader might want to ask. Return only the questions, one per line.\n\n"
                f"Document:\n{document_text[:3000]}\n\n"
                "Suggested Questions:"
            )
            response = llm.invoke(prompt)
            questions = [
                q.strip().lstrip("0123456789.-) ")
                for q in response.content.strip().split("\n")
                if q.strip()
            ]
            return questions[:5]
        except Exception:
            return [
                "What is the main topic of this document?",
                "What are the key findings?",
                "Can you summarize the conclusion?",
                "What methodology was used?",
                "What are the main recommendations?",
            ]