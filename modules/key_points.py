"""
Key Points Extraction Module - Gemini Compatible
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .summarizer import create_llm


EXTRACT_PROMPT = """Analyze the following document text and extract the most important key points.

Rules:
1. Return between 5 and 15 key points.
2. Each point should be a single, clear sentence.
3. Order them by importance (most important first).
4. Cover different aspects/topics of the document.
5. Be specific — avoid vague statements.

Document:
{text}

Return ONLY the key points as a numbered list (1. 2. 3. ...):
"""

EXTRACT_WITH_PRIORITY_PROMPT = """Analyze the following document text and extract key points with priority levels.

For each key point, assign a priority:
- 🔴 CRITICAL — must know
- 🟡 IMPORTANT — should know
- 🟢 NICE-TO-KNOW — supplementary

Document:
{text}

Return as a numbered list with priority emoji prefix. Example:
1. 🔴 The main conclusion is ...
2. 🟡 The study found that ...
3. 🟢 A minor finding was ...
"""

MERGE_PROMPT = """Below are key points extracted from different sections of a document.
Merge, deduplicate, and rank them into a single list of the top 10-15 key points.

{text}

Return ONLY the merged, ranked key points as a numbered list:
"""


class KeyPointsExtractor:
    """Extracts key points from document text."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.llm = create_llm(api_key, model, temperature=0.2)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=500
        )

    def extract(self, text: str) -> List[str]:
        chunks = self.splitter.split_text(text)

        if len(chunks) == 1:
            prompt = EXTRACT_PROMPT.format(text=text[:8000])
            response = self.llm.invoke(prompt)
            return self._parse_points(response.content)

        all_points: List[str] = []
        for chunk in chunks:
            prompt = EXTRACT_PROMPT.format(text=chunk)
            response = self.llm.invoke(prompt)
            all_points.extend(self._parse_points(response.content))

        combined = "\n".join(f"- {p}" for p in all_points)
        merge_prompt = MERGE_PROMPT.format(text=combined)
        response = self.llm.invoke(merge_prompt)
        return self._parse_points(response.content)

    def extract_with_priority(self, text: str) -> List[Dict[str, str]]:
        chunks = self.splitter.split_text(text)
        text_for_prompt = chunks[0] if len(chunks) == 1 else "\n\n".join(chunks[:3])

        prompt = EXTRACT_WITH_PRIORITY_PROMPT.format(text=text_for_prompt[:8000])
        response = self.llm.invoke(prompt)

        results = []
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for i, ch in enumerate(line):
                if ch == "." and i < 4:
                    line = line[i + 1:].strip()
                    break

            priority = "NICE-TO-KNOW"
            if "🔴" in line:
                priority = "CRITICAL"
                line = line.replace("🔴", "").strip()
            elif "🟡" in line:
                priority = "IMPORTANT"
                line = line.replace("🟡", "").strip()
            elif "🟢" in line:
                priority = "NICE-TO-KNOW"
                line = line.replace("🟢", "").strip()

            if line:
                results.append({"point": line, "priority": priority})
        return results

    @staticmethod
    def _parse_points(text: str) -> List[str]:
        points = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = line.lstrip("0123456789.-•) ").strip()
            if cleaned:
                points.append(cleaned)
        return points