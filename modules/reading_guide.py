"""
Reading Guide Module - Gemini Compatible
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from .summarizer import create_llm


WORDS_PER_MINUTE = 220

STRUCTURE_PROMPT = """Analyze the following document text and identify its structure.

For each section/chapter you find, provide:
- Section title
- A one-sentence description
- Importance: MUST-READ / RECOMMENDED / OPTIONAL
- Suggested reading order number

Also provide:
- An overall reading strategy (how to approach this document)
- Which sections to skip if the reader is short on time

Document (first ~6000 chars):
{text}

Return your analysis in this exact format:

READING STRATEGY:
<your strategy here>

SECTIONS:
1. [Title] | [Description] | [Importance] | [Order]
2. [Title] | [Description] | [Importance] | [Order]
...

SKIP IF SHORT ON TIME:
<comma-separated section titles>
"""


@dataclass
class Section:
    title: str
    start_page: int
    end_page: int
    word_count: int
    reading_time_min: float
    importance: str = "RECOMMENDED"
    description: str = ""
    order: int = 0


@dataclass
class ReadingGuide:
    sections: List[Section]
    total_reading_time_min: float
    reading_strategy: str
    skip_if_short: List[str]
    document_type: str = "General"


class ReadingGuideGenerator:
    """Generates a smart reading guide for a document."""

    HEADING_PATTERNS = [
        r"^(Chapter|CHAPTER)\s+\d+",
        r"^(Section|SECTION)\s+\d+",
        r"^\d+\.\s+[A-Z]",
        r"^\d+\.\d+\s+[A-Z]",
        r"^(Part|PART)\s+[IVXLCDM\d]+",
        r"^(Introduction|Conclusion|Abstract|Summary|References|Bibliography|Appendix)",
        r"^[A-Z][A-Z\s]{4,50}$",
    ]

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.llm = None
        if api_key:
            try:
                self.llm = create_llm(api_key, model, temperature=0.2)
            except Exception:
                pass

    def generate(self, pages_data: List[Dict]) -> ReadingGuide:
        sections = self._detect_sections(pages_data)

        for sec in sections:
            sec.reading_time_min = round(sec.word_count / WORDS_PER_MINUTE, 1)

        total_time = sum(s.reading_time_min for s in sections)

        reading_strategy = "Read the document from beginning to end."
        skip_list: List[str] = []

        if self.llm:
            full_text = "\n".join(p.get("text", "") for p in pages_data[:30])
            try:
                llm_analysis = self._llm_analyze(full_text, sections)
                reading_strategy = llm_analysis.get("strategy", reading_strategy)
                skip_list = llm_analysis.get("skip", [])
                for sec in sections:
                    for llm_sec in llm_analysis.get("sections", []):
                        if self._titles_match(sec.title, llm_sec.get("title", "")):
                            sec.importance = llm_sec.get("importance", sec.importance)
                            sec.description = llm_sec.get("description", "")
                            break
            except Exception:
                pass

        return ReadingGuide(
            sections=sections,
            total_reading_time_min=round(total_time, 1),
            reading_strategy=reading_strategy,
            skip_if_short=skip_list,
        )

    def _detect_sections(self, pages_data: List[Dict]) -> List[Section]:
        sections: List[Section] = []
        current_title = "Introduction"
        current_start = 1
        current_words = 0

        for page in pages_data:
            text = page.get("text", "")
            page_num = page.get("page_number", 1)
            lines = text.split("\n")

            for line in lines:
                stripped = line.strip()
                if self._is_heading(stripped):
                    if current_words > 0:
                        sections.append(
                            Section(
                                title=current_title,
                                start_page=current_start,
                                end_page=page_num,
                                word_count=current_words,
                                reading_time_min=0,
                            )
                        )
                    current_title = stripped[:80]
                    current_start = page_num
                    current_words = 0
                else:
                    current_words += len(stripped.split())

        if current_words > 0:
            last_page = pages_data[-1].get("page_number", 1) if pages_data else 1
            sections.append(
                Section(
                    title=current_title,
                    start_page=current_start,
                    end_page=last_page,
                    word_count=current_words,
                    reading_time_min=0,
                )
            )

        if not sections:
            total_words = sum(p.get("word_count", 0) for p in pages_data)
            last_page = pages_data[-1].get("page_number", 1) if pages_data else 1
            sections.append(
                Section(
                    title="Full Document",
                    start_page=1,
                    end_page=last_page,
                    word_count=total_words,
                    reading_time_min=0,
                )
            )

        for i, sec in enumerate(sections):
            sec.order = i + 1

        return sections

    def _is_heading(self, text: str) -> bool:
        if not text or len(text) > 100 or len(text) < 3:
            return False
        if len(text.split()) > 12:
            return False
        for pattern in self.HEADING_PATTERNS:
            if re.match(pattern, text):
                return True
        return False

    def _llm_analyze(self, text: str, sections: List[Section]) -> Dict:
        prompt = STRUCTURE_PROMPT.format(text=text[:6000])
        response = self.llm.invoke(prompt)
        return self._parse_llm_response(response.content)

    @staticmethod
    def _parse_llm_response(content: str) -> Dict:
        result: Dict = {"strategy": "", "sections": [], "skip": []}
        lines = content.strip().split("\n")
        mode = None

        for line in lines:
            stripped = line.strip()
            if "READING STRATEGY" in stripped.upper():
                mode = "strategy"
                continue
            elif "SECTIONS:" in stripped.upper():
                mode = "sections"
                continue
            elif "SKIP IF SHORT" in stripped.upper():
                mode = "skip"
                continue

            if mode == "strategy" and stripped:
                result["strategy"] += stripped + " "
            elif mode == "sections" and "|" in stripped:
                parts = [p.strip() for p in stripped.split("|")]
                if len(parts) >= 3:
                    title = parts[0].lstrip("0123456789. ").strip()
                    result["sections"].append(
                        {
                            "title": title,
                            "description": parts[1] if len(parts) > 1 else "",
                            "importance": parts[2] if len(parts) > 2 else "RECOMMENDED",
                        }
                    )
            elif mode == "skip" and stripped:
                result["skip"] = [s.strip() for s in stripped.split(",")]

        result["strategy"] = result["strategy"].strip()
        return result

    @staticmethod
    def _titles_match(t1: str, t2: str) -> bool:
        return t1.lower().strip()[:30] == t2.lower().strip()[:30]

    @staticmethod
    def estimate_reading_time(word_count: int) -> float:
        return round(word_count / WORDS_PER_MINUTE, 1)