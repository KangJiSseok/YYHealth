import json
import re
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pdfplumber

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback for environments without tiktoken
    tiktoken = None


CHUNK_TYPES = {
    "prose_paragraph",
    "section_summary",
    "table",
    "figure_caption",
    "guideline_recommendation_box",
    # Derived types for filtering
    "guideline_recommendation",
    "numeric_evidence",
    "explanation",
    "reference",
}


@dataclass
class LayoutBlock:
    text: str
    chunk_type: str
    page_number: int
    section_heading: Optional[str] = None
    table_struct: Optional[dict] = None


def load_pdf(path: str) -> pdfplumber.PDF:
    print(f"[pdf_chunker] Loading PDF from: {path}")
    return pdfplumber.open(path)


def _tokenizer():
    if tiktoken:
        return tiktoken.get_encoding("cl100k_base")
    return None


def _count_tokens(text: str, encoder) -> int:
    if encoder:
        return len(encoder.encode(text))
    return len(text.split())


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        # Fallback for PDFs without blank-line separation
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts


def normalize_text(text: str) -> str:
    """
    Clean PDF-extracted text before paragraph splitting:
    - join hyphenated line breaks for Latin/digit words
    - replace single line breaks with spaces (keep paragraph breaks)
    - collapse excessive spaces
    """
    if not text:
        return ""
    cleaned = re.sub(r"([A-Za-z0-9])-\s*\n([A-Za-z0-9])", r"\1\2", text)
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = cleaned.replace(" \n", "\n").strip()
    return cleaned


def _detect_two_columns(page) -> bool:
    words = page.extract_words() or []
    if not words:
        return False
    mid_x = page.width / 2.0
    left = sum(1 for w in words if float(w["x0"]) < mid_x)
    right = sum(1 for w in words if float(w["x0"]) >= mid_x)
    total = left + right
    if total == 0:
        return False
    left_ratio = left / total
    right_ratio = right / total
    # Heuristic: both columns have at least 20% of words
    return left_ratio >= 0.2 and right_ratio >= 0.2


def _detect_heading(line: str) -> bool:
    words = line.strip().split()
    if not words:
        return False
    is_short = len(" ".join(words)) < 80
    many_caps = sum(1 for w in words if w.isupper()) >= max(2, len(words) // 2)
    looks_numbered = bool(re.match(r"^\d+(\.\d+)*\s", line))
    return is_short and (many_caps or looks_numbered)


def _chunk_text(
    text: str, min_tokens: int, max_tokens: int, overlap_tokens: int, encoder
) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = _count_tokens(sent, encoder)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sent)
        current_tokens += sent_tokens
        if current_tokens >= min_tokens:
            chunks.append(" ".join(current))
            if overlap_tokens > 0 and chunks:
                overlap_text = " ".join(current)
                overlap_token_count = _count_tokens(overlap_text, encoder)
                while overlap_token_count > overlap_tokens and current:
                    overlap_token_count -= _count_tokens(current.pop(0), encoder)
            current = []
            current_tokens = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def _parse_table(page, table_data: List[List[str]]) -> dict:
    headers = table_data[0] if table_data else []
    rows = table_data[1:] if len(table_data) > 1 else []
    structured_rows = []
    for row in rows:
        row_dict = {}
        for idx, cell in enumerate(row):
            header = headers[idx] if idx < len(headers) else f"col_{idx}"
            row_dict[header] = cell
        structured_rows.append(row_dict)
    return {"headers": headers, "rows": structured_rows}


def _summarize_table(structured_table: dict) -> str:
    try:
        headers = structured_table.get("headers", [])
        rows = structured_table.get("rows", [])
        preview = []
        for row in rows[:3]:
            preview.append(", ".join(f"{k}: {v}" for k, v in row.items()))
        return f"Table summary: headers={headers}; sample rows: " + " | ".join(preview)
    except Exception:
        return "Table summary unavailable"


REFERENCE_PATTERNS = [
    r"\bet al\.\b",
    r"\b(am j|ann med|clin nutr|obes|metab|diabetes obes metab|diabetes care)\b",
    r"\[[0-9]{1,3}\]",
]


def _is_reference_text(text: str, heading: str) -> bool:
    norm = normalize_text(text).lower()
    heading_lower = (heading or "").lower()
    if heading_lower.startswith("reference"):
        return True
    if norm.startswith("references"):
        return True
    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", norm))
    if year_hits >= 3 and ";" in norm:
        return True
    if any(re.search(pat, norm) for pat in REFERENCE_PATTERNS):
        return True
    return False


def _is_numeric_evidence(text: str) -> bool:
    norm = normalize_text(text).lower()
    has_pct = bool(re.search(r"\d+(\.\d+)?\s*%", norm))
    has_unit = bool(re.search(r"\d+(\.\d+)?\s*(kcal|mg|g|kg|mmol|mol|iu|µg)", norm))
    context_hits = any(
        kw in norm
        for kw in [
            "table",
            "figure",
            "fig.",
            "mean",
            "median",
            "trial",
            "randomized",
            "compared",
            "versus",
            "increase",
            "decrease",
            "reduction",
            "baseline",
        ]
    )
    return (has_pct or has_unit) and context_hits


def _classify_chunk_type(text: str, heading: str, base_type: str) -> str:
    if base_type == "table":
        return base_type
    norm = normalize_text(text).lower()
    heading_lower = (heading or "").lower()
    if _is_reference_text(norm, heading_lower):
        return "reference"
    if base_type == "guideline_recommendation_box":
        return "guideline_recommendation"
    if (
        ("recommend" in norm or "should" in norm or "recommended" in norm or "advise" in norm or "limit" in norm)
        and ("%" in norm or " g" in norm or "gram" in norm or re.search(r"\d+\s*[–-]\s*\d+% ", norm))
    ):
        return "guideline_recommendation"
    if _is_numeric_evidence(norm):
        return "numeric_evidence"
    return "explanation"


def _strip_reference_noise(text: str) -> str:
    if not text:
        return text
    cleaned = text
    patterns = [
        r"\b(am j|ann med|diabetes care|diabetes obes metab|clin nutr|nutrients)\b",
        r"\bet al\.\b",
        r"\bcorresponding author[s]?:?.*",
        r"\bdoi:\s*\S+",
        r"\bclinicaltrials\.gov\S*",
        r"\b\d{4};\d+\b",
        r"\[[0-9]{1,3}\]",
    ]
    for pat in patterns:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def extract_layout_blocks(pdf: pdfplumber.PDF, source_pdf: str) -> List[LayoutBlock]:
    print(f"[pdf_chunker] Extracting layout blocks from: {source_pdf}")
    blocks: List[LayoutBlock] = []
    for page_index, page in enumerate(pdf.pages):
        page_number = page_index + 1

        # Tables – skipped per requirement to exclude tables/images
        # try:
        #     table_objs = page.find_tables()
        # except Exception:
        #     table_objs = []
        # for table_obj in table_objs:
        #     data = table_obj.extract() or []
        #     struct = _parse_table(page, data)
        #     blocks.append(
        #         LayoutBlock(
        #             text=json.dumps(struct, ensure_ascii=False),
        #             chunk_type="table",
        #             page_number=page_number,
        #             section_heading=None,
        #             table_struct=struct,
        #         )
        #     )

        # Text (figure captions will be treated as prose and classified later; tables are skipped)
        is_two_col = _detect_two_columns(page)
        if is_two_col:
            mid = page.width / 2.0
            col_boxes = [(0, 0, mid, page.height), (mid, 0, page.width, page.height)]
        else:
            col_boxes = [(0, 0, page.width, page.height)]

        def add_paragraph_blocks(subpage, current_heading: Optional[str]) -> int:
            added = 0
            page_text = normalize_text(subpage.extract_text(layout=True) or "")
            paragraphs = _split_paragraphs(page_text)
            for para in paragraphs:
                lines = para.split("\n")
                if lines and _detect_heading(lines[0]):
                    current_heading = lines[0].strip()
                    para_body = "\n".join(lines[1:]).strip()
                    if not para_body:
                        continue
                    para = para_body
                para_lower = para.lower()
                if para_lower.startswith("figure"):
                    chunk_type = "figure_caption"
                elif any(line.strip().startswith(("-", "•", "▪")) for line in lines):
                    chunk_type = "guideline_recommendation_box"
                else:
                    chunk_type = "prose_paragraph"
                blocks.append(
                    LayoutBlock(
                        text=para,
                        chunk_type=chunk_type,
                        page_number=page_number,
                        section_heading=current_heading,
                    )
                )
                added += 1
            return added

        current_heading: Optional[str] = None
        text_added = 0
        for bbox in col_boxes:
            subpage = page.crop(bbox)
            text_added += add_paragraph_blocks(subpage, current_heading)

        # Fallback: if two-column heuristic failed to extract text, retry full page
        if text_added == 0 and is_two_col:
            text_added += add_paragraph_blocks(page, current_heading)

    print(f"[pdf_chunker] Found {len(blocks)} layout blocks")
    return blocks


def build_chunks(blocks: Iterable[LayoutBlock], source_pdf: str) -> List[Dict]:
    encoder = _tokenizer()
    chunks: List[Dict] = []
    for block in blocks:
        if block.chunk_type == "table":
            table_chunk = _base_chunk(block, source_pdf)
            table_chunk["chunk_type"] = "table"
            table_chunk["text"] = block.text
            chunks.append(table_chunk)

            summary_text = _summarize_table(block.table_struct or {})
            summary_chunk = _base_chunk(block, source_pdf)
            summary_chunk["chunk_type"] = "section_summary"
            summary_chunk["text"] = summary_text
            chunks.append(summary_chunk)
            continue

        if block.chunk_type == "figure_caption":
            text_parts = _chunk_text(
                block.text, min_tokens=80, max_tokens=150, overlap_tokens=0, encoder=encoder
            )
        elif block.chunk_type == "guideline_recommendation_box":
            text_parts = _chunk_text(
                block.text, min_tokens=80, max_tokens=180, overlap_tokens=0, encoder=encoder
            )
        else:
            text_parts = _chunk_text(
                block.text, min_tokens=120, max_tokens=220, overlap_tokens=30, encoder=encoder
            )
        for part in text_parts:
            chunk = _base_chunk(block, source_pdf)
            clean_text = _strip_reference_noise(part)
            if not clean_text.strip():
                continue
            chunk["text"] = clean_text
            new_type = _classify_chunk_type(clean_text, block.section_heading or "", block.chunk_type)
            if new_type == "reference":
                continue
            chunk["chunk_type"] = new_type
            chunks.append(chunk)
    print(f"[pdf_chunker] Built {len(chunks)} chunks from {len(blocks)} blocks")
    return chunks


def _base_chunk(block: LayoutBlock, source_pdf: str) -> Dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "source_pdf": source_pdf,
        "page_number": block.page_number,
        "section_heading": block.section_heading,
        "chunk_type": block.chunk_type,
        "text": "",
        "disease_tags": [],
        "nutrient_focus": [],
        "population": [],
        "evidence_type": "unknown",
        "safety_flags": [],
    }


def attach_metadata(chunks: List[Dict]) -> List[Dict]:
    print(f"[pdf_chunker] Attaching metadata to {len(chunks)} chunks (rule-based tags)")
    disease_keywords = {
        "type2_diabetes": [
            "diabetes",
            "type 2 diabetes",
            "type ii diabetes",
            "t2d",
            "glycemic",
            "hyperglycemia",
        ],
        "prediabetes": ["prediabetes", "impaired fasting glucose"],
        "obesity": ["obesity", "obese", "weight loss", "weight reduction"],
        "metabolic_syndrome": ["metabolic syndrome"],
        "ascvd": ["atherosclerotic", "cardiovascular", "ldl", "cholesterol"],
        "dyslipidemia": ["dyslipidemia", "hyperlipidemia", "hypercholesterolemia", "triglyceride"],
    }
    nutrient_keywords = {
        "carbohydrate": [
            "carbohydrate",
            "low carbohydrate",
            "low-carbohydrate",
            "carbohydrate-restricted",
            "glycemic",
            "glucose",
        ],
        "protein": ["protein"],
        "fat": ["fat", "lipid", "fatty acid"],
        "saturated_fat": ["saturated fat", "sfa"],
        "fiber": ["fiber"],
        "energy": ["calorie", "energy", "kcal"],
    }
    population_keywords = {
        "pediatric": ["child", "children", "adolescent", "pediatric"],
        "adult": ["adult", "patients", "people with", "individuals with"],
        "older": ["elderly", "older", "senior"],
        "pregnancy": ["pregnant", "pregnancy", "trimester"],
        "lactation": ["lactation", "breastfeeding"],
        "female": ["female", "women", "woman"],
        "male": ["male", "men", "man"],
        "korean": ["korean", "east asian"],
    }

    def match_tags(text: str, mapping: dict) -> List[str]:
        norm_text = normalize_text(text).lower()
        tags = []
        for tag, kws in mapping.items():
            if any(kw in norm_text for kw in kws):
                tags.append(tag)
        return tags

    filtered: List[Dict] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        heading = (chunk.get("section_heading") or "").strip().lower()
        norm_text = normalize_text(text).lower()
        year_hits = len(re.findall(r"\b(19|20)\d{2}\b", norm_text))

        # Exclude obvious reference sections or citation-only lines
        if heading.startswith("reference"):
            continue
        if norm_text.startswith("references"):
            continue
        if year_hits >= 3 and ";" in norm_text:
            continue
        chunk["disease_tags"] = match_tags(text, disease_keywords)
        chunk["nutrient_focus"] = match_tags(text, nutrient_keywords)
        chunk["population"] = match_tags(text, population_keywords)
        if chunk.get("evidence_type") is None:
            chunk["evidence_type"] = "unknown"
        # keep specialized chunk_type if already set, otherwise fallback to explanation
        if chunk.get("chunk_type") not in CHUNK_TYPES:
            chunk["chunk_type"] = "explanation"
        filtered.append(chunk)
    return filtered


def process_pdf(path: str) -> List[Dict]:
    """
    Convenience runner: load PDF, extract blocks, build chunks, attach metadata.
    """
    from pathlib import Path

    pdf_path = Path(path)
    pdf = load_pdf(str(pdf_path))
    source_pdf = pdf_path.name
    blocks = extract_layout_blocks(pdf, source_pdf)
    chunks = build_chunks(blocks, source_pdf)
    return attach_metadata(chunks)
