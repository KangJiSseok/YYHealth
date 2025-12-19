from typing import Dict, List, Optional

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    disease: str
    source_pdf: Optional[str]
    page_number: Optional[int]
    chunk_type: Optional[str]
    text: Optional[str]
    score: Optional[float]


class NutritionWithEvidence(BaseModel):
    calculation: dict
    evidence: List[EvidenceItem]
    rationale: List[str] = []
    summaries: Dict[str, str] = {}
    calc_summaries: Dict[str, str] = {}
