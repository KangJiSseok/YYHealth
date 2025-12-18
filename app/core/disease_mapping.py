"""
Disease name normalization.
Supports English canonical keys used in calculation/disease rules.
Includes common Korean labels for user inputs.

Canonical disease keys (used downstream):
- type2_diabetes
- prediabetes
- obesity
- metabolic_syndrome
- ascvd
- dyslipidemia
"""

from typing import Iterable, List

# Synonym map: lowercased input -> canonical disease key
DISEASE_SYNONYMS: dict[str, str] = {
    # Type 2 Diabetes
    "type 2 diabetes": "type2_diabetes",
    "type ii diabetes": "type2_diabetes",
    "t2d": "type2_diabetes",
    "diabetes": "type2_diabetes",
    "당뇨": "type2_diabetes",
    "당뇨병": "type2_diabetes",
    # Prediabetes
    "prediabetes": "prediabetes",
    "pre-diabetes": "prediabetes",
    "당뇨전단계": "prediabetes",
    # Obesity
    "obesity": "obesity",
    "obese": "obesity",
    "비만": "obesity",
    # Metabolic syndrome
    "metabolic syndrome": "metabolic_syndrome",
    "metabolic_syndrome": "metabolic_syndrome",
    "대사증후군": "metabolic_syndrome",
    # ASCVD
    "ascvd": "ascvd",
    "atherosclerotic cardiovascular disease": "ascvd",
    "죽상경화성심혈관질환": "ascvd",
    # Dyslipidemia
    "dyslipidemia": "dyslipidemia",
    "hyperlipidemia": "dyslipidemia",
    "hypercholesterolemia": "dyslipidemia",
    "이상지질혈증": "dyslipidemia",
}


def normalize_diseases(diseases: Iterable[str]) -> List[str]:
    normalized: set[str] = set()
    for name in diseases:
        key = name.strip().lower()
        if key in DISEASE_SYNONYMS:
            normalized.add(DISEASE_SYNONYMS[key])
        else:
            normalized.add(key)  # fallback to lowercased input
    return list(normalized)
