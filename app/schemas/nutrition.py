from typing import Dict

from pydantic import BaseModel


class MacronutrientBreakdown(BaseModel):
    recommended_g: float
    min_g: float
    max_g: float
    recommended_ratio: float
    min_ratio: float
    max_ratio: float


class NutritionResult(BaseModel):
    eer: float
    adjusted_eer: float
    pa: float
    macronutrients: Dict[str, MacronutrientBreakdown]
    life_stage: str
    formula_name: str
