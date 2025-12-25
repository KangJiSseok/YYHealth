from typing import List, Optional
from pydantic import BaseModel


class ReportDayStat(BaseModel):
    date: str
    calories: Optional[float] = 0.0
    carbohydrate: Optional[float] = 0.0
    protein: Optional[float] = 0.0
    fat: Optional[float] = 0.0
    breakfastCalories: Optional[float] = 0.0
    lunchCalories: Optional[float] = 0.0
    dinnerCalories: Optional[float] = 0.0


class ReportAnalysisRequest(BaseModel):
    weekly: List[ReportDayStat] = []
    monthly: List[ReportDayStat] = []
    notes: List[str] = []
    diseases: List[str] = []
    kcal: Optional[float] = None
    protein: Optional[float] = None
    fat: Optional[float] = None
    carbohydrate: Optional[float] = None
    proteinMin: Optional[float] = None
    proteinMax: Optional[float] = None
    fatMin: Optional[float] = None
    fatMax: Optional[float] = None
    carbohydrateMin: Optional[float] = None
    carbohydrateMax: Optional[float] = None


class ReportAnalysisResponse(BaseModel):
    overall: str
    weekly: str
    monthly: str
    weeklyEvidence: List[dict] = []
    monthlyEvidence: List[dict] = []
