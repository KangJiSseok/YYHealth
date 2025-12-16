from fastapi import FastAPI

from app.core.calculator import calculate_nutrition
from app.schemas.nutrition import NutritionResult
from app.schemas.survey import SurveyRequest

app = FastAPI()


@app.post("/nutrition", response_model=NutritionResult)
def compute_nutrition(survey: SurveyRequest) -> NutritionResult:
    return calculate_nutrition(survey)
