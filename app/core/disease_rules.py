from typing import Iterable

from app.core.amdr import MacronutrientRatios

OBESITY_EER_DEFICIT_FACTOR = 0.90
DIABETES_CARB_MAX_RATIO = 0.60


def apply_obesity_deficit(eer: float) -> float:
    return eer * OBESITY_EER_DEFICIT_FACTOR


def apply_diabetes_carb_limit(carb_ratios: MacronutrientRatios) -> MacronutrientRatios:
    new_max = min(carb_ratios.max_ratio, DIABETES_CARB_MAX_RATIO)
    return MacronutrientRatios(min_ratio=carb_ratios.min_ratio, max_ratio=new_max)


def apply_disease_adjustments(
    eer: float, carb_ratios: MacronutrientRatios, diseases: Iterable[str]
) -> tuple[float, MacronutrientRatios]:
    normalized = {disease.lower() for disease in diseases}
    adjusted_eer = eer
    adjusted_carb_ratios = carb_ratios

    if "obesity" in normalized:
        adjusted_eer = apply_obesity_deficit(adjusted_eer)

    if "diabetes" in normalized:
        adjusted_carb_ratios = apply_diabetes_carb_limit(adjusted_carb_ratios)

    return adjusted_eer, adjusted_carb_ratios
