from app.core.amdr import MacronutrientRatios, calculate_macronutrients, default_ratios
from app.core.disease_rules import apply_disease_adjustments
from app.core.eer import EERDetails, calculate_eer_life_stage
from app.core.pa import resolve_pa
from app.schemas.nutrition import MacronutrientBreakdown, NutritionResult
from app.schemas.survey import SurveyRequest


def calculate_nutrition(survey: SurveyRequest) -> NutritionResult:
    """
    Steps:
    1. Validate input
    2. Resolve age group
    3. Resolve PA
    4. Calculate base EER
    5. Apply disease adjustments
    6. Calculate AMDR-based macronutrients
    7. Return NutritionResult
    """
    age_years = survey.age_years or 0
    age_months = survey.age_months or 0

    details: EERDetails = calculate_eer_life_stage(
        sex=survey.sex,
        age_years=age_years,
        age_months=age_months,
        height_cm=survey.height_cm,
        weight_kg=survey.weight_kg,
        activity_level=survey.activity_level,
        pa_resolver=resolve_pa,
        pregnancy_stage=survey.pregnancy_stage,
        lactation=survey.lactation,
    )

    base_carb_ratios: MacronutrientRatios = default_ratios()["carbohydrate"]

    adjusted_eer, carb_ratios = apply_disease_adjustments(
        eer=details.value, carb_ratios=base_carb_ratios, diseases=survey.diseases
    )

    macros = calculate_macronutrients(adjusted_eer, carb_ratios)

    macronutrients = {}
    for name, profile in macros.items():
        macronutrients[name] = MacronutrientBreakdown(
            recommended_g=profile.recommended_g,
            min_g=profile.min_g,
            max_g=profile.max_g,
            recommended_ratio=profile.recommended_ratio,
            min_ratio=profile.min_ratio,
            max_ratio=profile.max_ratio,
        )

    return NutritionResult(
        eer=details.value,
        adjusted_eer=adjusted_eer,
        pa=details.pa,
        macronutrients=macronutrients,
        life_stage=details.life_stage,
        formula_name=details.formula_name,
    )
