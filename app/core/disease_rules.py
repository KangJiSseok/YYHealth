from typing import Iterable

from app.core.amdr import FAT_MIN_RATIO, FAT_MAX_RATIO, MacronutrientRatios

# Evidence-backed adjustment: dyslipidemia/ASCVD -> lower total fat upper bound.
DYSLIPIDEMIA_FAT_MAX_RATIO = 0.25  # derived from dyslipidemia/ASCVD guidance to reduce total fat upper limit


def apply_disease_adjustments(
    eer: float,
    carb_ratios: MacronutrientRatios,
    diseases: Iterable[str],
    fat_ratios: MacronutrientRatios | None = None,
) -> tuple[float, MacronutrientRatios, MacronutrientRatios]:
    normalized = {disease.lower() for disease in diseases}
    adjusted_eer = eer
    adjusted_carb_ratios = carb_ratios
    adjusted_fat_ratios = fat_ratios or MacronutrientRatios(FAT_MIN_RATIO, FAT_MAX_RATIO)

    if "dyslipidemia" in normalized or "ascvd" in normalized:
        new_fat_max = min(adjusted_fat_ratios.max_ratio, DYSLIPIDEMIA_FAT_MAX_RATIO)
        adjusted_fat_ratios = MacronutrientRatios(
            min_ratio=adjusted_fat_ratios.min_ratio,
            max_ratio=new_fat_max,
        )

    return adjusted_eer, adjusted_carb_ratios, adjusted_fat_ratios
