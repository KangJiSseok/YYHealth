from dataclasses import dataclass


CARB_MIN_RATIO = 0.55
CARB_MAX_RATIO = 0.65

PROTEIN_MIN_RATIO = 0.07
PROTEIN_MAX_RATIO = 0.20

FAT_MIN_RATIO = 0.15
FAT_MAX_RATIO = 0.30

CARB_KCAL_PER_G = 4.0
PROTEIN_KCAL_PER_G = 4.0
FAT_KCAL_PER_G = 9.0


@dataclass(frozen=True)
class MacronutrientRatios:
    min_ratio: float
    max_ratio: float

    @property
    def recommended_ratio(self) -> float:
        return (self.min_ratio + self.max_ratio) / 2.0


@dataclass(frozen=True)
class MacronutrientProfile:
    recommended_g: float
    min_g: float
    max_g: float
    recommended_ratio: float
    min_ratio: float
    max_ratio: float


def default_ratios() -> dict:
    return {
        "carbohydrate": MacronutrientRatios(min_ratio=CARB_MIN_RATIO, max_ratio=CARB_MAX_RATIO),
        "protein": MacronutrientRatios(min_ratio=PROTEIN_MIN_RATIO, max_ratio=PROTEIN_MAX_RATIO),
        "fat": MacronutrientRatios(min_ratio=FAT_MIN_RATIO, max_ratio=FAT_MAX_RATIO),
    }


def _ratio_to_grams(eer: float, ratio: float, kcal_per_g: float) -> float:
    return (eer * ratio) / kcal_per_g


def calculate_macronutrients(eer: float, carb_ratios: MacronutrientRatios) -> dict:
    ratios = {
        "carbohydrate": carb_ratios,
        "protein": MacronutrientRatios(PROTEIN_MIN_RATIO, PROTEIN_MAX_RATIO),
        "fat": MacronutrientRatios(FAT_MIN_RATIO, FAT_MAX_RATIO),
    }
    kcal_map = {
        "carbohydrate": CARB_KCAL_PER_G,
        "protein": PROTEIN_KCAL_PER_G,
        "fat": FAT_KCAL_PER_G,
    }
    profile = {}
    for macro, ratio in ratios.items():
        kcal_per_g = kcal_map[macro]
        min_g = _ratio_to_grams(eer, ratio.min_ratio, kcal_per_g)
        max_g = _ratio_to_grams(eer, ratio.max_ratio, kcal_per_g)
        recommended_g = _ratio_to_grams(eer, ratio.recommended_ratio, kcal_per_g)
        profile[macro] = MacronutrientProfile(
            recommended_g=recommended_g,
            min_g=min_g,
            max_g=max_g,
            recommended_ratio=ratio.recommended_ratio,
            min_ratio=ratio.min_ratio,
            max_ratio=ratio.max_ratio,
        )
    return profile
