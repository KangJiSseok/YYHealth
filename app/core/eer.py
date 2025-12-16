import math
from dataclasses import dataclass
from typing import Literal


def _height_m(height_cm: float) -> float:
    return height_cm / 100.0


def eer_age_1_to_2(weight_kg: float) -> float:
    return 89.0 * weight_kg - 100.0 + 20.0


def eer_infant(weight_kg: float, months: int) -> float:
    base = 89.0 * weight_kg - 100.0
    if 0 <= months <= 5:
        return base + 115.5
    if 6 <= months <= 11:
        return base + 22.0
    raise ValueError("Infant months must be between 0 and 11")


def eer_child_male(age: int, height_cm: float, weight_kg: float, pa: float) -> float:
    height_m = _height_m(height_cm)
    return 88.5 - 61.9 * age + pa * (26.7 * weight_kg + 903.0 * height_m)


def eer_child_female(age: int, height_cm: float, weight_kg: float, pa: float) -> float:
    height_m = _height_m(height_cm)
    return 135.3 - 30.8 * age + pa * (10.0 * weight_kg + 934.0 * height_m)


def eer_male_15_to_19(age: int, height_cm: float, weight_kg: float, pa: float) -> float:
    height_m = _height_m(height_cm)
    return 662.0 - 9.53 * age + pa * (15.91 * weight_kg + 539.6 * height_m)


def eer_female_15_to_19(
    age: int, height_cm: float, weight_kg: float, pa: float
) -> float:
    height_m = _height_m(height_cm)
    return 354.0 - 6.91 * age + pa * (9.36 * weight_kg + 726.0 * height_m)


def eer_male_20_plus(age: int, height_cm: float, weight_kg: float, pa: float) -> float:
    height_m = _height_m(height_cm)
    return 662.0 - 9.53 * age + pa * (15.91 * weight_kg + 539.6 * height_m)


def eer_female_20_plus(
    age: int, height_cm: float, weight_kg: float, pa: float
) -> float:
    height_m = _height_m(height_cm)
    return 354.0 - 6.91 * age + pa * (9.36 * weight_kg + 726.0 * height_m)


def round_down_100(value: float) -> float:
    return math.floor(value / 100.0) * 100.0


@dataclass(frozen=True)
class EERDetails:
    value: float
    life_stage: str
    formula_name: str
    pa: float


def calculate_eer_life_stage(
    sex: Literal["male", "female"],
    age_years: int,
    age_months: int,
    height_cm: float or None,
    weight_kg: float,
    activity_level: Literal["inactive", "low", "active", "very_active"],
    pa_resolver,
    pregnancy_stage: Literal["none", "trimester_1", "trimester_2", "trimester_3"],
    lactation: bool,
) -> EERDetails:
    pa_value: float = 1.0

    if age_years == 0 and 0 <= age_months <= 11:
        raw = eer_infant(weight_kg, age_months)
        life_stage = "infancy_0_5_months" if age_months <= 5 else "infancy_6_11_months"
        return EERDetails(
            value=round_down_100(raw),
            life_stage=life_stage,
            formula_name=life_stage,
            pa=pa_value,
        )

    if 1 <= age_years <= 2:
        raw = eer_age_1_to_2(weight_kg)
        return EERDetails(
            value=round_down_100(raw),
            life_stage="early_childhood_1_2_years",
            formula_name="age_1_2",
            pa=pa_value,
        )

    if 3 <= age_years <= 8:
        pa_value = pa_resolver(age_years, sex, activity_level)
        growth_addition = 20.0
        if sex == "male":
            base = eer_child_male(age_years, height_cm, weight_kg, pa_value)  # type: ignore[arg-type]
            formula_name = "child_male_pa_growth"
        else:
            base = eer_child_female(age_years, height_cm, weight_kg, pa_value)  # type: ignore[arg-type]
            formula_name = "child_female_pa_growth"
        raw = base + growth_addition
        return EERDetails(
            value=round_down_100(raw),
            life_stage="childhood_3_8_years",
            formula_name=formula_name,
            pa=pa_value,
        )

    if 9 <= age_years <= 14:
        pa_value = pa_resolver(age_years, sex, activity_level)
        growth_addition = 25.0
        if sex == "male":
            base = eer_child_male(age_years, height_cm, weight_kg, pa_value)  # type: ignore[arg-type]
            formula_name = "child_male_pa_growth"
        else:
            base = eer_child_female(age_years, height_cm, weight_kg, pa_value)  # type: ignore[arg-type]
            formula_name = "child_female_pa_growth"
        raw = base + growth_addition
        return EERDetails(
            value=round_down_100(raw),
            life_stage="adolescence_9_18_years",
            formula_name=formula_name,
            pa=pa_value,
        )

    if 15 <= age_years <= 18:
        pa_value = pa_resolver(age_years, sex, activity_level)
        growth_addition = 25.0
        if sex == "male":
            base = eer_male_15_to_19(age_years, height_cm, weight_kg, pa_value)  # type: ignore[arg-type]
            formula_name = "male_15_18_pa_growth"
        else:
            base = eer_female_15_to_19(age_years, height_cm, weight_kg, pa_value)  # type: ignore[arg-type]
            formula_name = "female_15_18_pa_growth"
        raw = base + growth_addition
        return EERDetails(
            value=round_down_100(raw),
            life_stage="adolescence_9_18_years",
            formula_name=formula_name,
            pa=pa_value,
        )

    if 19 <= age_years <= 64:
        pa_value = pa_resolver(age_years, sex, activity_level)
        if sex == "male":
            base = eer_male_20_plus(age_years, height_cm, weight_kg, pa_value)  # type: ignore[arg-type]
            formula_name = "male_20_plus"
        else:
            base = eer_female_20_plus(age_years, height_cm, weight_kg, pa_value)  # type: ignore[arg-type]
            formula_name = "female_20_plus"
        base_eer = base
        if pregnancy_stage != "none":
            addition = 0.0
            if pregnancy_stage == "trimester_2":
                addition = 340.0
            elif pregnancy_stage == "trimester_3":
                addition = 450.0
            raw = base_eer + addition
            return EERDetails(
                value=round_down_100(raw),
                life_stage=f"pregnancy_{pregnancy_stage}",
                formula_name=formula_name,
                pa=pa_value,
            )
        if lactation:
            raw = base_eer + 340.0
            return EERDetails(
                value=round_down_100(raw),
                life_stage="lactation",
                formula_name=formula_name,
                pa=pa_value,
            )
        return EERDetails(
            value=round_down_100(base_eer),
            life_stage="adulthood_19_64_years",
            formula_name=formula_name,
            pa=pa_value,
        )

    if age_years >= 65:
        pa_value = pa_resolver(age_years, sex, activity_level)
        if sex == "male":
            base = eer_male_20_plus(age_years, height_cm, weight_kg, pa_value)  # type: ignore[arg-type]
            formula_name = "male_20_plus"
        else:
            base = eer_female_20_plus(age_years, height_cm, weight_kg, pa_value)  # type: ignore[arg-type]
            formula_name = "female_20_plus"
        return EERDetails(
            value=round_down_100(base),
            life_stage="elderly_65_plus_years",
            formula_name=formula_name,
            pa=pa_value,
        )

    raise ValueError("EER formula not defined for the provided age")
