import math

import pytest

from app.core import eer, pa
from app.core.calculator import calculate_nutrition
from app.schemas.survey import SurveyRequest


def _base_request(**kwargs) -> SurveyRequest:
    defaults = dict(
        sex="male",
        age_years=20,
        age_months=None,
        height_cm=175.0,
        weight_kg=70.0,
        activity_level="low",
        pregnancy_stage="none",
        lactation=False,
        diseases=[],
    )
    defaults.update(kwargs)
    return SurveyRequest(**defaults)


def test_infant_groups_apply_growth_and_rounding():
    infant_4m = SurveyRequest(
        sex="female",
        age_years=0,
        age_months=4,
        height_cm=None,
        weight_kg=5.5,
        activity_level="inactive",
        pregnancy_stage="none",
        lactation=False,
        diseases=[],
    )
    infant_8m = infant_4m.model_copy(update={"age_months": 8, "weight_kg": 8.4})

    result_4m = calculate_nutrition(infant_4m)
    result_8m = calculate_nutrition(infant_8m)

    assert result_4m.eer == 500.0
    assert result_8m.eer == 600.0


def test_growth_addition_boundaries():
    age2 = SurveyRequest(
        sex="male",
        age_years=2,
        age_months=None,
        height_cm=None,
        weight_kg=12.0,
        activity_level="inactive",
        pregnancy_stage="none",
        lactation=False,
        diseases=[],
    )
    age3 = SurveyRequest(
        sex="male",
        age_years=3,
        age_months=None,
        height_cm=95.0,
        weight_kg=12.0,
        activity_level="low",
        pregnancy_stage="none",
        lactation=False,
        diseases=[],
    )
    age8 = age3.model_copy(update={"age_years": 8, "height_cm": 130.0, "weight_kg": 25.0})
    age9 = age8.model_copy(update={"age_years": 9})

    result_2 = calculate_nutrition(age2)
    result_3 = calculate_nutrition(age3)
    result_8 = calculate_nutrition(age8)
    result_9 = calculate_nutrition(age9)

    assert result_3.eer >= result_2.eer
    pa_8 = pa.resolve_pa(8, "male", "low")
    pa_9 = pa.resolve_pa(9, "male", "low")
    expected_8 = eer.round_down_100(eer.eer_child_male(8, 130.0, 25.0, pa_8) + 20.0)
    expected_9 = eer.round_down_100(eer.eer_child_male(9, 130.0, 25.0, pa_9) + 25.0)

    assert result_8.eer == expected_8
    assert result_9.eer == expected_9


def test_age_transition_14_to_15_changes_formula_and_pa():
    survey_14 = SurveyRequest(
        sex="male",
        age_years=14,
        age_months=None,
        height_cm=170.0,
        weight_kg=60.0,
        activity_level="active",
        pregnancy_stage="none",
        lactation=False,
        diseases=[],
    )
    survey_15 = survey_14.model_copy(update={"age_years": 15})

    result_14 = calculate_nutrition(survey_14)
    result_15 = calculate_nutrition(survey_15)

    pa_14 = pa.resolve_pa(14, "male", "active")
    expected_14_raw = eer.eer_child_male(14, 170.0, 60.0, pa_14) + 25.0
    pa_15 = pa.resolve_pa(15, "male", "active")
    expected_15_raw = eer.eer_male_15_to_19(15, 170.0, 60.0, pa_15) + 25.0

    assert math.isclose(result_14.eer, eer.round_down_100(expected_14_raw))
    assert math.isclose(result_15.eer, eer.round_down_100(expected_15_raw))
    assert result_15.eer != result_14.eer


def test_age_transition_19_to_20_allows_adult_formula():
    survey_19 = SurveyRequest(
        sex="female",
        age_years=19,
        age_months=None,
        height_cm=162.0,
        weight_kg=55.0,
        activity_level="low",
        pregnancy_stage="none",
        lactation=False,
        diseases=[],
    )
    survey_20 = survey_19.model_copy(update={"age_years": 20})

    result_19 = calculate_nutrition(survey_19)
    result_20 = calculate_nutrition(survey_20)

    assert result_19.life_stage == "adulthood_19_64_years"
    assert result_20.life_stage == "adulthood_19_64_years"
    assert result_19.eer != 0
    assert result_20.eer != 0


def test_sex_difference_changes_eer():
    male = _base_request(sex="male", age_years=20, height_cm=175.0, weight_kg=70.0)
    female = _base_request(sex="female", age_years=20, height_cm=175.0, weight_kg=70.0)

    result_male = calculate_nutrition(male)
    result_female = calculate_nutrition(female)

    assert result_male.eer != result_female.eer


def test_activity_level_affects_eer():
    base = dict(
        sex="female",
        age_years=14,
        age_months=None,
        height_cm=160.0,
        weight_kg=50.0,
        pregnancy_stage="none",
        lactation=False,
        diseases=[],
    )
    inactive = SurveyRequest(activity_level="inactive", **base)
    very_active = SurveyRequest(activity_level="very_active", **base)

    result_inactive = calculate_nutrition(inactive)
    result_very_active = calculate_nutrition(very_active)

    assert result_very_active.eer >= result_inactive.eer


def test_obesity_applies_calorie_deficit():
    survey = _base_request(
        sex="male",
        age_years=20,
        height_cm=180.0,
        weight_kg=80.0,
        activity_level="active",
    )
    with_obesity = survey.model_copy(update={"diseases": ["obesity"]})

    base_result = calculate_nutrition(survey)
    obesity_result = calculate_nutrition(with_obesity)

    assert math.isclose(obesity_result.adjusted_eer, base_result.eer * 0.90, rel_tol=1e-9)


def test_diabetes_reduces_carb_upper_ratio():
    survey = _base_request(
        sex="female",
        age_years=20,
        height_cm=165.0,
        weight_kg=60.0,
        activity_level="inactive",
        diseases=["diabetes"],
    )
    result = calculate_nutrition(survey)
    carb = result.macronutrients["carbohydrate"]

    assert carb.max_ratio == pytest.approx(0.60)
    assert carb.recommended_ratio == pytest.approx((carb.min_ratio + carb.max_ratio) / 2.0)


def test_reproducibility_same_input_same_output():
    survey = _base_request(
        sex="male",
        age_years=15,
        height_cm=172.0,
        weight_kg=65.0,
        activity_level="low",
        diseases=["hypertension"],
    )
    first = calculate_nutrition(survey)
    second = calculate_nutrition(survey)

    assert first.model_dump() == second.model_dump()


def test_pregnancy_and_lactation_mutually_exclusive():
    with pytest.raises(ValueError):
        SurveyRequest(
            sex="female",
            age_years=25,
            age_months=None,
            height_cm=165.0,
            weight_kg=60.0,
            activity_level="low",
            pregnancy_stage="trimester_1",
            lactation=True,
            diseases=[],
        )


def test_pregnancy_trimester_adjustments():
    base = _base_request(sex="female", age_years=30, height_cm=165.0, weight_kg=60.0)
    trimester2 = base.model_copy(update={"pregnancy_stage": "trimester_2"})
    trimester3 = base.model_copy(update={"pregnancy_stage": "trimester_3"})

    base_result = calculate_nutrition(base)
    t2_result = calculate_nutrition(trimester2)
    t3_result = calculate_nutrition(trimester3)

    assert t2_result.eer == eer.round_down_100(base_result.eer + 340.0)
    assert t3_result.eer == eer.round_down_100(base_result.eer + 450.0)
    assert t2_result.life_stage.startswith("pregnancy")
    assert t3_result.life_stage.startswith("pregnancy")


def test_lactation_adjustment():
    base = _base_request(sex="female", age_years=28, height_cm=160.0, weight_kg=55.0)
    lactating = base.model_copy(update={"lactation": True})

    base_result = calculate_nutrition(base)
    lact_result = calculate_nutrition(lactating)

    assert lact_result.eer == eer.round_down_100(base_result.eer + 340.0)
    assert lact_result.life_stage == "lactation"


def test_rounding_down_to_nearest_100():
    survey = _base_request(sex="male", age_years=2, age_months=None, height_cm=None, weight_kg=13.7)
    result = calculate_nutrition(survey)
    assert result.eer % 100 == 0
