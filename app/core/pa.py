from typing import Literal

PA_12_14 = {
    "male": {
        "inactive": 1.0,
        "low": 1.13,
        "active": 1.26,
        "very_active": 1.42,
    },
    "female": {
        "inactive": 1.0,
        "low": 1.16,
        "active": 1.31,
        "very_active": 1.56,
    },
}

PA_15_PLUS = {
    "male": {
        "inactive": 1.0,
        "low": 1.11,
        "active": 1.25,
        "very_active": 1.48,
    },
    "female": {
        "inactive": 1.0,
        "low": 1.12,
        "active": 1.27,
        "very_active": 1.45,
    },
}


def resolve_pa(
    age: int,
    sex: Literal["male", "female"],
    activity_level: Literal["inactive", "low", "active", "very_active"],
) -> float:
    """
    Select the PA coefficient based on age group, sex, and activity level.
    Ages below 12 do not use PA in the provided EER formulas.
    """
    if age < 12:
        return 1.0
    if 12 <= age <= 14:
        return PA_12_14[sex][activity_level]
    if age >= 15:
        return PA_15_PLUS[sex][activity_level]
    raise ValueError("PA not defined for the provided age")
