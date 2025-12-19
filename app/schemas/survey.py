from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SurveyRequest(BaseModel):
    sex: Literal["male", "female"]
    age_years: Optional[int] = None
    age_months: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: float
    activity_level: Literal["inactive", "low", "active", "very_active"]
    pregnancy_stage: Literal["none", "trimester_1", "trimester_2", "trimester_3"]
    lactation: bool
    diseases: List[str] = Field(default_factory=list)

    @field_validator("age_years")
    @classmethod
    def validate_age_years(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError("age_years must be non-negative")
        return value

    @field_validator("age_months")
    @classmethod
    def validate_age_months(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and (value < 0 or value > 11):
            raise ValueError("age_months must be between 0 and 11")
        return value

    @field_validator("height_cm")
    @classmethod
    def validate_height(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("height_cm must be positive")
        return value

    @field_validator("weight_kg")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("weight_kg must be positive")
        return value

    @model_validator(mode="after")
    def validate_age_combination(self) -> "SurveyRequest":
        if self.age_years is None and self.age_months is None:
            raise ValueError("age_years or age_months must be provided")

        if (self.age_years is None or self.age_years == 0) and self.age_months is None:
            raise ValueError("infants must provide age_months")

        if self.age_years is not None and self.age_years >= 1 and self.age_months is not None:
            if self.age_years >= 1 and self.age_months >= 12:
                raise ValueError("age_months must be less than 12")

        if self.pregnancy_stage != "none" and self.lactation:
            raise ValueError("pregnancy and lactation cannot both be true")

        if self.pregnancy_stage != "none" or self.lactation:
            if self.sex != "female":
                raise ValueError("pregnancy or lactation only allowed for females")
            if self.age_years is None or self.age_years < 19:
                raise ValueError("pregnancy or lactation only allowed for adult females")

        if self.age_years is not None and self.age_years >= 3 and self.height_cm is None:
            raise ValueError("height_cm is required for ages 3 and above")

        return self
