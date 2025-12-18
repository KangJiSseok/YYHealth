from typing import List, Optional

from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    height: Optional[float] = None
    weight: Optional[float] = None
    birthdate: Optional[str] = None
    kcal: Optional[float] = None
    protein: Optional[float] = None
    fat: Optional[float] = None
    protein_min: Optional[float] = None
    protein_max: Optional[float] = None
    fat_min: Optional[float] = None
    fat_max: Optional[float] = None


class ChatRequest(BaseModel):
    conversation_id: Optional[int] = Field(default=None, alias="conversationId")
    message: str
    diseases: Optional[List[str]] = []
    user_info: Optional[UserInfo] = Field(default=None, alias="userInfo")

    class Config:
        populate_by_name = True


class ChatResponse(BaseModel):
    conversation_id: Optional[int] = Field(default=None, alias="conversationId")
    answer: str
    evidence: List[dict] = []

    class Config:
        populate_by_name = True
