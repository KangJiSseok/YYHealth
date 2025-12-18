from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: Optional[int] = Field(default=None, alias="conversationId")
    message: str
    diseases: Optional[List[str]] = []

    class Config:
        populate_by_name = True


class ChatResponse(BaseModel):
    conversation_id: Optional[int] = Field(default=None, alias="conversationId")
    answer: str
    evidence: List[dict] = []

    class Config:
        populate_by_name = True
