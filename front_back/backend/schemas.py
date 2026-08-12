from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=5000)


class ChatResponse(BaseModel):
    session_id: str
    message: str
    crisis_detected: bool = False


class ClearChatRequest(BaseModel):
    session_id: str