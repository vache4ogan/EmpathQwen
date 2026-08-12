from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from front_back.backend.schemas import (
    ChatRequest,
    ChatResponse,
    ClearChatRequest,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/message", response_model=ChatResponse)
async def send_message(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    service = request.app.state.chat_service

    answer, crisis_detected = await run_in_threadpool(
        service.send_message,
        payload.session_id,
        payload.message,
    )

    return ChatResponse(
        session_id=payload.session_id,
        message=answer,
        crisis_detected=crisis_detected,
    )


@router.delete("/session")
async def clear_session(
    payload: ClearChatRequest,
    request: Request,
) -> dict[str, bool]:
    request.app.state.chat_service.clear_session(payload.session_id)
    return {"ok": True}
