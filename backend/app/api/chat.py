import json
import uuid
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.dependencies import Principal, get_principal
from app.generation.citations import citation_payload
from app.main_services import get_generator, get_retrieval_service, get_storage
from app.schemas import ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse, RetrievalDebug


router = APIRouter(prefix="/api", tags=["chat"])


def build_response(request: ChatRequest, principal: Principal) -> ChatResponse:
    retrieval = get_retrieval_service().retrieve(request.question, principal, request.top_k)
    answer, fallback = get_generator().answer(request.question, retrieval.selected)
    citations = [citation_payload(chunk) for chunk in retrieval.selected] if not fallback else []
    session_id = request.session_id or str(uuid.uuid4())
    message_id = get_storage().save_chat_message(
        session_id=session_id,
        user_id=principal.user_id,
        question=request.question,
        answer=answer,
        citations=citations,
        fallback=fallback,
    )
    debug = None
    if request.include_debug:
        debug = RetrievalDebug(
            dense_results=retrieval.dense_count,
            bm25_results=retrieval.bm25_count,
            fused_results=retrieval.fused_count,
            selected_results=len(retrieval.selected),
            confidence=retrieval.confidence,
        )
    return ChatResponse(
        answer=answer,
        citations=citations,
        fallback=fallback,
        session_id=session_id,
        message_id=message_id,
        debug=debug,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, principal: Principal = Depends(get_principal)):
    return build_response(request, principal)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, principal: Principal = Depends(get_principal)):
    response = build_response(request, principal)

    async def events():
        for token in response.answer.split(" "):
            yield json.dumps({"type": "token", "value": token + " "}) + "\n"
        yield json.dumps({"type": "citations", "value": [item.model_dump() for item in response.citations]}) + "\n"
        yield json.dumps(
            {
                "type": "done",
                "message_id": response.message_id,
                "session_id": response.session_id,
                "fallback": response.fallback,
            }
        ) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest, principal: Principal = Depends(get_principal)):
    feedback_id = get_storage().save_feedback(
        message_id=request.message_id,
        user_id=principal.user_id,
        rating=request.rating,
        comment=request.comment,
    )
    return FeedbackResponse(feedback_id=feedback_id, accepted=True)

