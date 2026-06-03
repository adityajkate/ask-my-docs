from fastapi import APIRouter, Depends
from app.dependencies import Principal, get_principal
from app.generation.citations import extract_citation_ids
from app.main_services import get_generator, get_retrieval_service
from app.schemas import EvaluationRequest, EvaluationResult


router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


@router.post("/run", response_model=EvaluationResult)
async def run_evaluation(request: EvaluationRequest, principal: Principal = Depends(get_principal)):
    results = []
    recall_hits = 0
    citation_valid = 0
    for case in request.cases:
        retrieval = get_retrieval_service().retrieve(case.query, principal)
        answer, fallback = get_generator().answer(case.query, retrieval.selected)
        retrieved_ids = [chunk.chunk_id for chunk in retrieval.selected]
        expected = set(case.expected_chunk_ids)
        if not expected or expected & set(retrieved_ids):
            recall_hits += 1
        cited_ids = extract_citation_ids(answer)
        if fallback or cited_ids.issubset(set(retrieved_ids)):
            citation_valid += 1
        results.append(
            {
                "query": case.query,
                "retrieved_chunk_ids": retrieved_ids,
                "expected_chunk_ids": case.expected_chunk_ids,
                "answer": answer,
                "fallback": fallback,
            }
        )
    total = max(1, len(request.cases))
    return EvaluationResult(
        total_cases=len(request.cases),
        context_recall=recall_hits / total,
        citation_validity=citation_valid / total,
        results=results,
    )

