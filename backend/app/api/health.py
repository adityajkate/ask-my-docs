from fastapi import APIRouter
from app.config import get_settings


router = APIRouter()


@router.get("/health")
async def health():
    settings = get_settings()
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}

