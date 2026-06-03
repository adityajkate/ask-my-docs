from functools import lru_cache
from app.config import get_settings
from app.generation.generator import GroundedGenerator
from app.ingestion.service import IngestionService
from app.retrieval.service import RetrievalService
from app.storage import Storage


@lru_cache
def get_storage() -> Storage:
    return Storage(get_settings().database_path)


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(get_storage(), get_settings())


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(get_storage(), get_settings())


@lru_cache
def get_generator() -> GroundedGenerator:
    return GroundedGenerator(get_settings())

