from fastapi import APIRouter
from .analysis import LogAnalyzer

# Create API router
router = APIRouter(prefix="/api", tags=["analysis"])

__all__ = ["LogAnalyzer", "router"]