from fastapi import APIRouter
from analysis import LogAnalyzer

router = APIRouter()

@router.get("/analysis/basic")
async def get_basic_analysis():
    analyzer = LogAnalyzer()
    return analyzer.get_basic_stats()

@router.get("/analysis/temporal")
async def get_temporal_analysis():
    analyzer = LogAnalyzer()
    return analyzer.temporal_analysis()

@router.get("/analysis/report")
async def get_full_report():
    analyzer = LogAnalyzer()
    return {"report": analyzer.generate_report()}