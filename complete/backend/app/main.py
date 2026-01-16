from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import sys
import os
import re

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from combined_evaluation_system import CombinedEvaluator, EvaluationScore, ComparisonResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ManavaAI Backend", description="AI Text Detection and Humanization API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models
evaluator: Optional[CombinedEvaluator] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the evaluator on startup"""
    global evaluator
    try:
        logger.info("Initializing Combined Evaluator with trained quotes model...")
        # Define local path to trained model
        local_model_path = r"c:\Users\hp\Downloads\ManavaAI-main (1)\ManavaAI-main\output_quotes"
        
        # Initialize with local path
        evaluator = CombinedEvaluator(model_name=local_model_path, use_grammar_check=True)
        logger.info("Evaluator initialized successfully with local model.")
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        # We might not want to crash the whole app if one model fails, 
        # but for this specific app, the evaluator is core.
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global evaluator
    if evaluator:
        del evaluator

class AnalyzeRequest(BaseModel):
    text: str

class CompareRequest(BaseModel):
    original_text: str
    humanized_text: str
    original_label: str = "Original"
    humanized_label: str = "Humanized"

@app.get("/")
async def root():
    return {"status": "online", "message": "ManavaAI Backend 2.0 is running"}

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text for AI detection metrics
    """
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        score = evaluator.evaluate(request.text)
        return score.to_dict()
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare", response_model=Dict[str, Any])
async def compare_texts(request: CompareRequest):
    """
    Compare two texts (before vs after)
    """
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")

    if not request.original_text.strip() or not request.humanized_text.strip():
        raise HTTPException(status_code=400, detail="Both texts must be non-empty")

    try:
        result = evaluator.compare(
            request.original_text, 
            request.humanized_text,
            request.original_label,
            request.humanized_label
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/humanize")
async def humanize_text(request: AnalyzeRequest):
    """
    Humanize text to improve readability and bypass AI detection.
    This is a hybrid implementation using rule-based improvements.
    """
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # 1. AI-Powered Humanization (Using trained model)
        humanized_text = evaluator.humanize(request.text)

        # 2. Evaluate the transformation
        result = evaluator.compare(request.text, humanized_text)
        
        return {
            "original_text": request.text,
            "humanized_text": humanized_text,
            "score": result.after_score.to_dict(),
            "comparison": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Error during humanization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
