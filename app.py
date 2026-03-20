"""
FastAPI application entry point for the IPL Auction Simulator.

Run with:
    cd backend
    uvicorn app:app --reload --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.api.routes import router
from src.services.auction_service import get_service

# Load .env from backend directory
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Lifespan: initialise on startup
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    base = Path(__file__).parent

    batting_csv = str(base / "data" / "IPL_Auction_data_Batting.csv")
    bowling_csv = str(base / "data" / "IPL_Auction_data_Bowling.csv")
    docs_dir = str(base / "docs")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    if not gemini_key:
        logger.warning("GEMINI_API_KEY not set — RAG features will be disabled.")

    svc = get_service()
    await svc.initialise(batting_csv, bowling_csv, docs_dir, gemini_key)
    logger.info("AuctionService ready.")

    yield  # application runs

    logger.info("Shutting down.")


# ─────────────────────────────────────────
# App factory
# ─────────────────────────────────────────

app = FastAPI(
    title="IPL Auction Simulator API",
    version="1.0.0",
    description=(
        "Monte Carlo IPL auction simulator with 10 AI franchise agents, "
        "price-range distributions, and Gemini-powered RAG."
    ),
    lifespan=lifespan,
)

# Allow the Vite dev server and production build to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "IPL Auction Simulator API", "docs": "/docs"}
