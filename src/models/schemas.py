"""
Pydantic models for all FastAPI request/response shapes.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


# ─────────────────────────────────────────
# Player schemas
# ─────────────────────────────────────────

class PlayerSummary(BaseModel):
    name: str
    role: str
    origin: str
    base_price: int                  # in rupees
    base_price_cr: float             # in crores
    set_type: str
    avg_price_cr: Optional[float] = None
    max_price_cr: Optional[float] = None
    trajectory: Optional[str] = None
    batting_avg: Optional[float] = None
    batting_sr: Optional[float] = None
    total_runs: Optional[int] = None
    wickets: Optional[int] = None
    economy: Optional[float] = None
    bowler_category: Optional[str] = None


class NominatedPlayer(BaseModel):
    """Currently nominated player in the live auction console."""
    name: str
    role: str
    origin: str
    base_price: int
    base_price_cr: float
    set_type: str
    batting_category: Optional[str] = None
    bowling_category: Optional[str] = None
    avg_price_cr: Optional[float] = None
    historical_prices: list[dict] = []
    index_in_pool: int = 0
    total_in_pool: int = 0


# ─────────────────────────────────────────
# Squad schemas
# ─────────────────────────────────────────

class SquadPlayer(BaseModel):
    name: str
    role: str
    origin: str
    price_paid_cr: Optional[float] = None


class SquadCounters(BaseModel):
    batsmen: int
    bowlers: int
    allrounders: int
    wicketkeepers: int
    indians: int
    overseas: int
    overseas_max: int = 8


class TeamSquadResponse(BaseModel):
    team_code: str
    team_name: str
    remaining_purse_cr: float
    total_purse_cr: float
    squad: list[SquadPlayer]
    counters: SquadCounters


# ─────────────────────────────────────────
# Simulation schemas
# ─────────────────────────────────────────

class SimulationRequest(BaseModel):
    n_runs: int = 100                # 100, 500, or 1000
    mode: str = "next_player"        # "next_player" | "set" | "full_auction"


class PriceDistribution(BaseModel):
    player_name: str
    role: str
    origin: str
    n_runs: int
    sold_count: int
    unsold_count: int
    unsold_probability: float
    # Price percentiles (only for sold outcomes), all in crores
    min_cr: float
    p10_cr: float
    p25_cr: float
    median_cr: float
    p75_cr: float
    p90_cr: float
    max_cr: float
    mean_cr: float
    # Team win probabilities
    team_win_probabilities: dict[str, float]


class SimulationResult(BaseModel):
    mode: str
    n_runs: int
    distributions: list[PriceDistribution]
    duration_ms: float


class SimulationStatus(BaseModel):
    running: bool
    progress: int      # 0-100
    current_run: int
    total_runs: int
    mode: str


# ─────────────────────────────────────────
# RAG schemas
# ─────────────────────────────────────────

class RAGQuery(BaseModel):
    question: str
    player_name: Optional[str] = None


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources_used: list[str] = []


# ─────────────────────────────────────────
# Team / general schemas
# ─────────────────────────────────────────

class TeamInfo(BaseModel):
    code: str
    name: str
    remaining_purse_cr: float


class AuctionStateResponse(BaseModel):
    current_player: Optional[NominatedPlayer]
    current_set: str
    players_remaining: int
    teams: list[TeamInfo]
    simulation_running: bool


class HealthResponse(BaseModel):
    status: str
    players_loaded: int
    data_sources: list[str]
