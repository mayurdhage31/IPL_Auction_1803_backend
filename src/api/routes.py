"""
FastAPI route definitions for the IPL Auction Simulator.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from ..services.auction_service import get_service
from ..models.schemas import (
    SimulationRequest, RAGQuery,
    TeamSquadResponse, SquadPlayer, SquadCounters,
    NominatedPlayer, AuctionStateResponse, TeamInfo,
    SimulationResult, PriceDistribution, SimulationStatus,
    RAGResponse, HealthResponse, PlayerSummary,
)

router = APIRouter()


# ─────────────────────────────────────────
# Health
# ─────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    svc = get_service()
    n = len(svc.all_pool)
    sources = []
    if svc.processor:
        if svc.processor.batting_csv:
            sources.append("batting_csv")
        if svc.processor.bowling_csv:
            sources.append("bowling_csv")
    return HealthResponse(
        status="ok" if n > 0 else "no_data",
        players_loaded=n,
        data_sources=sources,
    )


# ─────────────────────────────────────────
# Teams
# ─────────────────────────────────────────

@router.get("/teams")
async def get_teams():
    svc = get_service()
    return svc.get_all_teams()


@router.get("/teams/{team_code}/squad", response_model=TeamSquadResponse)
async def get_team_squad(team_code: str):
    svc = get_service()
    data = svc.get_team_squad(team_code.upper())
    if not data:
        raise HTTPException(status_code=404, detail=f"Team {team_code} not found")

    return TeamSquadResponse(
        team_code=data["team_code"],
        team_name=data["team_name"],
        remaining_purse_cr=data["remaining_purse_cr"],
        total_purse_cr=data["total_purse_cr"],
        squad=[SquadPlayer(**p) for p in data["squad"]],
        counters=SquadCounters(**data["counters"]),
    )


# ─────────────────────────────────────────
# Players / Nomination
# ─────────────────────────────────────────

@router.get("/players/current")
async def get_current_player():
    svc = get_service()
    player = svc.get_current_player()
    if not player:
        raise HTTPException(status_code=404, detail="No current player — pool exhausted")
    return player


@router.post("/players/next")
async def advance_player():
    svc = get_service()
    player = svc.advance_player()
    if not player:
        raise HTTPException(status_code=404, detail="No more players in pool")
    return player


@router.get("/players")
async def list_players(
    role: Optional[str] = None,
    origin: Optional[str] = None,
    set_type: Optional[str] = None,
    team: Optional[str] = None,
    limit: int = 50,
):
    svc = get_service()
    # Team filter: players historically bought by this team
    if team:
        return svc.get_players_for_team(team.upper())[:limit]

    players = svc.all_pool
    if role:
        players = [p for p in players if p.role.value.lower() == role.lower()]
    if origin:
        players = [p for p in players if p.origin.value.lower() == origin.lower()]
    if set_type:
        players = [p for p in players if p.set_type.value.lower() == set_type.lower()]
    return [
        {
            "name": p.name,
            "role": p.role.value,
            "origin": p.origin.value,
            "base_price_cr": round(p.base_price / 1e7, 2),
            "set_type": p.set_type.value,
            "avg_price_cr": round(p.stats["avg_price"] / 1e7, 2),
        }
        for p in players[:limit]
    ]


@router.get("/players/{player_name}")
async def get_player_detail(player_name: str):
    svc = get_service()
    player = svc.processor.get_player(player_name) if svc.processor else None
    if not player:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
    return svc.get_player_detail(player)


# ─────────────────────────────────────────
# Auction state
# ─────────────────────────────────────────

@router.get("/auction/state")
async def get_auction_state():
    svc = get_service()
    current = svc.get_current_player()
    teams = svc.get_all_teams()
    cur_set = "marquee"
    if current:
        cur_set = current.get("set_type", "capped")
    return {
        "current_player": current,
        "current_set": cur_set,
        "current_index": svc.current_index,
        "players_remaining": max(0, len(svc.all_pool) - svc.current_index),
        "teams": teams,
        "simulation_running": svc.simulation_running,
    }


@router.post("/auction/reset")
async def reset_auction():
    """Reset nomination cursor and all team states."""
    svc = get_service()
    svc.current_index = 0
    svc._reset_team_states()
    svc.latest_results = None
    svc.simulation_running = False
    return {"reset": True}


# ─────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────

@router.post("/simulate/next-player")
async def simulate_next_player(req: SimulationRequest):
    svc = get_service()
    result = await svc.start_simulation("next_player", req.n_runs)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/simulate/set")
async def simulate_set(req: SimulationRequest):
    svc = get_service()
    result = await svc.start_simulation("set", req.n_runs)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/simulate/auction")
async def simulate_full_auction(req: SimulationRequest):
    svc = get_service()
    result = await svc.start_simulation("full_auction", req.n_runs)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/simulation/stop")
async def stop_simulation():
    svc = get_service()
    svc.stop_simulation()
    return {"stopped": True}


@router.get("/simulation/status")
async def simulation_status():
    svc = get_service()
    return svc.get_simulation_status()


@router.get("/results/latest")
async def get_latest_results():
    svc = get_service()
    if not svc.latest_results:
        return {"available": False, "distributions": []}
    return {"available": True, **svc.latest_results}


# ─────────────────────────────────────────
# RAG
# ─────────────────────────────────────────

@router.post("/rag/query")
async def rag_query(body: RAGQuery):
    svc = get_service()
    result = await svc.rag_query(body.question, body.player_name)
    return result
