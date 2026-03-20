"""
Auction Service — singleton that owns the global auction state.

Manages:
- Player pool (loaded from CSVs)
- Current nominated player and position in pool
- Team states (squad + purse)
- Simulation lifecycle (start / stop / results)
"""

import asyncio
import random
import time
import logging
from pathlib import Path
from typing import Optional

from ..config import (
    ACTIVE_TEAMS, TEAM_FULL_NAMES, MEGA_AUCTION_CONFIG, AuctionSetType,
    PlayerRole, PlayerOrigin,
)
from ..data.processor import AuctionDataProcessor
from ..auction.simulation import (
    run_player_simulation,
    run_set_simulation,
    run_full_auction_simulation,
    compute_price_distribution,
    SimTeamState,
)
from ..rag.gemini_rag import GeminiRAG

logger = logging.getLogger(__name__)

# Base price slabs for random nomination assignment (20L – 2Cr)
_BASE_PRICE_SLABS = [
    2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000,
    15_000_000, 20_000_000,
]


def _random_base_price(player) -> int:
    """Assign a random-ish base price weighted toward player's tier."""
    avg = player.stats.get("avg_price", 5_000_000)
    # Base price ≈ 10-30% of avg historical price
    estimated = avg * random.uniform(0.10, 0.30)
    for slab in _BASE_PRICE_SLABS:
        if estimated <= slab:
            return slab
    return 20_000_000  # 2 Cr max


class AuctionService:
    """
    Global singleton managing auction state across all API calls.
    Thread-safe using asyncio Lock.
    """

    _instance: Optional["AuctionService"] = None

    def __init__(self):
        self._lock = asyncio.Lock()

        # Data processor
        self.processor: Optional[AuctionDataProcessor] = None

        # Player pool organised by set
        self.marquee_pool: list = []
        self.capped_pool: list = []
        self.uncapped_pool: list = []
        self.all_pool: list = []           # flat list in auction order

        # Current nomination cursor
        self.current_index: int = 0

        # Team live states (reset each time we start a new auction session)
        self.team_states: dict[str, SimTeamState] = {}
        self.team_purchases: dict[str, list] = {}  # team_code -> [{name, role, origin, price_cr}]

        # Simulation state
        self.simulation_running: bool = False
        self.simulation_progress: int = 0
        self.simulation_current_run: int = 0
        self.simulation_total_runs: int = 0
        self.simulation_mode: str = ""
        self._stop_flag: list = [False]   # mutable list passed into simulation workers
        self.latest_results: Optional[dict] = None

        # RAG
        self.rag: Optional[GeminiRAG] = None
        self.selected_team: str = "CSK"

    # ─────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────

    async def initialise(self, batting_csv: str, bowling_csv: str, docs_dir: str, gemini_key: str):
        """Load data, build pools, wire RAG. Call once at startup."""
        async with self._lock:
            logger.info("AuctionService initialising…")

            # Load data
            self.processor = AuctionDataProcessor(batting_csv, bowling_csv)

            # Build pools
            all_players = self.processor.generate_auction_pool()
            self.marquee_pool = [p for p in all_players if p.set_type == AuctionSetType.MARQUEE]
            self.capped_pool = [p for p in all_players if p.set_type == AuctionSetType.CAPPED]
            self.uncapped_pool = [p for p in all_players if p.set_type == AuctionSetType.UNCAPPED]
            self.all_pool = self.marquee_pool + self.capped_pool + self.uncapped_pool

            # Randomise base prices for display
            for p in self.all_pool:
                p.base_price = _random_base_price(p)

            # Initialise team states
            self._reset_team_states()

            logger.info(
                f"Loaded {len(self.all_pool)} players: "
                f"{len(self.marquee_pool)} marquee, "
                f"{len(self.capped_pool)} capped, "
                f"{len(self.uncapped_pool)} uncapped."
            )

            # Wire RAG (non-blocking — uploads happen in background)
            if gemini_key:
                self.rag = GeminiRAG(api_key=gemini_key, docs_dir=docs_dir)
                asyncio.create_task(self._init_rag())

    async def _init_rag(self):
        """Upload docs to Gemini in the background."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.rag.initialise)
            logger.info("RAG initialised successfully.")
        except Exception as e:
            logger.error(f"RAG init failed: {e}")

    def _reset_team_states(self):
        config = MEGA_AUCTION_CONFIG
        self.team_states = {
            code: SimTeamState(
                team_code=code,
                remaining_purse=config.total_purse,
                total_purse=config.total_purse,
                max_squad=config.max_squad_size,
                min_squad=config.min_squad_size,
            )
            for code in ACTIVE_TEAMS
        }
        self.team_purchases = {code: [] for code in ACTIVE_TEAMS}

    # ─────────────────────────────────────────
    # Nomination
    # ─────────────────────────────────────────

    def get_current_player(self) -> Optional[dict]:
        if not self.all_pool or self.current_index >= len(self.all_pool):
            return None
        p = self.all_pool[self.current_index]
        return self._player_to_dict(p, self.current_index)

    def advance_player(self) -> Optional[dict]:
        """Move nomination to the next player."""
        if self.current_index < len(self.all_pool) - 1:
            self.current_index += 1
        return self.get_current_player()

    def _player_to_dict(self, p, idx: int) -> dict:
        s = p.stats
        batting_cat = None
        if "batting_sr" in s and s["batting_sr"]:
            batting_cat = "Power Hitter" if s["batting_sr"] > 140 else "Anchor" if s["batting_sr"] < 120 else "Middle Order"
        bowling_cat = s.get("bowler_category") or None

        return {
            "name": p.name,
            "role": p.role.value,
            "origin": p.origin.value,
            "base_price": p.base_price,
            "base_price_cr": round(p.base_price / 1e7, 2),
            "set_type": p.set_type.value,
            "batting_category": batting_cat,
            "bowling_category": bowling_cat,
            "avg_price_cr": round(s["avg_price"] / 1e7, 2) if s.get("avg_price") else None,
            "historical_prices": p.historical_prices[-5:],
            "index_in_pool": idx,
            "total_in_pool": len(self.all_pool),
        }

    # ─────────────────────────────────────────
    # Team Squad (for right-side panel)
    # ─────────────────────────────────────────

    def get_team_squad(self, team_code: str) -> dict:
        state = self.team_states.get(team_code)
        if not state:
            return {}

        purchases = self.team_purchases.get(team_code, [])

        # Count roles
        role_counts = {"Batsman": 0, "Bowler": 0, "All-Rounder": 0, "Wicket Keeper": 0}
        indian_count = 0
        for p in purchases:
            role_counts[p.get("role", "Batsman")] = role_counts.get(p.get("role", "Batsman"), 0) + 1
            if p.get("origin") == "Indian":
                indian_count += 1

        return {
            "team_code": team_code,
            "team_name": TEAM_FULL_NAMES.get(team_code, team_code),
            "remaining_purse_cr": round(state.remaining_purse / 1e7, 2),
            "total_purse_cr": round(state.total_purse / 1e7, 2),
            "squad": purchases,
            "counters": {
                "batsmen": role_counts["Batsman"],
                "bowlers": role_counts["Bowler"],
                "allrounders": role_counts["All-Rounder"],
                "wicketkeepers": role_counts["Wicket Keeper"],
                "indians": indian_count,
                "overseas": state.overseas_count,
                "overseas_max": state.max_overseas,
            },
        }

    @staticmethod
    def _safe_float(val, decimals: int = 2):
        """Return rounded float or None, converting nan/inf to None."""
        import math
        if val is None:
            return None
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return None
            return round(f, decimals)
        except (TypeError, ValueError):
            return None

    def get_player_detail(self, player) -> dict:
        """Return a rich player profile dict for the Scout panel."""
        s = player.stats
        sr = self._safe_float(s.get("batting_sr"))
        batting_cat = None
        if sr is not None:
            batting_cat = "Power Hitter" if sr > 140 else "Anchor" if sr < 120 else "Middle Order"
        bowling_cat = s.get("bowler_category") or None
        idx = next((i for i, p in enumerate(self.all_pool) if p.name == player.name), -1)

        return {
            "name": player.name,
            "role": player.role.value,
            "origin": player.origin.value,
            "base_price": player.base_price,
            "base_price_cr": round(player.base_price / 1e7, 2),
            "set_type": player.set_type.value,
            "batting_category": batting_cat,
            "bowling_category": bowling_cat,
            "avg_price_cr": self._safe_float(s.get("avg_price", 0) / 1e7 if s.get("avg_price") else None),
            "max_price_cr": self._safe_float(s.get("max_price", 0) / 1e7 if s.get("max_price") else None),
            "latest_price_cr": self._safe_float(s.get("latest_price", 0) / 1e7 if s.get("latest_price") else None),
            "latest_year": s.get("latest_year"),
            "peak_year": s.get("peak_year"),
            "trajectory": s.get("trajectory"),
            "price_trend": self._safe_float(s.get("price_trend"), 4),
            "volatility": self._safe_float(s.get("volatility"), 3),
            "auction_appearances": s.get("auction_appearances", 0),
            "total_teams": s.get("total_teams", 0),
            # Batting stats
            "batting_avg": self._safe_float(s.get("batting_avg")),
            "batting_sr": sr,
            "total_runs": s.get("total_runs"),
            # Bowling stats
            "wickets": s.get("wickets"),
            "economy": self._safe_float(s.get("economy")),
            "bowling_sr": self._safe_float(s.get("bowling_sr")),
            # Full price history (all records, not just last 5)
            "historical_prices": player.historical_prices,
            "index_in_pool": idx,
            "total_in_pool": len(self.all_pool),
        }

    def get_players_for_team(self, team_code: str) -> list[dict]:
        """Return players who have historically been bought by a given team."""
        results = []
        for p in self.all_pool:
            if any(h.get("team_code") == team_code for h in p.historical_prices):
                results.append({
                    "name": p.name,
                    "role": p.role.value,
                    "origin": p.origin.value,
                    "set_type": p.set_type.value,
                    "avg_price_cr": round(p.stats["avg_price"] / 1e7, 2) if p.stats.get("avg_price") else 0,
                })
        results.sort(key=lambda x: x["avg_price_cr"], reverse=True)
        return results

    def get_all_teams(self) -> list[dict]:
        return [
            {
                "code": code,
                "name": TEAM_FULL_NAMES[code],
                "remaining_purse_cr": round(self.team_states[code].remaining_purse / 1e7, 2),
            }
            for code in ACTIVE_TEAMS
        ]

    # ─────────────────────────────────────────
    # Simulation
    # ─────────────────────────────────────────

    def get_simulation_status(self) -> dict:
        return {
            "running": self.simulation_running,
            "progress": self.simulation_progress,
            "current_run": self.simulation_current_run,
            "total_runs": self.simulation_total_runs,
            "mode": self.simulation_mode,
        }

    async def start_simulation(self, mode: str, n_runs: int) -> dict:
        """Launch simulation in a background thread."""
        if self.simulation_running:
            return {"error": "Simulation already running. Stop it first."}

        if not self.all_pool:
            return {"error": "No players loaded. Initialise the service first."}

        self._stop_flag = [False]
        self.simulation_running = True
        self.simulation_progress = 0
        self.simulation_current_run = 0
        self.simulation_total_runs = n_runs
        self.simulation_mode = mode
        self.latest_results = None

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self._run_simulation_sync, mode, n_runs)

        return {"started": True, "mode": mode, "n_runs": n_runs}

    def _run_simulation_sync(self, mode: str, n_runs: int):
        """Synchronous simulation work — runs in thread pool."""
        try:
            start = time.time()

            if mode == "next_player":
                player = self.all_pool[self.current_index] if self.current_index < len(self.all_pool) else None
                if not player:
                    self.simulation_running = False
                    return
                raw = run_player_simulation(player, n_runs=n_runs, stop_flag=self._stop_flag)
                dist = compute_price_distribution(player, raw)
                distributions = [dist]

            elif mode == "set":
                # Determine which set the current player belongs to
                cur = self.all_pool[self.current_index] if self.current_index < len(self.all_pool) else None
                if cur and cur.set_type == AuctionSetType.MARQUEE:
                    pool = self.marquee_pool
                elif cur and cur.set_type == AuctionSetType.UNCAPPED:
                    pool = self.uncapped_pool
                else:
                    pool = self.capped_pool

                # Limit to reasonable set size for performance
                pool = pool[:30]
                raw_set = run_set_simulation(pool, n_runs=n_runs, stop_flag=self._stop_flag)
                distributions = [
                    compute_price_distribution(p, raw_set[p.name])
                    for p in pool
                    if p.name in raw_set
                ]

            elif mode == "full_auction":
                # Cap each group for performance
                marquee = self.marquee_pool[:20]
                capped = self.capped_pool[:40]
                uncapped = self.uncapped_pool[:30]
                raw_full = run_full_auction_simulation(
                    marquee, capped, uncapped, n_runs=n_runs, stop_flag=self._stop_flag
                )
                all_p = marquee + capped + uncapped
                distributions = [
                    compute_price_distribution(p, raw_full[p.name])
                    for p in all_p
                    if p.name in raw_full
                ]
            else:
                distributions = []

            elapsed_ms = (time.time() - start) * 1000
            self.latest_results = {
                "mode": mode,
                "n_runs": n_runs,
                "distributions": distributions,
                "duration_ms": round(elapsed_ms, 1),
            }
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self.simulation_running = False
            self.simulation_progress = 100

    def stop_simulation(self):
        """Signal the running simulation to stop."""
        self._stop_flag[0] = True
        self.simulation_running = False

    # ─────────────────────────────────────────
    # RAG
    # ─────────────────────────────────────────

    async def rag_query(self, question: str, player_name: Optional[str] = None) -> dict:
        if not self.rag:
            return {
                "question": question,
                "answer": "RAG not available. Set GEMINI_API_KEY in .env.",
                "sources_used": [],
            }

        player_context = ""
        if player_name and self.processor:
            player_context = self.processor.get_price_prediction_context(player_name)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.rag.query(question, player_context)
        )
        return {"question": question, **result}


# ─────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────

_service_instance: Optional[AuctionService] = None


def get_service() -> AuctionService:
    global _service_instance
    if _service_instance is None:
        _service_instance = AuctionService()
    return _service_instance
