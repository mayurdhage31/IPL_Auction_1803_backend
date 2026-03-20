"""
Monte Carlo Simulation Engine — heuristic-based, no LLM calls.

Runs 100–1000 simulations per player/set/full-auction and produces
realistic price distributions using team DNA, squad state, and randomisation.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ..config import (
    Player, PlayerRole, PlayerOrigin, AuctionConfig,
    TEAM_DNA, ACTIVE_TEAMS, TEAM_FULL_NAMES, MEGA_AUCTION_CONFIG,
    get_bid_increment,
)


# ─────────────────────────────────────────
# Lightweight team state used inside each simulation run
# ─────────────────────────────────────────

@dataclass
class SimTeamState:
    team_code: str
    remaining_purse: int
    total_purse: int
    squad_size: int = 0
    max_squad: int = 25
    min_squad: int = 18
    overseas_count: int = 0
    max_overseas: int = 8
    role_counts: dict = field(default_factory=dict)  # role_value -> count

    @property
    def slots_remaining(self) -> int:
        return self.max_squad - self.squad_size

    @property
    def overseas_slots(self) -> int:
        return self.max_overseas - self.overseas_count

    @property
    def min_slots_to_fill(self) -> int:
        return max(0, self.min_squad - self.squad_size)

    @property
    def effective_max_bid(self) -> int:
        reserve = max(0, self.min_slots_to_fill - 1) * 2_000_000
        return max(0, self.remaining_purse - reserve)

    def can_buy(self, player: Player) -> bool:
        if self.slots_remaining <= 0:
            return False
        if player.origin == PlayerOrigin.OVERSEAS and self.overseas_slots <= 0:
            return False
        if self.remaining_purse < player.base_price:
            return False
        return True

    def buy(self, player: Player, price: int):
        self.remaining_purse -= price
        self.squad_size += 1
        if player.origin == PlayerOrigin.OVERSEAS:
            self.overseas_count += 1
        r = player.role.value
        self.role_counts[r] = self.role_counts.get(r, 0) + 1


# ─────────────────────────────────────────
# Heuristic max-willing-price calculation
# ─────────────────────────────────────────

# Team-level risk multipliers derived from TEAM_DNA
_RISK_MULT = {
    "very high": 1.35, "high": 1.20, "medium-high": 1.10,
    "medium": 1.00, "low-medium": 0.88, "low": 0.78,
}

# Role scarcity thresholds (if a team has < this many, they want more)
_ROLE_NEED_THRESH = {
    "Batsman": 4, "Bowler": 4, "All-Rounder": 2, "Wicket Keeper": 1,
}


def _calc_max_willing(team_code: str, player: Player, state: SimTeamState) -> int:
    """
    Estimate the max price a team will pay, incorporating:
    - Historical fair value of the player
    - Team DNA role preferences
    - Role scarcity in current squad
    - Purse utilisation
    - Team risk personality
    - Gaussian noise for realism
    """
    dna = TEAM_DNA[team_code]
    prefs = dna["historical_preferences"]

    base = player.stats.get("avg_price", player.base_price)

    # Role preference multiplier
    role_pref = 1.15 if player.role.value in prefs.get("prefer_roles", []) else 0.92

    # Origin preference
    origin_pref_str = prefs.get("prefer_origin", "balanced")
    if origin_pref_str == "Indian-heavy" and player.origin == PlayerOrigin.INDIAN:
        origin_mult = 1.10
    elif origin_pref_str == "overseas-heavy" and player.origin == PlayerOrigin.OVERSEAS:
        origin_mult = 1.10
    else:
        origin_mult = 1.00

    # Scarcity: how badly does this team need this role?
    current_role_count = state.role_counts.get(player.role.value, 0)
    thresh = _ROLE_NEED_THRESH.get(player.role.value, 3)
    if current_role_count == 0:
        scarcity = 1.30
    elif current_role_count < thresh // 2:
        scarcity = 1.15
    elif current_role_count < thresh:
        scarcity = 1.00
    else:
        scarcity = 0.72   # role already well covered

    # Purse utilisation — teams with more purse left bid higher
    purse_pct = state.remaining_purse / state.total_purse if state.total_purse > 0 else 0
    purse_mult = 1.12 if purse_pct > 0.6 else 0.92 if purse_pct < 0.3 else 1.00

    # Team personality risk tolerance
    risk_str = prefs.get("risk_tolerance", "medium")
    risk_mult = _RISK_MULT.get(risk_str, 1.0)

    # Urgency: if team needs many more players before pool thins out
    urgency = 1.0 + (state.min_slots_to_fill / max(state.slots_remaining, 1)) * 0.2

    # Gaussian noise — each simulation run produces a different outcome
    noise = random.gauss(1.0, 0.18)
    noise = max(0.60, min(noise, 1.60))   # cap extreme outliers

    max_willing = int(base * role_pref * origin_mult * scarcity * purse_mult * risk_mult * urgency * noise)

    # Cannot exceed effective budget
    max_willing = min(max_willing, state.effective_max_bid)

    return max(player.base_price, max_willing)


# ─────────────────────────────────────────
# Single-player simulation
# ─────────────────────────────────────────

def simulate_single_auction(
    player: Player,
    team_states: dict[str, SimTeamState],
) -> tuple[Optional[str], int]:
    """
    Simulate one auction of a single player.
    Returns (winning_team_code, final_price) or (None, 0) if unsold.
    """
    current_bid = player.base_price
    highest_bidder: Optional[str] = None

    # Determine which teams are eligible + their max willing price
    max_willing: dict[str, int] = {}
    for code, state in team_states.items():
        if not state.can_buy(player):
            continue
        mw = _calc_max_willing(code, player, state)
        if mw >= current_bid:
            max_willing[code] = mw

    if not max_willing:
        return None, 0

    # Simulate incremental bidding
    for _ in range(50):   # max 50 rounds per player
        next_bid = current_bid + get_bid_increment(current_bid)

        # Find teams willing to bid next amount
        eligible = {
            code: mw for code, mw in max_willing.items()
            if code != highest_bidder
            and mw >= next_bid
            and team_states[code].effective_max_bid >= next_bid
        }

        if not eligible:
            break

        # Weighted random winner (higher willingness = higher probability)
        codes = list(eligible.keys())
        weights = [eligible[c] for c in codes]
        winner = random.choices(codes, weights=weights, k=1)[0]

        highest_bidder = winner
        current_bid = next_bid

        # Remove teams that can no longer afford to bid higher
        max_willing = {
            code: mw for code, mw in max_willing.items()
            if mw >= current_bid + get_bid_increment(current_bid)
            or code == highest_bidder
        }

        # If only the highest bidder is left, stop
        remaining_challengers = [c for c in max_willing if c != highest_bidder]
        if not remaining_challengers:
            break

    if highest_bidder:
        return highest_bidder, current_bid
    return None, 0


# ─────────────────────────────────────────
# Fresh team state factory
# ─────────────────────────────────────────

def _fresh_team_states(config: AuctionConfig) -> dict[str, SimTeamState]:
    return {
        code: SimTeamState(
            team_code=code,
            remaining_purse=config.total_purse,
            total_purse=config.total_purse,
            max_squad=config.max_squad_size,
            min_squad=config.min_squad_size,
        )
        for code in ACTIVE_TEAMS
    }


# ─────────────────────────────────────────
# Monte Carlo runners
# ─────────────────────────────────────────

def run_player_simulation(
    player: Player,
    n_runs: int = 100,
    config: AuctionConfig = None,
    stop_flag: Optional[list] = None,   # pass [False]; set [0]=True to abort
) -> dict:
    """
    Simulate N independent auctions of a single player.
    Each run uses fresh team states so outcomes are independent.
    Returns raw price list and team win counts.
    """
    if config is None:
        config = MEGA_AUCTION_CONFIG

    prices: list[float] = []
    team_wins: dict[str, int] = {code: 0 for code in ACTIVE_TEAMS}
    unsold_count = 0

    for _ in range(n_runs):
        if stop_flag and stop_flag[0]:
            break
        states = _fresh_team_states(config)
        winner, price = simulate_single_auction(player, states)
        if winner:
            prices.append(price / 1e7)
            team_wins[winner] = team_wins.get(winner, 0) + 1
        else:
            unsold_count += 1

    return {
        "prices_cr": prices,
        "team_wins": team_wins,
        "unsold_count": unsold_count,
        "n_runs": n_runs,
    }


def run_set_simulation(
    players: list[Player],
    n_runs: int = 100,
    config: AuctionConfig = None,
    stop_flag: Optional[list] = None,
) -> dict[str, dict]:
    """
    Simulate N full-set auctions.
    Teams share purse/squad state across players within each run,
    capturing inter-player budget dependencies.
    Returns per-player price lists and win counts.
    """
    if config is None:
        config = MEGA_AUCTION_CONFIG

    results: dict[str, dict] = {
        p.name: {"prices_cr": [], "team_wins": {c: 0 for c in ACTIVE_TEAMS}, "unsold_count": 0}
        for p in players
    }

    for _ in range(n_runs):
        if stop_flag and stop_flag[0]:
            break

        states = _fresh_team_states(config)
        shuffled = list(players)
        random.shuffle(shuffled)  # randomise bidding order within set

        for player in shuffled:
            winner, price = simulate_single_auction(player, states)
            entry = results[player.name]
            if winner:
                entry["prices_cr"].append(price / 1e7)
                entry["team_wins"][winner] = entry["team_wins"].get(winner, 0) + 1
                states[winner].buy(player, price)
            else:
                entry["unsold_count"] += 1

    for name in results:
        results[name]["n_runs"] = n_runs

    return results


def run_full_auction_simulation(
    marquee: list[Player],
    capped: list[Player],
    uncapped: list[Player],
    n_runs: int = 100,
    config: AuctionConfig = None,
    stop_flag: Optional[list] = None,
) -> dict[str, dict]:
    """
    Simulate N complete auctions (marquee → capped → uncapped).
    Player order within each set is randomised per run.
    """
    if config is None:
        config = MEGA_AUCTION_CONFIG

    all_players = marquee + capped + uncapped
    results: dict[str, dict] = {
        p.name: {"prices_cr": [], "team_wins": {c: 0 for c in ACTIVE_TEAMS}, "unsold_count": 0}
        for p in all_players
    }

    for _ in range(n_runs):
        if stop_flag and stop_flag[0]:
            break

        states = _fresh_team_states(config)

        # Within each set, shuffle; sets themselves stay in marquee→capped→uncapped order
        for group in [list(marquee), list(capped), list(uncapped)]:
            random.shuffle(group)
            for player in group:
                winner, price = simulate_single_auction(player, states)
                entry = results[player.name]
                if winner:
                    entry["prices_cr"].append(price / 1e7)
                    entry["team_wins"][winner] = entry["team_wins"].get(winner, 0) + 1
                    states[winner].buy(player, price)
                else:
                    entry["unsold_count"] += 1

    for name in results:
        results[name]["n_runs"] = n_runs

    return results


# ─────────────────────────────────────────
# Statistics computation
# ─────────────────────────────────────────

def compute_price_distribution(
    player: Player,
    sim_data: dict,
) -> dict:
    """
    Convert raw simulation output into a PriceDistribution-compatible dict.
    """
    prices = sim_data["prices_cr"]
    n_runs = sim_data["n_runs"]
    unsold = sim_data["unsold_count"]
    team_wins = sim_data["team_wins"]

    sold_count = len(prices)
    unsold_prob = unsold / n_runs if n_runs > 0 else 0.0

    if prices:
        arr = np.array(prices)
        dist = {
            "min_cr": float(np.min(arr)),
            "p10_cr": float(np.percentile(arr, 10)),
            "p25_cr": float(np.percentile(arr, 25)),
            "median_cr": float(np.median(arr)),
            "p75_cr": float(np.percentile(arr, 75)),
            "p90_cr": float(np.percentile(arr, 90)),
            "max_cr": float(np.max(arr)),
            "mean_cr": float(np.mean(arr)),
        }
    else:
        dist = {k: 0.0 for k in ["min_cr", "p10_cr", "p25_cr", "median_cr",
                                   "p75_cr", "p90_cr", "max_cr", "mean_cr"]}

    # Team win probabilities (only among sold lots)
    team_win_probs: dict[str, float] = {}
    if sold_count > 0:
        team_win_probs = {code: wins / n_runs for code, wins in team_wins.items() if wins > 0}

    return {
        "player_name": player.name,
        "role": player.role.value,
        "origin": player.origin.value,
        "n_runs": n_runs,
        "sold_count": sold_count,
        "unsold_count": unsold,
        "unsold_probability": round(unsold_prob, 3),
        **dist,
        "team_win_probabilities": team_win_probs,
    }
