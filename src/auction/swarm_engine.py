"""
Swarm Intelligence Engine — MiroFish-style Emergent Auction Dynamics

This replaces the old sequential "ask each agent independently" approach
with a proper multi-agent interaction system where:

1. SHARED OBSERVATION SPACE — All agents see the same auction room state,
   including who's bidding, body language signals, and spending patterns.

2. AGENT MEMORY (replaces Zep) — Each agent maintains a persistent memory
   of the entire auction: who bought whom, which teams they're competing
   with, patterns they've noticed, grudges, and evolving strategy.

3. SOCIAL ACTIONS — Agents don't just bid/pass. They can:
   - Signal interest (paddle up early = strong signal to rivals)
   - Bluff (bid on a player they don't want to drain rival purse)
   - React to rival bids (visible frustration, strategic retreat)
   - Adjust strategy mid-auction based on observed behavior

4. EMERGENT DYNAMICS — Bidding wars, tactical retreats, price inflation,
   panic buying, and coalition-like behavior all emerge from agent
   interaction rather than being explicitly programmed.

5. ROUND-BASED INTERACTION (replaces OASIS parallel sim) — Each bidding
   round, ALL agents observe the current state and can communicate
   reactions/signals before the next bid. This creates the feedback
   loops that produce emergent behavior.

Architecture:
  SwarmState (shared world state)
    └── AgentMemory × 10 (persistent per-agent memory)
    └── InteractionLog (all observable actions)
    └── SocialSignals (body language, reactions)
        └── SwarmAuctionEngine orchestrates rounds with full observability
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SignalType(str, Enum):
    """Observable social signals in the auction room."""
    PADDLE_UP_EARLY = "paddle_up_early"       # Strong interest signal
    PADDLE_UP_LATE = "paddle_up_late"         # Reluctant but still in
    DRAMATIC_PAUSE = "dramatic_pause"          # Thinking hard, might drop
    CONFIDENT_NOD = "confident_nod"            # Will go higher
    VISIBLE_FRUSTRATION = "visible_frustration"  # Losing a target
    STRATEGIC_RETREAT = "strategic_retreat"     # Dropping out deliberately
    CELEBRATORY = "celebratory"                # Won a key player
    URGENT_HUDDLE = "urgent_huddle"            # Team discussing frantically
    CALM_OBSERVATION = "calm_observation"       # Watching, not participating
    BLUFF_BID = "bluff_bid"                    # Bidding to inflate price


@dataclass
class SocialSignal:
    """An observable action/reaction from a team in the auction room."""
    team_code: str
    signal_type: SignalType
    target_player: str
    message: str         # What other agents can observe
    private_intent: str  # True internal reasoning (hidden from others)
    round_num: int
    timestamp: float = 0.0


@dataclass
class AgentMemory:
    """
    Persistent memory for a single team agent across the entire auction.
    This is our equivalent of MiroFish's Zep Cloud memory.
    """
    team_code: str

    # Factual memory
    players_won: list = field(default_factory=list)         # [{name, price, role}]
    players_lost: list = field(default_factory=list)         # [{name, price, won_by}]
    players_targeted: list = field(default_factory=list)     # pre-auction wishlist
    money_spent: float = 0.0

    # Social memory — what we've observed about OTHER teams
    rival_profiles: dict = field(default_factory=dict)  # {team_code: {observations}}
    # e.g. {"RCB": {"aggression_level": "high", "targets_roles": ["Batsman"],
    #                "remaining_purse_estimate": 45.0, "bidding_wars_entered": 3}}

    # Emotional state
    frustration_level: float = 0.0      # 0-1, rises when losing targets
    confidence_level: float = 0.5       # 0-1, rises with good buys
    urgency_level: float = 0.0          # 0-1, rises as slots need filling
    satisfaction_level: float = 0.5     # 0-1, overall auction satisfaction

    # Strategic memory
    bidding_wars_entered: int = 0
    bidding_wars_won: int = 0
    bidding_wars_lost: int = 0
    bluffs_attempted: int = 0
    times_outbid: int = 0
    max_price_paid_cr: float = 0.0

    # Pattern observations
    observed_patterns: list = field(default_factory=list)
    # e.g. ["RCB always enters early for batsmen",
    #        "MI drops out above 8Cr for uncapped players",
    #        "PBKS has been aggressive — may run out of purse"]

    strategy_adjustments: list = field(default_factory=list)
    # e.g. ["Switched to targeting bowlers after losing 2 batsmen",
    #        "Increased aggression — running out of marquee targets"]

    def add_observation(self, obs: str):
        self.observed_patterns.append(obs)
        # Keep last 20 observations
        if len(self.observed_patterns) > 20:
            self.observed_patterns = self.observed_patterns[-20:]

    def add_strategy_shift(self, shift: str):
        self.strategy_adjustments.append(shift)

    def update_rival_profile(self, rival_code: str, key: str, value):
        if rival_code not in self.rival_profiles:
            self.rival_profiles[rival_code] = {}
        self.rival_profiles[rival_code][key] = value

    def get_emotional_summary(self) -> str:
        emotions = []
        if self.frustration_level > 0.6:
            emotions.append("frustrated (losing targets)")
        if self.urgency_level > 0.7:
            emotions.append("urgent (slots to fill)")
        if self.confidence_level > 0.7:
            emotions.append("confident (good buys so far)")
        if self.confidence_level < 0.3:
            emotions.append("anxious (poor auction so far)")
        if self.satisfaction_level > 0.7:
            emotions.append("satisfied")
        return ", ".join(emotions) if emotions else "neutral, focused"

    def to_context_string(self) -> str:
        """Serialize memory into context for LLM prompts."""
        lines = [
            f"## Your Auction Memory ({self.team_code})",
            f"Players won: {len(self.players_won)} | Lost: {len(self.players_lost)}",
            f"Spent: ₹{self.money_spent:.2f}Cr | Max single purchase: ₹{self.max_price_paid_cr:.2f}Cr",
            f"Bidding wars: {self.bidding_wars_won}W-{self.bidding_wars_lost}L out of {self.bidding_wars_entered} entered",
            f"Emotional state: {self.get_emotional_summary()}",
        ]

        if self.rival_profiles:
            lines.append("\n### Rival Intelligence:")
            for rival, profile in self.rival_profiles.items():
                lines.append(f"  {rival}: {json.dumps(profile)}")

        if self.observed_patterns:
            lines.append("\n### Patterns You've Noticed:")
            for p in self.observed_patterns[-5:]:
                lines.append(f"  - {p}")

        if self.strategy_adjustments:
            lines.append("\n### Your Strategy Shifts:")
            for s in self.strategy_adjustments[-3:]:
                lines.append(f"  - {s}")

        return "\n".join(lines)


@dataclass
class SwarmState:
    """
    The shared world state visible to all agents.
    This is our equivalent of MiroFish's simulation environment.
    """
    # Current auction state
    current_player: Optional[str] = None
    current_bid: int = 0
    current_highest_bidder: Optional[str] = None
    bidding_round: int = 0
    auction_phase: str = "pre_auction"
    lot_number: int = 0

    # Full auction history (visible to all)
    completed_lots: list = field(default_factory=list)
    # [{player, sold, winner, price_cr, bidding_rounds, competing_teams}]

    # Team states (publicly visible information)
    team_purses: dict = field(default_factory=dict)       # {code: remaining_cr}
    team_squad_sizes: dict = field(default_factory=dict)   # {code: count}
    team_overseas_counts: dict = field(default_factory=dict)  # {code: count}

    # Social signals this lot (visible to all)
    signals_this_lot: list = field(default_factory=list)

    # Interaction log (full history of observable actions)
    interaction_log: list = field(default_factory=list)

    # Market dynamics
    total_money_spent_cr: float = 0.0
    avg_price_so_far_cr: float = 0.0
    biggest_purchase_cr: float = 0.0
    unsold_count: int = 0
    players_remaining: int = 0

    def add_signal(self, signal: SocialSignal):
        self.signals_this_lot.append(signal)
        self.interaction_log.append({
            "lot": self.lot_number,
            "round": self.bidding_round,
            "team": signal.team_code,
            "signal": signal.signal_type.value,
            "message": signal.message,
        })

    def clear_lot_signals(self):
        self.signals_this_lot = []

    def get_room_atmosphere(self) -> str:
        """Describe the current auction room vibe based on recent activity."""
        if not self.completed_lots:
            return "Tense anticipation. The auction is just beginning."

        recent = self.completed_lots[-3:]
        avg_recent = sum(l.get("price_cr", 0) for l in recent) / len(recent) if recent else 0
        wars = sum(1 for l in recent if l.get("bidding_rounds", 0) > 3)

        if wars >= 2:
            atmosphere = "Heated — multiple bidding wars in recent lots. Teams are getting aggressive."
        elif avg_recent > self.avg_price_so_far_cr * 1.5:
            atmosphere = "Prices are escalating. The room feels urgent. Big money is flowing."
        elif self.unsold_count > len(self.completed_lots) * 0.3:
            atmosphere = "Cautious — many players going unsold. Teams are being conservative."
        else:
            atmosphere = "Steady. Teams are bidding methodically with occasional surges."

        # Add signal-based color
        active_signals = [s for s in self.signals_this_lot]
        frustrations = [s for s in active_signals if s.signal_type == SignalType.VISIBLE_FRUSTRATION]
        if frustrations:
            atmosphere += f" {frustrations[0].team_code} showing visible frustration."

        return atmosphere

    def to_context_string(self) -> str:
        """Full room state for agent prompts."""
        lines = [
            f"## Auction Room State (Lot #{self.lot_number})",
            f"Phase: {self.auction_phase} | Players remaining: {self.players_remaining}",
            f"Market so far: {len(self.completed_lots)} sold, {self.unsold_count} unsold",
            f"Total spent: ₹{self.total_money_spent_cr:.1f}Cr | Avg price: ₹{self.avg_price_so_far_cr:.2f}Cr",
            f"Biggest purchase: ₹{self.biggest_purchase_cr:.2f}Cr",
            f"\nRoom atmosphere: {self.get_room_atmosphere()}",
            f"\n### Team Purse Board (visible to all):",
        ]

        for code in sorted(self.team_purses.keys()):
            purse = self.team_purses[code]
            squad = self.team_squad_sizes.get(code, 0)
            overseas = self.team_overseas_counts.get(code, 0)
            lines.append(f"  {code}: ₹{purse:.1f}Cr | {squad} players | {overseas}/8 overseas")

        if self.current_player:
            lines.append(f"\n### Currently on the block: {self.current_player}")
            lines.append(f"Current bid: ₹{self.current_bid/1e7:.2f}Cr by {self.current_highest_bidder or 'nobody'}")
            lines.append(f"Bidding round: {self.bidding_round}")

        # Recent signals
        if self.signals_this_lot:
            lines.append(f"\n### What you can see in the room right now:")
            for s in self.signals_this_lot[-8:]:
                lines.append(f"  [{s.team_code}] {s.message}")

        # Last 3 completed lots
        if self.completed_lots:
            lines.append(f"\n### Recent results:")
            for lot in self.completed_lots[-3:]:
                if lot["sold"]:
                    lines.append(f"  {lot['player']} → {lot['winner']} for ₹{lot['price_cr']:.2f}Cr "
                               f"({lot['bidding_rounds']} rounds, {lot.get('competing_teams', '?')} teams competed)")
                else:
                    lines.append(f"  {lot['player']} — UNSOLD")

        return "\n".join(lines)


class SwarmAuctionEngine:
    """
    The actual swarm intelligence auction engine.

    Key differences from the old sequential engine:
    1. Every agent sees the FULL room state + social signals before deciding
    2. Agents generate observable reactions (not just bid/pass)
    3. Agent memory persists and evolves across the entire auction
    4. Emotional states influence bidding behavior
    5. Agents observe and model rival behavior patterns
    6. Between-lot strategy reflection lets agents adapt their approach
    """

    def __init__(self, team_agents: dict, verbose: bool = True):
        """
        Args:
            team_agents: dict of team_code -> TeamAgent (from agents/team_agent.py)
            verbose: print auction progress
        """
        self.agents = team_agents
        self.verbose = verbose

        # Initialize swarm components
        self.state = SwarmState()
        self.memories: dict[str, AgentMemory] = {}
        for code in team_agents:
            self.memories[code] = AgentMemory(team_code=code)

        # Initialize public team state
        for code, agent in team_agents.items():
            self.state.team_purses[code] = agent.squad.remaining_purse / 1e7
            self.state.team_squad_sizes[code] = agent.squad.current_size
            self.state.team_overseas_counts[code] = agent.squad.overseas_count

        # Results
        self.all_lots: list = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ─────────────────────────────────────────
    # Core: Swarm Bidding Round
    # ─────────────────────────────────────────

    def run_player_auction(self, player, from_config) -> dict:
        """
        Run the full swarm-style auction for a single player.

        Unlike the old engine which asks each agent in isolation,
        this runs iterative rounds where:
        1. All agents see the current state + signals
        2. Each agent decides: bid / pass / react
        3. Signals are broadcast to the room
        4. State updates, next round begins
        5. Continues until only 1 bidder remains or all pass
        """
        from ..config import get_bid_increment, PlayerOrigin, BidAction, TEAM_FULL_NAMES

        self.state.lot_number += 1
        self.state.current_player = player.name
        self.state.current_bid = player.base_price
        self.state.current_highest_bidder = None
        self.state.bidding_round = 0
        self.state.clear_lot_signals()

        lot_result = {
            "player": player.name,
            "role": player.role.value,
            "origin": player.origin.value,
            "base_price_cr": player.base_price / 1e7,
            "sold": False,
            "winner": None,
            "price_cr": 0,
            "bidding_rounds": 0,
            "bid_history": [],
            "signals": [],
            "competing_teams": 0,
        }

        self._log(f"\n{'='*60}")
        self._log(f"🏏 LOT #{self.state.lot_number}: {player.name} "
                  f"({player.role.value}, {player.origin.value})")
        self._log(f"   Base: ₹{player.base_price/1e7:.2f}Cr | "
                  f"Phase: {self.state.auction_phase}")
        self._log(f"{'='*60}")

        # Track who's still interested
        active_teams = set()
        passed_teams = set()
        teams_that_ever_bid = set()

        max_rounds = 40

        for round_num in range(1, max_rounds + 1):
            self.state.bidding_round = round_num
            round_bids = []

            # ── Step 1: Collect decisions from ALL agents ──
            for code, agent in self.agents.items():
                if code in passed_teams:
                    continue
                if code == self.state.current_highest_bidder:
                    continue  # Already the high bidder

                # Hard constraints
                if (player.origin == PlayerOrigin.OVERSEAS and
                    not agent.squad.can_buy_overseas()):
                    passed_teams.add(code)
                    continue
                if agent.squad.slots_remaining <= 0:
                    passed_teams.add(code)
                    continue

                next_bid_amount = self.state.current_bid + get_bid_increment(self.state.current_bid)
                if next_bid_amount > agent.squad.effective_max_bid:
                    passed_teams.add(code)
                    continue

                # ── THE KEY DIFFERENCE: Agent sees full swarm context ──
                decision = self._get_swarm_aware_decision(
                    agent=agent,
                    player=player,
                    next_bid_amount=next_bid_amount,
                    active_teams=active_teams,
                    passed_teams=passed_teams,
                )

                if decision["action"] == "bid":
                    round_bids.append({
                        "team_code": code,
                        "amount": next_bid_amount,
                        "reasoning": decision["reasoning"],
                        "confidence": decision["confidence"],
                        "max_willing": decision.get("max_willing_price", next_bid_amount),
                        "signal": decision.get("signal", SignalType.PADDLE_UP_EARLY),
                    })
                    teams_that_ever_bid.add(code)
                elif decision["action"] == "pass":
                    passed_teams.add(code)
                    # Generate a visible signal for passing
                    signal = SocialSignal(
                        team_code=code,
                        signal_type=decision.get("signal", SignalType.STRATEGIC_RETREAT),
                        target_player=player.name,
                        message=decision.get("visible_reaction", f"{TEAM_FULL_NAMES[code]} drops out."),
                        private_intent=decision["reasoning"],
                        round_num=round_num,
                    )
                    self.state.add_signal(signal)
                    self._log(f"  💨 {code} drops out: {signal.message}")

            # ── Step 2: Resolve this round ──
            if not round_bids:
                break  # Nobody new is bidding

            # Pick the winning bid (highest confidence → simulates auction speed)
            winning_bid = max(round_bids, key=lambda b: (b["confidence"], b["amount"]))
            self.state.current_highest_bidder = winning_bid["team_code"]
            self.state.current_bid = winning_bid["amount"]
            active_teams = {b["team_code"] for b in round_bids}

            # Broadcast the bid as a social signal
            signal_type = winning_bid.get("signal", SignalType.PADDLE_UP_EARLY)
            bid_signal = SocialSignal(
                team_code=winning_bid["team_code"],
                signal_type=signal_type,
                target_player=player.name,
                message=f"{TEAM_FULL_NAMES[winning_bid['team_code']]} bids "
                       f"₹{winning_bid['amount']/1e7:.2f}Cr"
                       f"{' with conviction!' if winning_bid['confidence'] > 0.8 else '.'}"
                       f"{' (bluff?)' if signal_type == SignalType.BLUFF_BID else ''}",
                private_intent=winning_bid["reasoning"],
                round_num=round_num,
            )
            self.state.add_signal(bid_signal)

            lot_result["bid_history"].append({
                "round": round_num,
                "team": winning_bid["team_code"],
                "amount_cr": winning_bid["amount"] / 1e7,
                "signal": signal_type.value,
                "competitors_remaining": len(active_teams),
            })

            self._log(f"  🔨 R{round_num}: {TEAM_FULL_NAMES[winning_bid['team_code']]} "
                      f"→ ₹{winning_bid['amount']/1e7:.2f}Cr "
                      f"(conf: {winning_bid['confidence']:.0%}) | "
                      f"{len(active_teams)} still competing")

            # Check if only one bidder remains willing above current price
            remaining = [code for code in self.agents
                        if code not in passed_teams
                        and code != self.state.current_highest_bidder]
            if not remaining:
                break

        # ── Step 3: Resolve the lot ──
        lot_result["bidding_rounds"] = self.state.bidding_round
        lot_result["competing_teams"] = len(teams_that_ever_bid)
        lot_result["signals"] = [
            {"team": s.team_code, "signal": s.signal_type.value, "message": s.message}
            for s in self.state.signals_this_lot
        ]

        if self.state.current_highest_bidder:
            winner = self.state.current_highest_bidder
            price = self.state.current_bid
            lot_result["sold"] = True
            lot_result["winner"] = winner
            lot_result["price_cr"] = price / 1e7

            # Update winning agent
            self.agents[winner].register_purchase(player, price)
            self.memories[winner].players_won.append({
                "name": player.name, "price_cr": price / 1e7, "role": player.role.value
            })
            self.memories[winner].money_spent += price / 1e7
            self.memories[winner].max_price_paid_cr = max(
                self.memories[winner].max_price_paid_cr, price / 1e7
            )
            self.memories[winner].confidence_level = min(1.0, self.memories[winner].confidence_level + 0.1)

            if len(teams_that_ever_bid) > 1:
                self.memories[winner].bidding_wars_entered += 1
                self.memories[winner].bidding_wars_won += 1

            # Update losing agents
            for code in teams_that_ever_bid:
                if code != winner:
                    self.agents[code].register_loss(player, winner, price)
                    self.memories[code].players_lost.append({
                        "name": player.name, "price_cr": price / 1e7, "won_by": winner
                    })
                    self.memories[code].frustration_level = min(1.0, self.memories[code].frustration_level + 0.15)
                    self.memories[code].bidding_wars_entered += 1
                    self.memories[code].bidding_wars_lost += 1
                    self.memories[code].times_outbid += 1

                    # Observe rival behavior
                    self.memories[code].update_rival_profile(
                        winner, "last_purchase", {"player": player.name, "price_cr": price / 1e7}
                    )
                    aggression = self.memories[code].rival_profiles.get(winner, {}).get("aggression_score", 5)
                    self.memories[code].update_rival_profile(
                        winner, "aggression_score", min(10, aggression + 1)
                    )

            self._log(f"\n  ✅ SOLD! {player.name} → {TEAM_FULL_NAMES[winner]} "
                      f"for ₹{price/1e7:.2f}Cr "
                      f"({lot_result['bidding_rounds']} rounds, "
                      f"{lot_result['competing_teams']} teams competed)")
        else:
            lot_result["sold"] = False
            self._log(f"\n  ❌ UNSOLD: {player.name}")
            self.state.unsold_count += 1

        # ── Step 4: Update shared world state ──
        self.state.completed_lots.append(lot_result)
        if lot_result["sold"]:
            self.state.total_money_spent_cr += lot_result["price_cr"]
            sold_count = sum(1 for l in self.state.completed_lots if l["sold"])
            self.state.avg_price_so_far_cr = self.state.total_money_spent_cr / sold_count
            self.state.biggest_purchase_cr = max(
                self.state.biggest_purchase_cr, lot_result["price_cr"]
            )

        # Update public board
        for code, agent in self.agents.items():
            self.state.team_purses[code] = agent.squad.remaining_purse / 1e7
            self.state.team_squad_sizes[code] = agent.squad.current_size
            self.state.team_overseas_counts[code] = agent.squad.overseas_count

        self.all_lots.append(lot_result)

        # ── Step 5: Between-lot strategy reflection ──
        if self.state.lot_number % 5 == 0:
            self._run_strategy_reflection()

        return lot_result

    # ─────────────────────────────────────────
    # The Key Method: Swarm-Aware Decision
    # ─────────────────────────────────────────

    def _get_swarm_aware_decision(
        self,
        agent,
        player,
        next_bid_amount: int,
        active_teams: set,
        passed_teams: set,
    ) -> dict:
        """
        Get a bid decision that's aware of the full swarm state.

        This is the fundamental difference from the old engine.
        The agent sees:
        - Full auction room state (who has what purse, what they've bought)
        - Social signals from this lot (who's bidding aggressively, who dropped)
        - Their own persistent memory (patterns, emotions, strategy shifts)
        - Rival intelligence (what they've observed about other teams)
        """
        from ..config import TEAM_FULL_NAMES, get_bid_increment

        memory = self.memories[agent.team_code]
        valuation = agent.valuations.get(player.name, {})
        valuation_summary = valuation.get("synthesis", "No pre-auction valuation.")

        # Build the swarm-aware prompt
        system = f"""You are the auction strategist for {agent.team_name}.

IDENTITY: {agent.dna['archetype']}
PHILOSOPHY: {agent.dna['philosophy']}
BIDDING STYLE: {agent.dna['historical_preferences']['bidding_style']}

You are sitting in a live IPL auction room. You can see every other team's
paddle movements, body language, and reactions. You remember everything
that's happened so far in this auction. Your emotions and observations
influence your decisions.

CRITICAL: You must respond with ONLY a valid JSON object, no other text:
{{
  "action": "bid" or "pass",
  "reasoning": "2-3 sentences explaining your decision",
  "confidence": float 0-1,
  "max_willing_price_cr": float,
  "signal_type": "paddle_up_early" or "paddle_up_late" or "confident_nod" or "bluff_bid" or "strategic_retreat" or "dramatic_pause",
  "visible_reaction": "What other teams SEE you doing (body language, brief comment)",
  "private_thought": "What you're actually thinking (hidden from others)",
  "rival_observation": "Something you noticed about another team this round"
}}"""

        user = f"""{self.state.to_context_string()}

{memory.to_context_string()}

## Player Being Auctioned
Name: {player.name}
Role: {player.role.value} | Origin: {player.origin.value}
Base price: ₹{player.base_price/1e7:.2f}Cr

## Pre-Auction Valuation
{valuation_summary}

## Your Squad Needs
- Squad: {agent.squad.current_size}/{agent.squad.max_squad_size} | Overseas: {agent.squad.overseas_count}/{agent.squad.max_overseas}
- Purse: ₹{agent.squad.remaining_purse/1e7:.2f}Cr | Effective max: ₹{agent.squad.effective_max_bid/1e7:.2f}Cr
- Roles: {json.dumps(agent.squad.role_distribution())}
- Min slots to fill: {agent.squad.min_slots_to_fill}

## This Bid
Next bid amount: ₹{next_bid_amount/1e7:.2f}Cr
Teams that dropped out this lot: {', '.join(passed_teams) if passed_teams else 'None yet'}
Teams still active: {', '.join(active_teams) if active_teams else 'Opening bids'}

**Decision: Bid ₹{next_bid_amount/1e7:.2f}Cr for {player.name}?**"""

        try:
            result_text = agent._call_claude(system, user)
            result_text = result_text.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result = json.loads(result_text)

            # Process rival observation into memory
            if result.get("rival_observation"):
                memory.add_observation(result["rival_observation"])

            # Map signal type
            signal_map = {
                "paddle_up_early": SignalType.PADDLE_UP_EARLY,
                "paddle_up_late": SignalType.PADDLE_UP_LATE,
                "confident_nod": SignalType.CONFIDENT_NOD,
                "bluff_bid": SignalType.BLUFF_BID,
                "strategic_retreat": SignalType.STRATEGIC_RETREAT,
                "dramatic_pause": SignalType.DRAMATIC_PAUSE,
            }

            return {
                "action": result.get("action", "pass"),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", 0.5),
                "max_willing_price": int(result.get("max_willing_price_cr", 0) * 1e7),
                "signal": signal_map.get(result.get("signal_type", ""), SignalType.CALM_OBSERVATION),
                "visible_reaction": result.get("visible_reaction", ""),
                "private_thought": result.get("private_thought", ""),
            }
        except Exception as e:
            # Heuristic fallback
            return self._heuristic_swarm_decision(agent, player, next_bid_amount, active_teams)

    def _heuristic_swarm_decision(self, agent, player, next_bid_amount, active_teams) -> dict:
        """Fallback when LLM fails — still uses swarm awareness."""
        memory = self.memories[agent.team_code]
        fair_value = player.stats.get("avg_price", next_bid_amount)
        role_needed = agent.squad.role_distribution().get(player.role.value, 0) < 2

        # Emotional modifiers
        urgency_boost = 1.0 + (memory.urgency_level * 0.3)
        frustration_boost = 1.0 + (memory.frustration_level * 0.2)

        adjusted_max = fair_value * urgency_boost * frustration_boost

        if next_bid_amount <= adjusted_max and role_needed:
            return {
                "action": "bid",
                "reasoning": f"Heuristic: ₹{next_bid_amount/1e7:.2f}Cr below adjusted max ₹{adjusted_max/1e7:.2f}Cr",
                "confidence": 0.6,
                "max_willing_price": int(adjusted_max),
                "signal": SignalType.PADDLE_UP_LATE,
                "visible_reaction": f"{agent.team_name} raises paddle.",
            }
        return {
            "action": "pass",
            "reasoning": "Heuristic: price too high or role covered.",
            "confidence": 0.5,
            "signal": SignalType.STRATEGIC_RETREAT,
            "visible_reaction": f"{agent.team_name} shakes head.",
        }

    # ─────────────────────────────────────────
    # Between-Lot Strategy Reflection
    # ─────────────────────────────────────────

    def _run_strategy_reflection(self):
        """
        Every 5 lots, each agent reflects on the auction and adjusts strategy.
        This is the autoresearch "iterate and improve" pattern applied to
        auction strategy rather than ML training.
        """
        self._log(f"\n  🧠 Strategy reflection round (after {self.state.lot_number} lots)...")

        for code, agent in self.agents.items():
            memory = self.memories[code]

            # Update urgency based on slots remaining
            slots_left = agent.squad.min_slots_to_fill
            total_players_remaining = self.state.players_remaining
            if total_players_remaining > 0:
                memory.urgency_level = min(1.0, slots_left / max(1, total_players_remaining) * 3)

            # Decay frustration slightly (time heals)
            memory.frustration_level = max(0, memory.frustration_level - 0.05)

            # Auto-observations based on data
            for rival_code in self.agents:
                if rival_code == code:
                    continue
                rival_purse = self.state.team_purses.get(rival_code, 0)
                rival_squad = self.state.team_squad_sizes.get(rival_code, 0)
                if rival_purse < 20 and rival_squad < 15:
                    memory.add_observation(
                        f"{rival_code} running low on purse (₹{rival_purse:.1f}Cr) "
                        f"with only {rival_squad} players — may get desperate"
                    )

    # ─────────────────────────────────────────
    # Full Auction Run
    # ─────────────────────────────────────────

    def run_full_auction(self, player_pool: list, auction_config) -> dict:
        """Run the complete swarm-style auction."""
        from ..config import AuctionSetType, TEAM_FULL_NAMES

        self._log("\n" + "🏆" * 30)
        self._log(f"  IPL SWARM AUCTION BEGINS!")
        self._log(f"  {len(player_pool)} players | {len(self.agents)} teams")
        self._log(f"  Purse: ₹{auction_config.total_purse/1e7:.0f}Cr per team")
        self._log("🏆" * 30)

        # Sort into sets
        marquee = [p for p in player_pool if p.set_type == AuctionSetType.MARQUEE]
        capped = [p for p in player_pool if p.set_type == AuctionSetType.CAPPED]
        uncapped = [p for p in player_pool if p.set_type == AuctionSetType.UNCAPPED]

        self.state.players_remaining = len(player_pool)

        # Phase 1: Marquee
        self.state.auction_phase = "marquee"
        self._log(f"\n📌 PHASE 1: MARQUEE SET ({len(marquee)} players)")
        for player in marquee:
            self.state.players_remaining -= 1
            self.run_player_auction(player, auction_config)

        # Phase 2: Capped
        self.state.auction_phase = "capped"
        self._log(f"\n📌 PHASE 2: CAPPED PLAYERS ({len(capped)} players)")
        for player in capped:
            self.state.players_remaining -= 1
            self.run_player_auction(player, auction_config)

        # Phase 3: Uncapped
        self.state.auction_phase = "uncapped"
        self._log(f"\n📌 PHASE 3: UNCAPPED PLAYERS ({len(uncapped)} players)")
        for player in uncapped:
            self.state.players_remaining -= 1
            self.run_player_auction(player, auction_config)

        # Accelerated round
        if auction_config.has_accelerated_round:
            unsold = [lot for lot in self.all_lots if not lot["sold"]]
            if unsold:
                self.state.auction_phase = "accelerated"
                self._log(f"\n📌 ACCELERATED: {len(unsold)} unsold at reduced base")
                for lot in unsold:
                    # Find the player object
                    player_match = [p for p in player_pool if p.name == lot["player"]]
                    if player_match:
                        p = player_match[0]
                        original_base = p.base_price
                        p.base_price = max(2_000_000, p.base_price // 2)
                        self.run_player_auction(p, auction_config)
                        p.base_price = original_base

        self._log("\n" + "🏁" * 30)
        self._log("  SWARM AUCTION COMPLETE!")
        self._log("🏁" * 30)

        return self.get_results()

    def get_results(self) -> dict:
        """Comprehensive results including swarm dynamics analysis."""
        sold = [l for l in self.all_lots if l["sold"]]
        unsold = [l for l in self.all_lots if not l["sold"]]

        results = {
            "summary": {
                "total_lots": len(self.all_lots),
                "sold": len(sold),
                "unsold": len(unsold),
                "total_spend_cr": sum(l["price_cr"] for l in sold),
                "avg_price_cr": sum(l["price_cr"] for l in sold) / len(sold) if sold else 0,
                "biggest_purchase": max(sold, key=lambda l: l["price_cr"]) if sold else None,
                "most_contested": max(sold, key=lambda l: l["competing_teams"]) if sold else None,
                "longest_war": max(sold, key=lambda l: l["bidding_rounds"]) if sold else None,
            },
            "team_results": {},
            "lots": self.all_lots,
            "swarm_dynamics": {
                "total_signals": len(self.state.interaction_log),
                "bidding_wars": sum(1 for l in sold if l["competing_teams"] > 2),
                "market_atmosphere_final": self.state.get_room_atmosphere(),
            },
            "agent_memories": {},
        }

        for code, agent in self.agents.items():
            memory = self.memories[code]
            results["team_results"][code] = {
                "team_name": agent.team_name,
                "players_bought": agent.squad.current_size,
                "spent_cr": (agent.squad.total_purse - agent.squad.remaining_purse) / 1e7,
                "remaining_cr": agent.squad.remaining_purse / 1e7,
                "overseas": agent.squad.overseas_count,
                "squad": [{"name": p.name, "role": p.role.value, "origin": p.origin.value}
                         for p in agent.squad.players],
            }
            results["agent_memories"][code] = {
                "emotional_state": memory.get_emotional_summary(),
                "frustration": memory.frustration_level,
                "confidence": memory.confidence_level,
                "wars_won": memory.bidding_wars_won,
                "wars_lost": memory.bidding_wars_lost,
                "observations": memory.observed_patterns[-5:],
                "strategy_shifts": memory.strategy_adjustments,
                "rival_intel": {k: v for k, v in memory.rival_profiles.items()},
            }

        return results
