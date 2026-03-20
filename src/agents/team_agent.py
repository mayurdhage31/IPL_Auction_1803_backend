"""
Layer 2 — Team Agent System
Each IPL franchise is represented by a Team Agent with internal sub-analysts
(ai-hedge-fund pattern) and a distinct personality (MiroFish persona).

Each team agent internally runs:
  1. Squad Needs Analyst — what roles/slots need filling
  2. Budget Strategist — optimal purse allocation
  3. Team DNA Agent — franchise-specific philosophy filter
  4. Hype Analyst — estimates rival interest level
  5. Auction Strategist — synthesizes all inputs → final bid/pass decision
"""

import json
from typing import Optional
from anthropic import Anthropic

from ..config import (
    Player, PlayerRole, PlayerOrigin, TeamSquad, BidAction,
    AuctionConfig, TEAM_DNA, ACTIVE_TEAMS, TEAM_FULL_NAMES,
    get_bid_increment,
)


class TeamAgent:
    """
    Autonomous agent representing an IPL franchise in the auction.
    Makes bid/pass/RTM decisions using internal analyst pipeline.
    """

    def __init__(
        self,
        team_code: str,
        config: AuctionConfig,
        valuations: dict[str, dict],
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.team_code = team_code
        self.team_name = TEAM_FULL_NAMES[team_code]
        self.dna = TEAM_DNA[team_code]
        self.config = config
        self.valuations = valuations
        self.model = model
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

        # Initialize squad state
        self.squad = TeamSquad(
            remaining_purse=config.total_purse,
            total_purse=config.total_purse,
            max_squad_size=config.max_squad_size,
            min_squad_size=config.min_squad_size,
            max_overseas=config.max_overseas,
            rtm_cards=config.rtm_cards_per_team,
        )

        # Internal state for auction memory
        self.bid_history: list[dict] = []
        self.target_list: list[str] = []
        self.avoided_players: list[str] = []
        self.auction_log: list[str] = []

    def _call_claude(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    # ─────────────────────────────────────────
    # Internal Analyst 1: Squad Needs
    # ─────────────────────────────────────────

    def _analyze_squad_needs(self, player: Player) -> str:
        current_roles = self.squad.role_distribution()
        return f"""Squad Analysis for {self.team_name}:
- Current squad size: {self.squad.current_size}/{self.squad.max_squad_size}
- Slots remaining: {self.squad.slots_remaining}
- Overseas count: {self.squad.overseas_count}/{self.squad.max_overseas}
- Overseas slots left: {self.squad.overseas_slots_remaining}
- Role distribution: {json.dumps(current_roles)}
- Min slots to fill: {self.squad.min_slots_to_fill}
- Can buy overseas: {self.squad.can_buy_overseas()}

Player being auctioned: {player.name} ({player.role.value}, {player.origin.value})

Need assessment:
- Role needed: {'YES - gap in squad' if current_roles.get(player.role.value, 0) < 3 else 'Covered but depth useful'}
- Origin constraint: {'BLOCKED - overseas full' if player.origin == PlayerOrigin.OVERSEAS and not self.squad.can_buy_overseas() else 'Available'}
"""

    # ─────────────────────────────────────────
    # Internal Analyst 2: Budget Strategy
    # ─────────────────────────────────────────

    def _analyze_budget(self, player: Player, current_bid: int) -> str:
        effective_max = self.squad.effective_max_bid
        pct_of_purse = (current_bid / self.squad.remaining_purse * 100) if self.squad.remaining_purse > 0 else 999

        return f"""Budget Analysis for {self.team_name}:
- Remaining purse: ₹{self.squad.remaining_purse/1e7:.2f}Cr
- Effective max bid (reserving for min squad): ₹{effective_max/1e7:.2f}Cr
- Current bid for {player.name}: ₹{current_bid/1e7:.2f}Cr
- This bid as % of remaining purse: {pct_of_purse:.1f}%
- Slots still to fill: {self.squad.min_slots_to_fill}

Budget recommendation:
- {'DANGER - bid exceeds effective max!' if current_bid > effective_max else 'Within budget'}
- {'HIGH ALLOCATION - over 15% of purse on one player' if pct_of_purse > 15 else 'Reasonable allocation'}
"""

    # ─────────────────────────────────────────
    # Internal Analyst 3: Team DNA Filter
    # ─────────────────────────────────────────

    def _apply_team_dna(self, player: Player) -> str:
        dna = self.dna
        prefs = dna["historical_preferences"]

        role_match = player.role.value in prefs.get("prefer_roles", [])
        origin_pref = prefs.get("prefer_origin", "balanced")

        return f"""Team DNA Assessment — {self.team_name} ({dna['archetype']}):
Philosophy: {dna['philosophy']}

Player fit:
- Role preference match: {'YES ✓' if role_match else 'Not preferred role'}
- Origin fit: {origin_pref} preference vs {player.origin.value} player
- Risk tolerance: {prefs.get('risk_tolerance', 'medium')} vs player volatility {player.stats.get('volatility', 0):.2f}
- Bidding style: {prefs.get('bidding_style', 'balanced')}
- Age preference: {prefs.get('age_preference', 'mixed')}

Iconic similar signings: {', '.join(dna.get('icon_players', []))}
"""

    # ─────────────────────────────────────────
    # Internal Analyst 4: Hype / Rival Interest
    # ─────────────────────────────────────────

    def _analyze_rival_interest(self, player: Player, active_bidders: list[str]) -> str:
        valuation = self.valuations.get(player.name, {})
        demand_info = valuation.get("raw_analyses", {}).get("demand", "No demand data available")

        return f"""Rival Interest Analysis for {player.name}:
Currently active bidders: {', '.join(active_bidders) if active_bidders else 'None yet'}
Number of teams still interested: {len(active_bidders)}

Demand intelligence:
{demand_info}

Strategic implications:
- {'Bidding war likely - multiple strong bidders' if len(active_bidders) >= 3 else 'Limited competition' if len(active_bidders) <= 1 else 'Moderate competition'}
"""

    # ─────────────────────────────────────────
    # Auction Strategist: Final Decision
    # ─────────────────────────────────────────

    def decide_bid(
        self,
        player: Player,
        current_bid: int,
        active_bidders: list[str],
        auction_context: str = "",
    ) -> dict:
        """
        Main decision method — should this team bid on this player at this price?

        Returns:
            {
                "action": "bid" | "pass" | "rtm",
                "max_willing_price": int,
                "reasoning": str,
                "confidence": float (0-1)
            }
        """
        # Hard constraints — auto-pass
        if player.origin == PlayerOrigin.OVERSEAS and not self.squad.can_buy_overseas():
            return {
                "action": BidAction.PASS.value,
                "max_willing_price": 0,
                "reasoning": "Overseas slots full — cannot bid.",
                "confidence": 1.0,
            }

        if current_bid > self.squad.effective_max_bid:
            return {
                "action": BidAction.PASS.value,
                "max_willing_price": 0,
                "reasoning": f"Price ₹{current_bid/1e7:.2f}Cr exceeds effective budget ₹{self.squad.effective_max_bid/1e7:.2f}Cr.",
                "confidence": 1.0,
            }

        if self.squad.slots_remaining <= 0:
            return {
                "action": BidAction.PASS.value,
                "max_willing_price": 0,
                "reasoning": "Squad full.",
                "confidence": 1.0,
            }

        # Gather all analyst inputs
        squad_analysis = self._analyze_squad_needs(player)
        budget_analysis = self._analyze_budget(player, current_bid)
        dna_analysis = self._apply_team_dna(player)
        rival_analysis = self._analyze_rival_interest(player, active_bidders)

        # Get pre-computed valuation
        valuation = self.valuations.get(player.name, {})
        valuation_summary = valuation.get("synthesis", "No pre-auction valuation available.")

        # LLM-powered final decision
        system = f"""You are the auction strategist for {self.team_name} ({self.dna['archetype']}).
Your personality: {self.dna['philosophy']}
Your bidding style: {self.dna['historical_preferences'].get('bidding_style', 'balanced')}

Make a bid/pass decision. You MUST respond with valid JSON only:
{{
  "action": "bid" or "pass",
  "max_willing_price_cr": float,
  "reasoning": "2-3 sentence explanation",
  "confidence": float between 0 and 1,
  "urgency": "must_have" or "nice_to_have" or "luxury" or "avoid"
}}"""

        user = f"""## Auction Decision Required

**Player on the block:** {player.name} ({player.role.value}, {player.origin.value})
**Current bid:** ₹{current_bid/1e7:.2f}Cr
**Next bid would be:** ₹{(current_bid + get_bid_increment(current_bid))/1e7:.2f}Cr

{auction_context}

### Analyst Reports:

{squad_analysis}

{budget_analysis}

{dna_analysis}

{rival_analysis}

### Pre-Auction Valuation:
{valuation_summary}

### Recent Auction History (this session):
Bids won: {len([b for b in self.bid_history if b.get('won')])}
Total spent: ₹{sum(b.get('amount', 0) for b in self.bid_history if b.get('won'))/1e7:.2f}Cr

**Decision: Should {self.team_name} bid ₹{(current_bid + get_bid_increment(current_bid))/1e7:.2f}Cr for {player.name}?**"""

        try:
            result_text = self._call_claude(system, user)
            # Extract JSON from response
            result_text = result_text.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result = json.loads(result_text)

            return {
                "action": result.get("action", BidAction.PASS.value),
                "max_willing_price": int(result.get("max_willing_price_cr", 0) * 1e7),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", 0.5),
                "urgency": result.get("urgency", "nice_to_have"),
            }
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: simple heuristic decision
            return self._heuristic_decision(player, current_bid)

    def _heuristic_decision(self, player: Player, current_bid: int) -> dict:
        """Fallback heuristic when LLM fails."""
        fair_value = player.stats.get("avg_price", current_bid)
        role_needed = self.squad.role_distribution().get(player.role.value, 0) < 2

        if current_bid <= fair_value * 0.8 and role_needed:
            return {
                "action": BidAction.BID.value,
                "max_willing_price": int(fair_value * 1.1),
                "reasoning": f"Heuristic: price below fair value and role needed.",
                "confidence": 0.6,
            }
        return {
            "action": BidAction.PASS.value,
            "max_willing_price": 0,
            "reasoning": "Heuristic: price too high or role not needed.",
            "confidence": 0.5,
        }

    # ─────────────────────────────────────────
    # Squad Management
    # ─────────────────────────────────────────

    def register_purchase(self, player: Player, amount: int):
        """Record that this team won a player."""
        self.squad.players.append(player)
        self.squad.remaining_purse -= amount
        self.bid_history.append({
            "player": player.name,
            "amount": amount,
            "amount_cr": amount / 1e7,
            "role": player.role.value,
            "origin": player.origin.value,
            "won": True,
        })
        self.auction_log.append(
            f"✅ WON {player.name} ({player.role.value}) for ₹{amount/1e7:.2f}Cr. "
            f"Purse: ₹{self.squad.remaining_purse/1e7:.2f}Cr remaining."
        )

    def register_loss(self, player: Player, winning_team: str, amount: int):
        """Record that this team lost a player to another team."""
        self.bid_history.append({
            "player": player.name,
            "amount": amount,
            "won": False,
            "lost_to": winning_team,
        })
        self.auction_log.append(
            f"❌ Lost {player.name} to {winning_team} at ₹{amount/1e7:.2f}Cr"
        )

    def get_squad_summary(self) -> str:
        """Human-readable squad summary."""
        lines = [
            f"## {self.team_name} ({self.team_code})",
            f"Purse remaining: ₹{self.squad.remaining_purse/1e7:.2f}Cr / ₹{self.squad.total_purse/1e7:.2f}Cr",
            f"Squad: {self.squad.current_size}/{self.squad.max_squad_size} "
            f"(Overseas: {self.squad.overseas_count}/{self.squad.max_overseas})",
            f"RTM cards: {self.squad.rtm_cards}",
            "",
            "### Players:",
        ]
        for p in self.squad.players:
            price = next(
                (b["amount_cr"] for b in self.bid_history if b["player"] == p.name and b.get("won")),
                0,
            )
            lines.append(f"  - {p.name} ({p.role.value}, {p.origin.value}) — ₹{price:.2f}Cr")

        return "\n".join(lines)

    def generate_persona_prompt(self) -> str:
        """Generate a MiroFish-compatible persona description for this team agent."""
        dna = self.dna
        return f"""You are the team management of {self.team_name}.

IDENTITY: {dna['archetype']}
PHILOSOPHY: {dna['philosophy']}

BEHAVIORAL RULES:
- Bidding style: {dna['historical_preferences']['bidding_style']}
- Risk tolerance: {dna['historical_preferences']['risk_tolerance']}
- Preferred roles: {', '.join(dna['historical_preferences']['prefer_roles'])}
- Age preference: {dna['historical_preferences']['age_preference']}
- Origin preference: {dna['historical_preferences']['prefer_origin']}

ICONIC SIGNINGS YOU ADMIRE: {', '.join(dna['icon_players'])}

You think and act like {self.team_name}'s real management. Your decisions should
reflect the franchise's actual historical behavior and strategic tendencies.
You have emotions — you get excited about players who fit your vision,
frustrated when rivals outbid you, and anxious when your target list thins out.
"""
