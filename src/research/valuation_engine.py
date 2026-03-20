"""
Layer 1 — Research & Valuation Engine
Uses Claude API to generate fair market valuations for each player.
Implements the autoresearch pattern: predict → compare → improve.

Sub-agents:
  1. Stats Analyst — raw performance analysis
  2. Historical Price Analyst — auction price pattern recognition
  3. Age & Fitness Analyst — career trajectory modeling
  4. Market Demand Predictor — multi-team demand estimation
"""

import json
from typing import Optional
from anthropic import Anthropic

from ..data.processor import AuctionDataProcessor
from ..config import Player, TEAM_DNA, ACTIVE_TEAMS


class ValuationEngine:
    """Orchestrates multiple analyst agents to produce a fair value estimate per player."""

    def __init__(self, data_processor: AuctionDataProcessor, api_key: Optional[str] = None):
        self.dp = data_processor
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.valuations: dict[str, dict] = {}  # player_name -> valuation result

    def _call_claude(self, system: str, user: str, model: str = "claude-sonnet-4-20250514") -> str:
        """Make a Claude API call."""
        response = self.client.messages.create(
            model=model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    # ─────────────────────────────────────────
    # Sub-Agent 1: Stats Analyst
    # ─────────────────────────────────────────

    def _stats_analysis(self, player: Player) -> str:
        system = """You are a cricket analytics expert specializing in IPL player evaluation.
Analyze the player's auction history and performance trajectory.
Be specific with numbers. Output a JSON object with:
{
  "performance_score": 1-100,
  "consistency_rating": "high/medium/low",
  "key_strengths": ["..."],
  "key_risks": ["..."],
  "comparable_value_cr": float (estimated fair value in crores)
}"""
        context = self.dp.get_price_prediction_context(player.name)
        return self._call_claude(system, f"Analyze this IPL player:\n\n{context}")

    # ─────────────────────────────────────────
    # Sub-Agent 2: Historical Price Analyst
    # ─────────────────────────────────────────

    def _price_analysis(self, player: Player) -> str:
        market = self.dp.get_market_summary()
        history = player.historical_prices

        system = """You are an IPL auction price prediction specialist.
You understand auction dynamics: inflation, bidding wars, base price psychology, mega vs mini auction effects.
Given a player's price history and market context, predict their next auction price.
Output JSON:
{
  "predicted_price_cr": float,
  "confidence": "high/medium/low",
  "price_range_cr": [min, max],
  "inflation_adjusted": float,
  "key_price_drivers": ["..."]
}"""
        user = f"""Player: {player.name} ({player.role.value}, {player.origin.value})
Price History: {json.dumps(history, indent=2)}
Market Summary: {json.dumps(market, indent=2)}
Player Stats: {json.dumps(player.stats, indent=2)}"""

        return self._call_claude(system, user)

    # ─────────────────────────────────────────
    # Sub-Agent 3: Age & Fitness Analyst
    # ─────────────────────────────────────────

    def _age_fitness_analysis(self, player: Player) -> str:
        system = """You are a sports science and career longevity analyst for IPL cricket.
Evaluate the player's career trajectory, age curve, and fitness risk.
Consider: years in IPL, price trend direction, typical career arcs for their role.
Output JSON:
{
  "career_phase": "rising/peak/plateau/declining",
  "years_of_peak_left": int,
  "injury_risk": "low/medium/high",
  "age_premium_or_discount_pct": float (positive = premium, negative = discount),
  "reasoning": "..."
}"""
        user = f"""Player: {player.name}
Role: {player.role.value}, Origin: {player.origin.value}
Auction appearances: {player.stats['auction_appearances']}
First auction: {player.historical_prices[0]['year'] if player.historical_prices else 'unknown'}
Latest auction: {player.stats.get('latest_year', 'unknown')}
Price trajectory: {player.stats.get('trajectory', 'unknown')}
Peak year: {player.stats.get('peak_year', 'unknown')}
Years since peak: {player.stats.get('years_since_peak', 'unknown')}"""

        return self._call_claude(system, user)

    # ─────────────────────────────────────────
    # Sub-Agent 4: Market Demand Predictor
    # ─────────────────────────────────────────

    def _demand_analysis(self, player: Player) -> str:
        # Build team needs context
        team_contexts = []
        for code in ACTIVE_TEAMS:
            dna = TEAM_DNA[code]
            team_contexts.append(
                f"- {dna['full_name']} ({code}): {dna['archetype']}. "
                f"Prefers {', '.join(dna['historical_preferences'].get('prefer_roles', []))}. "
                f"Risk tolerance: {dna['historical_preferences'].get('risk_tolerance', 'medium')}."
            )

        system = """You are an IPL auction strategist who predicts which teams will bid for a player.
Based on team philosophies and the player profile, estimate demand.
Output JSON:
{
  "estimated_bidding_teams": int (how many teams likely to bid),
  "most_likely_buyers": ["team_code1", "team_code2", "team_code3"],
  "bidding_war_probability": float (0-1),
  "demand_premium_pct": float (how much demand inflates price above fair value),
  "reasoning": "..."
}"""
        user = f"""Player: {player.name} ({player.role.value}, {player.origin.value})
Average historical price: ₹{player.stats['avg_price']/1e7:.2f}Cr
Trajectory: {player.stats.get('trajectory', 'unknown')}

Team Profiles:
{chr(10).join(team_contexts)}"""

        return self._call_claude(system, user)

    # ─────────────────────────────────────────
    # Orchestrator: Combine All Analyses
    # ─────────────────────────────────────────

    def valuate_player(self, player_name: str) -> dict:
        """Run all 4 sub-agents and synthesize a final valuation."""
        player = self.dp.get_player(player_name)
        if not player:
            return {"error": f"Player {player_name} not found"}

        # Run all analyses
        stats_result = self._stats_analysis(player)
        price_result = self._price_analysis(player)
        age_result = self._age_fitness_analysis(player)
        demand_result = self._demand_analysis(player)

        # Synthesize
        synthesis_system = """You are the chief IPL auction analyst synthesizing reports from 4 specialist analysts.
Combine their findings into a final valuation.
Output JSON:
{
  "player": "name",
  "fair_value_cr": float,
  "confidence_band_cr": [low, high],
  "confidence_level": "high/medium/low",
  "predicted_selling_price_cr": float (includes demand premium),
  "key_factors": ["..."],
  "risk_factors": ["..."],
  "recommendation": "marquee_target / solid_buy / value_pick / avoid",
  "one_line_summary": "..."
}"""
        synthesis_user = f"""Synthesize these 4 analyst reports for {player.name} ({player.role.value}, {player.origin.value}):

## Stats Analysis:
{stats_result}

## Price Analysis:
{price_result}

## Age & Fitness Analysis:
{age_result}

## Market Demand Analysis:
{demand_result}"""

        final = self._call_claude(synthesis_system, synthesis_user)

        valuation = {
            "player": player_name,
            "raw_analyses": {
                "stats": stats_result,
                "price": price_result,
                "age_fitness": age_result,
                "demand": demand_result,
            },
            "synthesis": final,
        }

        self.valuations[player_name] = valuation
        return valuation

    def valuate_pool(self, player_names: list[str]) -> dict[str, dict]:
        """Valuate an entire auction pool."""
        results = {}
        for name in player_names:
            results[name] = self.valuate_player(name)
        return results

    def get_valuation_summary(self) -> str:
        """Generate a human-readable summary of all valuations."""
        if not self.valuations:
            return "No valuations computed yet."

        lines = ["# IPL Auction Valuation Report\n"]
        for name, val in sorted(self.valuations.items()):
            lines.append(f"## {name}")
            lines.append(val.get("synthesis", "No synthesis available"))
            lines.append("")

        return "\n".join(lines)
