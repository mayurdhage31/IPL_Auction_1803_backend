"""
MiroFish Integration Layer
Bridges our IPL auction system with MiroFish's swarm intelligence engine.

This module handles:
1. Seed Material Generation — converts auction data into MiroFish-compatible seed documents
2. Knowledge Graph Schema — defines the entity/relationship structure for GraphRAG
3. Agent Persona Generation — creates MiroFish-compatible personas for team agents
4. Auction Platform Adapter — custom simulation platform (replaces Twitter/Reddit)
5. Report Agent Configuration — post-auction analysis via MiroFish's ReportAgent

Integration pattern:
  Our system → generates seed material → MiroFish GraphRAG builds knowledge graph
  Our system → generates personas → MiroFish OASIS engine runs simulation
  MiroFish simulation → auction platform adapter → our auction engine processes bids
"""

import json
from typing import Optional

from ..config import TEAM_DNA, ACTIVE_TEAMS, TEAM_FULL_NAMES
from ..data.processor import AuctionDataProcessor


class MiroFishSeedGenerator:
    """
    Generates seed material documents for MiroFish's GraphRAG pipeline.
    Converts structured IPL auction data into rich narrative documents
    that MiroFish can parse into a knowledge graph.
    """

    def __init__(self, data_processor: AuctionDataProcessor):
        self.dp = data_processor

    def generate_full_seed(self) -> str:
        """Generate the complete seed document for MiroFish."""
        sections = [
            self._generate_header(),
            self._generate_team_profiles(),
            self._generate_player_profiles(),
            self._generate_market_dynamics(),
            self._generate_historical_patterns(),
            self._generate_auction_rules(),
        ]
        return "\n\n".join(sections)

    def _generate_header(self) -> str:
        market = self.dp.get_market_summary()
        return f"""# IPL Auction Intelligence Report

## Overview
The Indian Premier League (IPL) auction is the annual player trading event where
10 franchise teams compete to build their squads. Each team has a fixed salary
cap (purse) and must build a squad of 18-25 players with a maximum of 8 overseas
players. The auction follows an ascending-bid English auction format.

## Market Statistics (All-Time)
- Total players auctioned: {market['total_players']}
- Total spend: ₹{market['total_spend_cr']}Cr
- Average price: ₹{market['avg_price_cr']}Cr
- Median price: ₹{market['median_price_cr']}Cr
- Highest individual price: ₹{market['max_price_cr']}Cr
"""

    def _generate_team_profiles(self) -> str:
        """Generate narrative team profiles for entity extraction."""
        sections = ["## IPL Franchise Profiles\n"]

        for code in ACTIVE_TEAMS:
            dna = TEAM_DNA[code]
            team_history = self.dp.get_team_history(code)
            total_spend = team_history['Amount'].sum() / 1e7 if len(team_history) > 0 else 0
            avg_buy = team_history['Amount'].mean() / 1e7 if len(team_history) > 0 else 0
            biggest_buy = team_history.nlargest(1, 'Amount')

            section = f"""### {dna['full_name']} ({code})
**Archetype:** {dna['archetype']}
**Philosophy:** {dna['philosophy']}
**Historical Spending:** ₹{total_spend:.1f}Cr total across all auctions
**Average Purchase Price:** ₹{avg_buy:.2f}Cr
"""
            if len(biggest_buy) > 0:
                row = biggest_buy.iloc[0]
                section += f"**Biggest Ever Purchase:** {row['Player']} for ₹{row['Amount']/1e7:.2f}Cr ({int(row['Year'])})\n"

            section += f"**Preferred Roles:** {', '.join(dna['historical_preferences']['prefer_roles'])}\n"
            section += f"**Bidding Style:** {dna['historical_preferences']['bidding_style']}\n"
            section += f"**Risk Tolerance:** {dna['historical_preferences']['risk_tolerance']}\n"
            section += f"**Iconic Players:** {', '.join(dna['icon_players'])}\n"

            sections.append(section)

        return "\n".join(sections)

    def _generate_player_profiles(self) -> str:
        """Generate player profiles — top 50 most valuable for the knowledge graph."""
        sections = ["## Key Player Profiles\n"]

        # Get top 50 by average price
        if self.dp.features_df is not None:
            top = self.dp.features_df.nlargest(50, "avg_price_cr")
            for _, row in top.iterrows():
                player = self.dp.get_player(row["player"])
                if not player:
                    continue

                history_lines = []
                for h in player.historical_prices:
                    history_lines.append(f"  - {h['year']}: {h['team']} for ₹{h['amount_cr']}Cr")

                section = f"""### {player.name}
**Role:** {player.role.value} | **Origin:** {player.origin.value}
**Avg Price:** ₹{row['avg_price_cr']:.2f}Cr | **Max Price:** ₹{row['max_price_cr']:.2f}Cr
**Trajectory:** {row['trajectory']} | **Volatility:** {row['volatility']:.2f}
**Auction History:**
{chr(10).join(history_lines)}
"""
                sections.append(section)

        return "\n".join(sections)

    def _generate_market_dynamics(self) -> str:
        """Generate market-level patterns and dynamics."""
        years = sorted(self.dp.df["Year"].dropna().unique())
        year_data = []
        for yr in years:
            ms = self.dp.get_market_summary(int(yr))
            year_data.append(
                f"- **{int(yr)}**: {ms['total_players']} players sold, "
                f"avg ₹{ms['avg_price_cr']}Cr, max ₹{ms['max_price_cr']}Cr"
            )

        return f"""## Market Dynamics & Trends

### Price Evolution Over Time
{chr(10).join(year_data)}

### Key Market Patterns
- Mega auction years (2014, 2018, 2022) show higher average prices due to teams rebuilding from scratch
- Mini auction years show more selective buying with higher per-player efficiency
- Overseas all-rounders consistently command the highest premiums
- Indian fast bowlers have seen the steepest price inflation in recent years
- The "bidding war premium" averages 40-60% above fair value for marquee players
- Teams with larger remaining purses in later rounds tend to overpay due to urgency
"""

    def _generate_historical_patterns(self) -> str:
        """Generate historical auction pattern analysis."""
        # Find most re-auctioned players
        player_counts = self.dp.df.groupby("Player").size().sort_values(ascending=False)
        top_reauctioned = player_counts.head(10)

        lines = ["## Historical Auction Patterns\n", "### Most Re-Auctioned Players"]
        for name, count in top_reauctioned.items():
            player = self.dp.get_player(name)
            if player:
                prices = [h["amount_cr"] for h in player.historical_prices]
                lines.append(
                    f"- {name}: {count} auctions, price range ₹{min(prices)}-{max(prices)}Cr"
                )

        # Find biggest price jumps
        lines.append("\n### Notable Price Movements")
        for name, player in self.dp.players_db.items():
            if len(player.historical_prices) >= 2:
                prices = [h["amount"] for h in player.historical_prices]
                max_jump = max(
                    (prices[i+1] - prices[i]) / prices[i]
                    for i in range(len(prices)-1)
                    if prices[i] > 0
                )
                if max_jump > 2.0:  # >200% increase
                    lines.append(f"- {name}: {max_jump*100:.0f}% price increase between auctions")

        return "\n".join(lines[:30])  # cap length

    def _generate_auction_rules(self) -> str:
        return """## IPL Auction Rules

### Mega Auction (Full Reset)
- All players released except those retained (max 4 retentions)
- Teams get RTM (Right to Match) cards
- Starting purse: ₹90-120Cr depending on season
- Squad size: 18-25 players, max 8 overseas
- Auction proceeds in sets: Marquee → Capped Indian → Capped Overseas → Uncapped → Accelerated

### Mini Auction (Top-Up)
- Teams retain most of their squad
- Released players enter auction pool
- Remaining purse varies by team (based on releases and retention costs)
- No RTM cards

### Bidding Rules
- Ascending English auction format
- Bid increments: ₹5L (up to 1Cr), ₹10L (1-2Cr), ₹25L (2-5Cr), ₹50L (5-10Cr), ₹1Cr (10-20Cr), ₹2.5Cr (20Cr+)
- Base price set by player (₹20L to ₹2Cr)
- Teams must maintain minimum purse to fill remaining squad slots at base price
"""


class MiroFishPersonaGenerator:
    """
    Generates MiroFish-compatible agent persona files for each team.
    These are used by MiroFish's environment setup stage to instantiate
    autonomous agents in the OASIS simulation.
    """

    @staticmethod
    def generate_personas() -> list[dict]:
        """Generate persona configs for all 10 teams."""
        personas = []
        for code in ACTIVE_TEAMS:
            dna = TEAM_DNA[code]
            persona = {
                "agent_id": f"team_{code.lower()}",
                "name": f"{dna['full_name']} Management",
                "platform": "auction",  # custom platform, not twitter/reddit
                "personality": {
                    "archetype": dna["archetype"],
                    "philosophy": dna["philosophy"],
                    "risk_tolerance": dna["historical_preferences"]["risk_tolerance"],
                    "bidding_style": dna["historical_preferences"]["bidding_style"],
                    "emotional_triggers": {
                        "excited_by": dna["historical_preferences"].get("prefer_roles", []),
                        "cautious_about": "overspending early",
                        "rival_teams": [],  # populated dynamically
                    },
                },
                "memory": {
                    "iconic_signings": dna["icon_players"],
                    "preferred_roles": dna["historical_preferences"]["prefer_roles"],
                    "origin_preference": dna["historical_preferences"]["prefer_origin"],
                    "age_preference": dna["historical_preferences"]["age_preference"],
                },
                "behavioral_rules": [
                    f"Bidding style: {dna['historical_preferences']['bidding_style']}",
                    f"Risk tolerance: {dna['historical_preferences']['risk_tolerance']}",
                    "Always consider squad composition before bidding",
                    "Track rival spending and adapt strategy accordingly",
                    "Express genuine emotions during the auction process",
                ],
            }
            personas.append(persona)

        return personas

    @staticmethod
    def generate_persona_csv() -> str:
        """Generate CSV format personas for MiroFish's profile import."""
        personas = MiroFishPersonaGenerator.generate_personas()
        lines = ["agent_id,name,personality,philosophy,risk_tolerance,bidding_style"]
        for p in personas:
            lines.append(
                f"{p['agent_id']},{p['name']},"
                f"\"{p['personality']['archetype']}\","
                f"\"{p['personality']['philosophy'][:100]}\","
                f"{p['personality']['risk_tolerance']},"
                f"{p['personality']['bidding_style']}"
            )
        return "\n".join(lines)


class AuctionPlatformAdapter:
    """
    Custom simulation platform adapter for MiroFish.
    Replaces the Twitter/Reddit platforms with an auction-specific platform.

    In MiroFish's architecture, agents interact on "platforms" (social media sims).
    We define a custom "auction" platform where:
    - Actions = bid, pass, RTM, discuss_strategy, react_to_bid
    - Posts = public bid announcements
    - Comments = inter-team reactions and bluffs
    - DMs = internal team strategy discussions

    This adapter translates between MiroFish's social simulation format
    and our auction engine's bid/pass format.
    """

    PLATFORM_CONFIG = {
        "platform_name": "ipl_auction",
        "platform_type": "custom",
        "actions": [
            {
                "name": "place_bid",
                "description": "Place a bid on the current player",
                "params": {"player_name": "str", "amount": "int"},
            },
            {
                "name": "pass_player",
                "description": "Decline to bid on the current player",
                "params": {"player_name": "str", "reason": "str"},
            },
            {
                "name": "use_rtm",
                "description": "Use Right to Match card",
                "params": {"player_name": "str"},
            },
            {
                "name": "react_to_bid",
                "description": "React to another team's bid (visible to all)",
                "params": {"reaction": "str", "target_team": "str"},
            },
            {
                "name": "strategy_update",
                "description": "Internal strategy memo (visible only to this team)",
                "params": {"memo": "str"},
            },
        ],
        "simulation_params": {
            "rounds_per_player": 50,
            "time_per_round_seconds": 30,
            "max_players_per_session": 200,
        },
    }

    @staticmethod
    def get_config() -> dict:
        return AuctionPlatformAdapter.PLATFORM_CONFIG

    @staticmethod
    def format_bid_as_post(team_code: str, player_name: str, amount: int) -> dict:
        """Format a bid as a MiroFish-compatible social action."""
        return {
            "action_type": "place_bid",
            "agent_id": f"team_{team_code.lower()}",
            "content": f"{TEAM_FULL_NAMES[team_code]} bids ₹{amount/1e7:.2f}Cr for {player_name}",
            "metadata": {
                "player": player_name,
                "amount": amount,
                "amount_cr": amount / 1e7,
            },
        }

    @staticmethod
    def format_reaction(team_code: str, reaction: str, target_team: str) -> dict:
        """Format a reaction as a MiroFish social interaction."""
        return {
            "action_type": "react_to_bid",
            "agent_id": f"team_{team_code.lower()}",
            "content": reaction,
            "target": f"team_{target_team.lower()}",
        }


def generate_mirofish_project(
    data_processor: AuctionDataProcessor,
    output_dir: str = "./mirofish_project",
) -> dict:
    """
    Generate a complete MiroFish project configuration.

    This creates all the files needed to run the IPL auction simulation
    through MiroFish's pipeline:
    1. Seed document (for GraphRAG)
    2. Persona files (for agent generation)
    3. Platform config (auction platform adapter)
    4. Simulation requirements (prediction goals)

    Returns:
        Dict with file paths and contents for the MiroFish project.
    """
    seed_gen = MiroFishSeedGenerator(data_processor)
    persona_gen = MiroFishPersonaGenerator()

    project = {
        "seed_document": seed_gen.generate_full_seed(),
        "personas": persona_gen.generate_personas(),
        "personas_csv": persona_gen.generate_persona_csv(),
        "platform_config": AuctionPlatformAdapter.get_config(),
        "requirements": {
            "prediction_goal": "Simulate the IPL mega auction and predict: "
                             "(1) which team buys which player, "
                             "(2) the selling price of each player, "
                             "(3) which players go unsold, "
                             "(4) bidding war dynamics between rival teams.",
            "simulation_rounds": 40,
            "agent_count": 10,
            "platform": "auction",
        },
    }

    return project
