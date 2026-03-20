"""
Layer 3 — Auction Simulation Engine
Orchestrates the full IPL auction: presents players, collects bids from all
10 team agents, resolves bidding wars, enforces rules, tracks state.

Supports both Mega and Mini auction formats.
Implements the MiroFish-inspired emergent interaction model where agents
observe each other's behavior and adapt strategies in real-time.
"""

import json
import time
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from ..config import (
    Player, PlayerRole, PlayerOrigin, AuctionConfig, BidAction,
    AuctionSetType, ACTIVE_TEAMS, TEAM_FULL_NAMES,
    get_bid_increment, MEGA_AUCTION_CONFIG, MINI_AUCTION_CONFIG,
)
from ..agents.team_agent import TeamAgent
from ..data.processor import AuctionDataProcessor


class AuctionPhase(str, Enum):
    PRE_AUCTION = "pre_auction"
    MARQUEE = "marquee"
    CAPPED = "capped"
    UNCAPPED = "uncapped"
    ACCELERATED = "accelerated"
    COMPLETED = "completed"


@dataclass
class BidRecord:
    """A single bid in a bidding round."""
    team_code: str
    amount: int
    reasoning: str
    confidence: float
    round_num: int


@dataclass
class AuctionLot:
    """Result of one player's auction."""
    player: Player
    sold: bool
    winning_team: Optional[str] = None
    winning_amount: int = 0
    bid_history: list[BidRecord] = field(default_factory=list)
    num_bidding_rounds: int = 0
    unsold_reason: Optional[str] = None


class AuctionEngine:
    """
    Main auction orchestrator.
    Presents players, runs bidding rounds, enforces rules, produces results.
    """

    def __init__(
        self,
        config: AuctionConfig,
        data_processor: AuctionDataProcessor,
        team_agents: dict[str, TeamAgent],
        verbose: bool = True,
    ):
        self.config = config
        self.dp = data_processor
        self.agents = team_agents
        self.verbose = verbose

        # Auction state
        self.phase = AuctionPhase.PRE_AUCTION
        self.lots: list[AuctionLot] = []
        self.current_lot: Optional[AuctionLot] = None
        self.auction_log: list[str] = []
        self.round_number = 0

        # Player pools by set
        self.player_pool: dict[AuctionSetType, list[Player]] = {
            AuctionSetType.MARQUEE: [],
            AuctionSetType.CAPPED: [],
            AuctionSetType.UNCAPPED: [],
        }

    def _log(self, msg: str):
        self.auction_log.append(msg)
        if self.verbose:
            print(msg)

    # ─────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────

    def load_auction_pool(self, players: list[Player]):
        """Distribute players into auction sets."""
        for player in players:
            set_type = player.set_type
            if set_type == AuctionSetType.MARQUEE:
                self.player_pool[AuctionSetType.MARQUEE].append(player)
            elif set_type == AuctionSetType.UNCAPPED:
                self.player_pool[AuctionSetType.UNCAPPED].append(player)
            else:
                self.player_pool[AuctionSetType.CAPPED].append(player)

        self._log(f"📋 Auction pool loaded: "
                   f"{len(self.player_pool[AuctionSetType.MARQUEE])} marquee, "
                   f"{len(self.player_pool[AuctionSetType.CAPPED])} capped, "
                   f"{len(self.player_pool[AuctionSetType.UNCAPPED])} uncapped. "
                   f"Total: {sum(len(v) for v in self.player_pool.values())} players.")

    # ─────────────────────────────────────────
    # Core Bidding Loop
    # ─────────────────────────────────────────

    def _run_bidding_for_player(self, player: Player) -> AuctionLot:
        """Run a complete bidding war for a single player."""
        self.round_number += 1
        lot = AuctionLot(player=player, sold=False)
        current_bid = player.base_price
        highest_bidder: Optional[str] = None
        bid_round = 0

        self._log(f"\n{'='*60}")
        self._log(f"🏏 NOW SELLING: {player.name} ({player.role.value}, {player.origin.value})")
        self._log(f"   Base Price: ₹{player.base_price/1e7:.2f}Cr")
        self._log(f"{'='*60}")

        # Generate auction context visible to all agents
        auction_context = self._generate_auction_context()

        # Bidding rounds
        max_rounds = 50  # safety limit
        while bid_round < max_rounds:
            bid_round += 1
            bids_this_round = []

            # Collect decisions from all eligible teams
            for code, agent in self.agents.items():
                # Skip the current highest bidder
                if code == highest_bidder:
                    continue

                # Get decision
                active_bidders = [highest_bidder] if highest_bidder else []
                decision = agent.decide_bid(
                    player=player,
                    current_bid=current_bid,
                    active_bidders=active_bidders,
                    auction_context=auction_context,
                )

                if decision["action"] == BidAction.BID.value:
                    next_bid = current_bid + get_bid_increment(current_bid)
                    # Verify the team can actually afford this
                    if next_bid <= agent.squad.effective_max_bid:
                        bids_this_round.append({
                            "team_code": code,
                            "amount": next_bid,
                            "reasoning": decision["reasoning"],
                            "confidence": decision["confidence"],
                            "max_willing": decision["max_willing_price"],
                        })

            if not bids_this_round:
                # No one is bidding — sold to current highest or unsold
                break

            # Resolve: highest confidence bidder wins this round
            # (In real auctions, the auctioneer picks; here we simulate)
            winning_bid = max(bids_this_round, key=lambda b: b["confidence"])
            highest_bidder = winning_bid["team_code"]
            current_bid = winning_bid["amount"]

            bid_record = BidRecord(
                team_code=highest_bidder,
                amount=current_bid,
                reasoning=winning_bid["reasoning"],
                confidence=winning_bid["confidence"],
                round_num=bid_round,
            )
            lot.bid_history.append(bid_record)

            self._log(f"  Round {bid_round}: {TEAM_FULL_NAMES.get(highest_bidder, highest_bidder)} "
                       f"bids ₹{current_bid/1e7:.2f}Cr — {winning_bid['reasoning'][:80]}")

            # Check if price exceeded everyone's max willing price
            remaining_willing = [
                b for b in bids_this_round
                if b["team_code"] != highest_bidder and b["max_willing"] > current_bid
            ]
            if not remaining_willing and bid_round > 1:
                # Only the highest bidder is left willing
                break

        # Resolve lot
        lot.num_bidding_rounds = bid_round

        if highest_bidder:
            lot.sold = True
            lot.winning_team = highest_bidder
            lot.winning_amount = current_bid

            # Update team states
            self.agents[highest_bidder].register_purchase(player, current_bid)
            for code, agent in self.agents.items():
                if code != highest_bidder:
                    agent.register_loss(player, highest_bidder, current_bid)

            self._log(f"\n  🔨 SOLD! {player.name} → {TEAM_FULL_NAMES[highest_bidder]} "
                       f"for ₹{current_bid/1e7:.2f}Cr ({bid_round} rounds)")
        else:
            lot.sold = False
            lot.unsold_reason = "No bids received"
            self._log(f"\n  ❌ UNSOLD: {player.name} — no bids at base price ₹{player.base_price/1e7:.2f}Cr")

        self.lots.append(lot)
        return lot

    # ─────────────────────────────────────────
    # Auction Phases
    # ─────────────────────────────────────────

    def run_full_auction(self):
        """Execute the complete auction across all phases."""
        self._log("\n" + "🏆" * 30)
        self._log(f"  IPL {self.config.auction_type.value.upper()} AUCTION BEGINS!")
        self._log(f"  Purse per team: ₹{self.config.total_purse/1e7:.2f}Cr")
        self._log(f"  Teams: {len(self.agents)}")
        self._log("🏆" * 30 + "\n")

        # Phase 1: Marquee players
        self.phase = AuctionPhase.MARQUEE
        self._log("\n📌 PHASE 1: MARQUEE SET")
        for player in self.player_pool[AuctionSetType.MARQUEE]:
            self._run_bidding_for_player(player)
            self._print_purse_status()

        # Phase 2: Capped players
        self.phase = AuctionPhase.CAPPED
        self._log("\n📌 PHASE 2: CAPPED PLAYERS")
        for player in self.player_pool[AuctionSetType.CAPPED]:
            self._run_bidding_for_player(player)

        # Phase 3: Uncapped players
        self.phase = AuctionPhase.UNCAPPED
        self._log("\n📌 PHASE 3: UNCAPPED PLAYERS")
        for player in self.player_pool[AuctionSetType.UNCAPPED]:
            self._run_bidding_for_player(player)

        # Accelerated round for unsold players
        if self.config.has_accelerated_round:
            self._run_accelerated_round()

        self.phase = AuctionPhase.COMPLETED
        self._log("\n" + "🏁" * 30)
        self._log("  AUCTION COMPLETE!")
        self._log("🏁" * 30)

    def _run_accelerated_round(self):
        """Re-auction unsold players at reduced base prices."""
        unsold = [lot.player for lot in self.lots if not lot.sold]
        if not unsold:
            self._log("\n📌 ACCELERATED ROUND: No unsold players!")
            return

        self.phase = AuctionPhase.ACCELERATED
        self._log(f"\n📌 ACCELERATED ROUND: {len(unsold)} unsold players at reduced base prices")

        for player in unsold:
            # Reduce base price by 50%
            original_base = player.base_price
            player.base_price = max(2_000_000, player.base_price // 2)
            self._log(f"  Base price reduced: ₹{original_base/1e7:.2f}Cr → ₹{player.base_price/1e7:.2f}Cr")
            self._run_bidding_for_player(player)
            player.base_price = original_base  # restore

    # ─────────────────────────────────────────
    # Context & Reporting
    # ─────────────────────────────────────────

    def _generate_auction_context(self) -> str:
        """Generate context visible to all agents about auction state."""
        lines = [
            f"Auction Phase: {self.phase.value}",
            f"Players sold so far: {sum(1 for l in self.lots if l.sold)}",
            f"Players unsold: {sum(1 for l in self.lots if not l.sold)}",
            "",
            "Team Purse Status:",
        ]
        for code in sorted(self.agents.keys()):
            agent = self.agents[code]
            squad = agent.squad
            lines.append(
                f"  {code}: ₹{squad.remaining_purse/1e7:.1f}Cr remaining, "
                f"{squad.current_size}/{squad.max_squad_size} players, "
                f"{squad.overseas_count}/{squad.max_overseas} overseas"
            )

        # Recent big buys (social pressure / market setting)
        recent_sales = [l for l in self.lots[-5:] if l.sold]
        if recent_sales:
            lines.append("\nRecent Sales:")
            for lot in recent_sales:
                lines.append(
                    f"  {lot.player.name} → {TEAM_FULL_NAMES.get(lot.winning_team, lot.winning_team)} "
                    f"for ₹{lot.winning_amount/1e7:.2f}Cr"
                )

        return "\n".join(lines)

    def _print_purse_status(self):
        """Print current purse status for all teams."""
        self._log("\n  💰 Purse Status:")
        for code in sorted(self.agents.keys()):
            agent = self.agents[code]
            self._log(f"    {code}: ₹{agent.squad.remaining_purse/1e7:.1f}Cr "
                       f"({agent.squad.current_size} players)")

    def get_results_summary(self) -> dict:
        """Generate comprehensive auction results."""
        results = {
            "config": {
                "type": self.config.auction_type.value,
                "purse": self.config.total_purse / 1e7,
            },
            "totals": {
                "players_sold": sum(1 for l in self.lots if l.sold),
                "players_unsold": sum(1 for l in self.lots if not l.sold),
                "total_spend_cr": sum(l.winning_amount for l in self.lots if l.sold) / 1e7,
                "avg_price_cr": 0,
                "most_expensive": None,
                "biggest_bidding_war": None,
            },
            "team_results": {},
            "lot_details": [],
        }

        sold_lots = [l for l in self.lots if l.sold]
        if sold_lots:
            results["totals"]["avg_price_cr"] = results["totals"]["total_spend_cr"] / len(sold_lots)
            most_exp = max(sold_lots, key=lambda l: l.winning_amount)
            results["totals"]["most_expensive"] = {
                "player": most_exp.player.name,
                "amount_cr": most_exp.winning_amount / 1e7,
                "team": most_exp.winning_team,
            }
            biggest_war = max(sold_lots, key=lambda l: l.num_bidding_rounds)
            results["totals"]["biggest_bidding_war"] = {
                "player": biggest_war.player.name,
                "rounds": biggest_war.num_bidding_rounds,
                "final_price_cr": biggest_war.winning_amount / 1e7,
            }

        for code, agent in self.agents.items():
            results["team_results"][code] = {
                "team_name": agent.team_name,
                "players_bought": agent.squad.current_size,
                "total_spent_cr": (agent.squad.total_purse - agent.squad.remaining_purse) / 1e7,
                "remaining_purse_cr": agent.squad.remaining_purse / 1e7,
                "overseas_count": agent.squad.overseas_count,
                "squad": [
                    {
                        "name": p.name,
                        "role": p.role.value,
                        "origin": p.origin.value,
                    }
                    for p in agent.squad.players
                ],
            }

        for lot in self.lots:
            results["lot_details"].append({
                "player": lot.player.name,
                "role": lot.player.role.value,
                "origin": lot.player.origin.value,
                "base_price_cr": lot.player.base_price / 1e7,
                "sold": lot.sold,
                "winning_team": lot.winning_team,
                "winning_amount_cr": lot.winning_amount / 1e7 if lot.sold else 0,
                "bidding_rounds": lot.num_bidding_rounds,
                "bid_history": [
                    {"team": b.team_code, "amount_cr": b.amount / 1e7, "round": b.round_num}
                    for b in lot.bid_history
                ],
            })

        return results

    def get_results_text(self) -> str:
        """Human-readable results report."""
        results = self.get_results_summary()
        lines = [
            "# 🏏 IPL Auction Results\n",
            f"**Format:** {results['config']['type'].upper()} Auction",
            f"**Starting Purse:** ₹{results['config']['purse']:.0f}Cr per team\n",
            f"## Summary",
            f"- Players Sold: {results['totals']['players_sold']}",
            f"- Players Unsold: {results['totals']['players_unsold']}",
            f"- Total Spend: ₹{results['totals']['total_spend_cr']:.2f}Cr",
            f"- Average Price: ₹{results['totals']['avg_price_cr']:.2f}Cr",
        ]

        if results["totals"]["most_expensive"]:
            me = results["totals"]["most_expensive"]
            lines.append(f"- Most Expensive: {me['player']} → {me['team']} for ₹{me['amount_cr']:.2f}Cr")

        if results["totals"]["biggest_bidding_war"]:
            bw = results["totals"]["biggest_bidding_war"]
            lines.append(f"- Biggest Bidding War: {bw['player']} ({bw['rounds']} rounds, ₹{bw['final_price_cr']:.2f}Cr)")

        lines.append("\n## Team Squads\n")
        for code in sorted(results["team_results"].keys()):
            tr = results["team_results"][code]
            lines.append(f"### {tr['team_name']} ({code})")
            lines.append(f"Spent: ₹{tr['total_spent_cr']:.2f}Cr | "
                          f"Remaining: ₹{tr['remaining_purse_cr']:.2f}Cr | "
                          f"Players: {tr['players_bought']} | "
                          f"Overseas: {tr['overseas_count']}")
            for p in tr["squad"]:
                lines.append(f"  - {p['name']} ({p['role']}, {p['origin']})")
            lines.append("")

        return "\n".join(lines)


# ─────────────────────────────────────────────
# Convenience: Full Simulation Runner
# ─────────────────────────────────────────────

def run_simulation(
    csv_path: str,
    auction_year: int,
    auction_type: str = "mega",
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    End-to-end simulation runner.

    Args:
        csv_path: Path to the historical auction CSV
        auction_year: Which year's auction pool to simulate
        auction_type: "mega" or "mini"
        api_key: Anthropic API key (or uses env var)
        verbose: Print auction progress

    Returns:
        Full auction results dict
    """
    from ..research.valuation_engine import ValuationEngine

    # 1. Load and process data
    dp = AuctionDataProcessor(csv_path)
    config = MEGA_AUCTION_CONFIG if auction_type == "mega" else MINI_AUCTION_CONFIG

    # 2. Get auction pool for the year
    pool = dp.generate_auction_pool(auction_year)
    if not pool:
        raise ValueError(f"No players found for year {auction_year}")

    # 3. Run valuations (Layer 1)
    if verbose:
        print(f"\n🔬 Running valuations for {len(pool)} players...")
    ve = ValuationEngine(dp, api_key=api_key)
    valuations = {}
    for player in pool:
        valuations[player.name] = ve.valuate_player(player.name)
        if verbose:
            print(f"  ✅ Valued: {player.name}")

    # 4. Create team agents (Layer 2)
    agents = {}
    for code in ACTIVE_TEAMS:
        agents[code] = TeamAgent(
            team_code=code,
            config=config,
            valuations=valuations,
            api_key=api_key,
        )

    # 5. Run auction (Layer 3)
    engine = AuctionEngine(
        config=config,
        data_processor=dp,
        team_agents=agents,
        verbose=verbose,
    )
    engine.load_auction_pool(pool)
    engine.run_full_auction()

    return engine.get_results_summary()
