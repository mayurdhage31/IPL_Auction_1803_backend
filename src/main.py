"""
IPL Auction AI — Main Entry Point
Run the full multi-agent auction simulation.

Usage:
    python -m src.main --csv data/IPLPlayerAuctionData.csv --year 2022 --mode simulate
    python -m src.main --csv data/IPLPlayerAuctionData.csv --mode valuate --player "Virat Kohli"
    python -m src.main --csv data/IPLPlayerAuctionData.csv --mode mirofish-export
    python -m src.main --csv data/IPLPlayerAuctionData.csv --mode backtest --years 2018,2020,2022
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .data.processor import AuctionDataProcessor
from .config import MEGA_AUCTION_CONFIG, MINI_AUCTION_CONFIG, ACTIVE_TEAMS, AuctionType


def cmd_explore(args):
    """Explore the dataset — no API calls needed."""
    dp = AuctionDataProcessor(args.csv)

    print("\n📊 IPL Auction Dataset Summary")
    print("=" * 50)

    market = dp.get_market_summary()
    print(f"Total records: {market['total_players']}")
    print(f"Total spend: ₹{market['total_spend_cr']}Cr")
    print(f"Average price: ₹{market['avg_price_cr']}Cr")
    print(f"Median price: ₹{market['median_price_cr']}Cr")
    print(f"Max price: ₹{market['max_price_cr']}Cr")

    print(f"\nPlayers in database: {len(dp.players_db)}")

    print("\n📈 Price by Role:")
    for role, avg in market['by_role'].items():
        print(f"  {role}: ₹{avg}Cr avg")

    print("\n🌍 Price by Origin:")
    for origin, avg in market['by_origin'].items():
        print(f"  {origin}: ₹{avg}Cr avg")

    if args.player:
        print(f"\n🏏 Player Profile: {args.player}")
        print("-" * 40)
        ctx = dp.get_price_prediction_context(args.player)
        print(ctx)

    if args.year:
        year_market = dp.get_market_summary(args.year)
        print(f"\n📅 Year {args.year} Summary:")
        print(f"  Players: {year_market['total_players']}")
        print(f"  Total spend: ₹{year_market['total_spend_cr']}Cr")
        print(f"  Average: ₹{year_market['avg_price_cr']}Cr")

        pool = dp.generate_auction_pool(args.year)
        print(f"  Auction pool size: {len(pool)}")


def cmd_valuate(args):
    """Run Layer 1 valuation for a player or pool."""
    from .research.valuation_engine import ValuationEngine

    dp = AuctionDataProcessor(args.csv)
    ve = ValuationEngine(dp, api_key=args.api_key)

    if args.player:
        print(f"\n🔬 Valuating {args.player}...")
        result = ve.valuate_player(args.player)
        print("\n" + "=" * 50)
        print(f"Valuation for {args.player}")
        print("=" * 50)
        print(result.get("synthesis", "No synthesis generated"))
    elif args.year:
        pool = dp.generate_auction_pool(args.year)
        if args.top_n:
            # Only valuate top N by historical price
            pool.sort(key=lambda p: p.stats.get("max_price", 0), reverse=True)
            pool = pool[:args.top_n]
        print(f"\n🔬 Valuating {len(pool)} players from {args.year} auction...")
        for player in pool:
            print(f"  Processing {player.name}...")
            ve.valuate_player(player.name)
        print("\n" + ve.get_valuation_summary())


def cmd_simulate(args):
    """Run full auction simulation using the Swarm Intelligence Engine."""
    from .research.valuation_engine import ValuationEngine
    from .agents.team_agent import TeamAgent
    from .auction.swarm_engine import SwarmAuctionEngine
    from .config import MEGA_AUCTION_CONFIG, MINI_AUCTION_CONFIG, ACTIVE_TEAMS

    config = MEGA_AUCTION_CONFIG if args.auction_type == "mega" else MINI_AUCTION_CONFIG

    print(f"\n🏏 Starting IPL SWARM Auction Simulation")
    print(f"   Year: {args.year} | Type: {args.auction_type.upper()}")
    print(f"   Engine: MiroFish-style Swarm Intelligence")
    print(f"   Features: Agent memory, social signals, emergent dynamics")
    print(f"   This will make many Claude API calls.\n")

    # Layer 0: Load data
    dp = AuctionDataProcessor(args.csv)
    pool = dp.generate_auction_pool(args.year)
    if not pool:
        print(f"❌ No players found for year {args.year}")
        return

    # Layer 1: Run valuations
    print(f"🔬 Layer 1: Valuating {len(pool)} players...")
    ve = ValuationEngine(dp, api_key=args.api_key)
    valuations = {}
    for player in pool:
        valuations[player.name] = ve.valuate_player(player.name)
        print(f"  ✅ {player.name}")

    # Layer 2: Create team agents
    print(f"\n🤖 Layer 2: Spawning {len(ACTIVE_TEAMS)} team agents with swarm awareness...")
    agents = {}
    for code in ACTIVE_TEAMS:
        agents[code] = TeamAgent(
            team_code=code, config=config,
            valuations=valuations, api_key=args.api_key,
        )
        print(f"  🏟️  {agents[code].team_name} ({agents[code].dna['archetype']})")

    # Layer 3: Run swarm auction
    print(f"\n🐟 Layer 3: Swarm Auction Engine starting...")
    swarm = SwarmAuctionEngine(team_agents=agents, verbose=True)
    results = swarm.run_full_auction(player_pool=pool, auction_config=config)

    # Save results
    output_path = args.output or f"swarm_auction_{args.year}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📁 Results saved to {output_path}")

    # Print swarm dynamics summary
    dynamics = results.get("swarm_dynamics", {})
    print(f"\n🐟 Swarm Dynamics:")
    print(f"   Total social signals: {dynamics.get('total_signals', 0)}")
    print(f"   Bidding wars (3+ teams): {dynamics.get('bidding_wars', 0)}")
    print(f"   Final atmosphere: {dynamics.get('market_atmosphere_final', '?')}")

    print(f"\n🧠 Agent Emotional States:")
    for code, mem in results.get("agent_memories", {}).items():
        print(f"   {code}: {mem.get('emotional_state', '?')} "
              f"(wars: {mem.get('wars_won', 0)}W-{mem.get('wars_lost', 0)}L)")


def cmd_mirofish_export(args):
    """Generate MiroFish project files for external simulation."""
    from .research.mirofish_bridge import generate_mirofish_project

    dp = AuctionDataProcessor(args.csv)
    project = generate_mirofish_project(dp)

    output_dir = Path(args.output or "./mirofish_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save seed document
    (output_dir / "seed_document.md").write_text(project["seed_document"])

    # Save personas
    (output_dir / "personas.json").write_text(json.dumps(project["personas"], indent=2))
    (output_dir / "personas.csv").write_text(project["personas_csv"])

    # Save platform config
    (output_dir / "platform_config.json").write_text(json.dumps(project["platform_config"], indent=2))

    # Save requirements
    (output_dir / "requirements.json").write_text(json.dumps(project["requirements"], indent=2))

    print(f"\n✅ MiroFish project exported to {output_dir}/")
    print(f"   Files generated:")
    for f in output_dir.iterdir():
        print(f"     - {f.name} ({f.stat().st_size / 1024:.1f}KB)")
    print(f"\n   To use with MiroFish:")
    print(f"   1. Upload seed_document.md as seed material")
    print(f"   2. Import personas.csv in environment setup")
    print(f"   3. Configure auction platform using platform_config.json")
    print(f"   4. Set prediction goals from requirements.json")


def cmd_backtest(args):
    """Backtest against historical auctions to measure prediction accuracy."""
    dp = AuctionDataProcessor(args.csv)
    years = [int(y) for y in args.years.split(",")]

    print(f"\n📊 Backtesting across years: {years}")
    print("=" * 50)

    for year in years:
        pool = dp.generate_auction_pool(year)
        actual_results = {}
        for _, row in dp.df[dp.df["Year"] == year].iterrows():
            actual_results[row["Player"]] = {
                "team": row["Team"],
                "amount": row["Amount"],
            }

        print(f"\n📅 Year {year}: {len(pool)} players")
        print(f"   Total actual spend: ₹{sum(r['amount'] for r in actual_results.values())/1e7:.1f}Cr")

        # Here we'd compare AI predictions vs actuals
        # For now, just show the data
        top_5 = sorted(actual_results.items(), key=lambda x: x[1]["amount"], reverse=True)[:5]
        print("   Top 5 actual purchases:")
        for name, data in top_5:
            print(f"     {name} → {data['team']} for ₹{data['amount']/1e7:.2f}Cr")


def main():
    parser = argparse.ArgumentParser(description="IPL Auction AI — Multi-Agent Simulator")
    parser.add_argument("--csv", required=True, help="Path to IPL auction CSV")
    parser.add_argument("--api-key", default=None, help="Anthropic API key")

    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")

    # Explore mode
    explore_p = subparsers.add_parser("explore", help="Explore dataset")
    explore_p.add_argument("--player", default=None, help="Player name to profile")
    explore_p.add_argument("--year", type=int, default=None, help="Year to analyze")

    # Valuate mode
    val_p = subparsers.add_parser("valuate", help="Run player valuations")
    val_p.add_argument("--player", default=None, help="Single player to valuate")
    val_p.add_argument("--year", type=int, default=None, help="Valuate entire year pool")
    val_p.add_argument("--top-n", type=int, default=None, help="Only top N players")

    # Simulate mode
    sim_p = subparsers.add_parser("simulate", help="Run full auction simulation")
    sim_p.add_argument("--year", type=int, required=True, help="Auction year to simulate")
    sim_p.add_argument("--auction-type", choices=["mega", "mini"], default="mega")
    sim_p.add_argument("--output", default=None, help="Output JSON path")

    # MiroFish export
    mf_p = subparsers.add_parser("mirofish-export", help="Export MiroFish project")
    mf_p.add_argument("--output", default=None, help="Output directory")

    # Backtest
    bt_p = subparsers.add_parser("backtest", help="Backtest against historical data")
    bt_p.add_argument("--years", required=True, help="Comma-separated years")

    args = parser.parse_args()

    if args.mode == "explore":
        cmd_explore(args)
    elif args.mode == "valuate":
        cmd_valuate(args)
    elif args.mode == "simulate":
        cmd_simulate(args)
    elif args.mode == "mirofish-export":
        cmd_mirofish_export(args)
    elif args.mode == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
