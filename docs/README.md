# IPL Auction AI — Multi-Agent Auction Simulator

An AI-powered IPL auction system where autonomous agents represent each franchise,
research player valuations, and compete in realistic auction simulations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AUCTION ENGINE (Layer 3)                  │
│  Orchestrates bidding rounds, enforces IPL rules,           │
│  manages state across all 10 teams                          │
├─────────────────────────────────────────────────────────────┤
│              TEAM AGENTS × 10 (Layer 2)                     │
│  Each team has:                                             │
│  ├── Squad Needs Analyst                                    │
│  ├── Budget Strategist                                      │
│  ├── Team DNA Agent (franchise philosophy)                  │
│  ├── Bid Ceiling Calculator                                 │
│  └── Auction Strategist (final decision maker)              │
├─────────────────────────────────────────────────────────────┤
│         RESEARCH & VALUATION ENGINE (Layer 1)               │
│  Pre-auction analysis per player:                           │
│  ├── Stats Analyst (T20/IPL performance)                    │
│  ├── Historical Price Analyst (auction price trends)        │
│  ├── Age & Fitness Analyst (career trajectory)              │
│  └── Market Demand Predictor (competing bidders)            │
└─────────────────────────────────────────────────────────────┘
```

## Inspired By
- [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) — Multi-agent orchestration pattern
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — Autonomous iteration loop

## Tech Stack
- **Python 3.11+**
- **Claude API (Anthropic)** — All agent LLM calls
- **LangGraph** — Agent orchestration & state management
- **Pandas** — Data processing
- **FastAPI + React** — Web dashboard (Phase 3)

## Quick Start
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key
python -m src.main --mode simulate --auction-type mega
```
