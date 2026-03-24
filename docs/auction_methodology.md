# IPL Auction Simulator — Methodology & Transparency

This document explains how the simulator works: where numbers come from, what assumptions are baked in, and where the system's current limits lie. It is intended to help users interpret results honestly rather than treat outputs as ground truth.

---

## 1. Franchise Spending Behaviour Assumptions

Each franchise is modelled using a hardcoded **Team DNA** profile that captures known historical tendencies. These profiles are not dynamic; they reflect observed patterns from real IPL auctions rather than real-time intelligence.

| Franchise | Archetype | Key Trait |
|-----------|-----------|-----------|
| CSK | The Wise Elder | Backs experience, patient bidding, low risk |
| MI | The Talent Factory | Invests in young Indians, avoids panic overbids |
| RCB | The Star Chaser | Aggressive on marquee names, prone to overspending |
| KKR | The Analyst | Data-driven, spin-obsessed, pre-planned targets |
| RR | The Moneyball Pioneer | Analytics-first, value-seeking, strict price ceilings |
| DC | The Rebuilder | Youth-focused, reactive bidding style |
| PBKS | The Wild Card | Most unpredictable, very high risk tolerance |
| SRH | The Bowling Machine | Disciplined, bowling-first, avoids bidding wars |
| GT | The New Power | Strategic aggression, win-now mentality |
| LSG | The Corporate | Methodical, pre-planned, process-driven |

**Risk tolerance multipliers** applied to a team's maximum willing price:

| Risk Level | Multiplier |
|------------|-----------|
| Very high | ×1.35 |
| High | ×1.20 |
| Medium-high | ×1.10 |
| Medium | ×1.00 |
| Low-medium | ×0.88 |
| Low | ×0.78 |

---

## 2. Player Price Determination

### 2.1 Base Anchor

The primary anchor for a player's value is their **historical average auction price** across all recorded IPL appearances. If a player has no auction history, the base price slab is used as the anchor.

### 2.2 Max Willing Price Formula

For each team evaluating a player in simulation, the maximum willingness to pay is computed as:

```
max_willing = avg_price
            × role_preference_mult    (1.15 if preferred role, 0.92 otherwise)
            × origin_preference_mult  (1.10 if preferred origin, 1.00 otherwise)
            × role_scarcity_mult      (1.30 if 0 of this role, down to 0.72 if well-covered)
            × purse_utilisation_mult  (1.12 if >60% purse left, 0.92 if <30% left)
            × risk_personality_mult   (see table above)
            × urgency_factor          (increases when min-squad deadline approaches)
            × gaussian_noise          (σ=0.18, capped [0.60, 1.60])
```

The result is capped at the team's effective available budget (remaining purse minus purse reserved for mandatory remaining slots).

### 2.3 Bidding Simulation

Auctions proceed in incremental rounds using official IPL bid increment slabs:

| Current Bid | Increment |
|-------------|-----------|
| < ₹1Cr | ₹50L |
| ₹1–2Cr | ₹1Cr |
| ₹2–5Cr | ₹25L |
| ₹5–10Cr | ₹50L |
| ₹10–20Cr | ₹1Cr |
| > ₹20Cr | ₹2.5Cr |

Teams bid in weighted random order; higher willingness = higher probability of being the active bidder at each step. Bidding stops when no remaining team can or will raise the current bid.

---

## 3. How Batting and Bowling Stats Are Used

Batting and bowling CSVs are merged on `(Player, Year, Team, Amount)`. Stats are used as descriptive context and **do not directly enter the price formula**. Their effect is indirect:

- **Batting average / strike rate** → used to label players as "Power Hitter" (SR > 140), "Anchor" (SR < 120), or "Middle Order". These labels appear in the UI but do not adjust the price multiplier.
- **Bowling wickets / economy / strike rate** → used to assign a `bowler_category` label (e.g., "Death Specialist", "Spinner") shown in the nomination panel.
- **Total runs / wickets** → shown in the Player Scout panel for human reference.

The price signal comes almost entirely from **historical auction prices**, not performance metrics. This is a deliberate simplification — the model trusts market pricing as the aggregate signal rather than attempting to derive value from raw stats.

---

## 4. Role Scarcity Logic

Inside each simulation run, every team maintains a live count of players acquired by role. The scarcity multiplier adjusts the team's willingness to pay based on squad composition at the moment of bidding:

| Role Count at Time of Bid | Scarcity Multiplier |
|--------------------------|---------------------|
| 0 players of this role | ×1.30 (strong need) |
| Less than half threshold | ×1.15 (moderate need) |
| Below threshold | ×1.00 (neutral) |
| At or above threshold | ×0.72 (role well-covered, low urgency) |

Role thresholds (minimum targets before scarcity drops off):

| Role | Threshold |
|------|-----------|
| Batsman | 4 |
| Bowler | 4 |
| All-Rounder | 2 |
| Wicket Keeper | 1 |

In live auction mode (not simulation), these counters are tracked in the persistent `team_states` object and updated whenever the user confirms a sale via the "Sell Player" action.

---

## 5. Purse Logic

### Auction Config (Mega Auction defaults)
- Starting purse per team: **₹90 Cr**
- Max squad size: **25 players**
- Min squad size: **18 players**
- Max overseas players: **8**

### Effective Maximum Bid

A team does not risk using its entire purse on one player if it still has minimum-squad obligations. The effective bid ceiling is:

```
effective_max_bid = remaining_purse − (min_slots_to_fill − 1) × ₹20L
```

This prevents simulation teams from going bankrupt before filling the minimum squad.

### Purse Utilisation Multiplier

Teams with more purse remaining are willing to bid more aggressively:
- > 60% purse remaining → ×1.12
- 30–60% remaining → ×1.00
- < 30% remaining → ×0.92

---

## 6. Indian vs Overseas Logic

### Overseas Cap

Each team may sign a maximum of **8 overseas players** (configurable). Once capped, the team is excluded from bidding on any further overseas player in that simulation run.

### Origin Preference

Each franchise has a stated origin preference (`Indian-heavy`, `overseas-heavy`, or `balanced`). If a player's nationality matches the team's preference, a ×1.10 multiplier is applied to willingness to pay.

### Set Assignment

Players are assigned to auction sets based on their history:
- **Marquee**: max career price ≥ ₹5Cr, or 4+ auction appearances
- **Uncapped**: Indian players with ≤ 1 auction appearance
- **Capped**: everyone else

Overseas players generally fall into Marquee or Capped rather than Uncapped, since the uncapped rule in real IPL refers specifically to uncapped Indian cricketers.

---

## 7. What Simulation Changes vs What Live Sell/Unsold Changes

This is the most important distinction to understand when using the tool.

### Monte Carlo Simulation (Forecast Mode)

- Runs 100 / 500 / 1000 independent auction scenarios.
- Each run starts with **fresh team states** — full purse, empty squad, zero overseas count.
- The current live state (what teams have bought so far) is **not carried into simulation**.
- Results show price distributions and team win probabilities as statistical forecasts.
- **Does NOT modify**: team purse, squad composition, player outcome, or any live state.
- Results appear in the Results Chart panel only.

### Live Auction Actions (Sell / Unsold)

- "Sell Player": permanently deducts the hammer price from the buying team's live purse, increments their squad count, and records the player under that team's squad.
- "Mark Unsold": records the player as unsold. No team state changes.
- Both actions update the right-side Squad Panel immediately.
- These outcomes persist across navigation — they are stored in server memory for the session.
- **Resetting the auction** (Reset button) clears all live outcomes and returns all teams to full purse and empty squads.

### Summary Table

| Action | Affects Squad Panel | Affects Purse | Affects Simulation Input |
|--------|---------------------|---------------|--------------------------|
| Run simulation | No | No | No |
| Sell Player (live) | Yes | Yes | No (simulation ignores live state) |
| Mark Unsold (live) | No | No | No |
| Reset | Clears all | Resets all | Resets all |

---

## 8. Set Navigation

The auction processes players in three sequential sets:

1. **Marquee Set** — star players auctioned first to generate early momentum
2. **Capped Set** — experienced veterans
3. **Uncapped Set** — rising talent

The UI shows position within the current set (e.g., "3 / 15 in MARQUEE SET") and overall pool position. There is no explicit "jump to next set" API endpoint — the nomination cursor advances player by player. The set boundary is crossed automatically as the cursor moves through the flat pool.

---

## 9. Current Limitations

### Data
- All pricing data is derived from historical IPL auction records. Players with only 1 auction appearance have high valuation uncertainty.
- Batting and bowling stats are pulled from aggregated CSV exports — they reflect career averages, not recent form or current fitness.
- No real-time data feeds are connected. The tool does not know about injuries, team retentions for the current auction cycle, or any in-season developments.

### Price Model
- The price formula is heuristic, not ML-trained. It does not use regression or neural models against historical auction outcomes.
- All team DNA parameters are manually coded. They encode known tendencies but will not adapt to a franchise changing strategy mid-auction.
- Gaussian noise (σ=0.18) adds per-run variance, which is intentional for realistic distribution spread but means individual runs should not be over-interpreted.

### Simulation Independence
- Single-player simulation uses **completely fresh** team states per run. It does not model the fact that Franchise A buying Player X earlier in the real auction affects what they pay for Player Y. Set and full-auction simulations do model this dependency within a run, but only for the players in the simulated pool (which is capped for performance).
- The live team state from real sell/unsold actions is not fed back into Monte Carlo runs. If 10 players have already been sold in the live auction, a simulation still starts all teams at ₹90Cr.

### Overseas Modelling
- Overseas cap is hard-coded at 8. The real IPL rule applies to playing XI (max 4 overseas per game), not the squad. The simulator enforces a squad-level cap, which is a reasonable proxy but not identical to real constraints.

### Session Persistence
- All live state is held in server memory. Restarting the backend server clears all squad data and outcomes. There is no database persistence layer.

### Accelerated Set
- The `accelerated` set type (re-nomination of unsold players) is defined in the config but is not yet implemented as an automatic flow. Unsold players can be manually re-nominated by the user advancing through the pool.
