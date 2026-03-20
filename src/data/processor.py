"""
Data Processing Layer — handles batting + bowling CSVs independently and merges them.
Produces a unified player database with combined stats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from ..config import (
    Player, PlayerRole, PlayerOrigin, AuctionSetType,
    TEAM_NAME_MAP, ACTIVE_TEAMS
)


class AuctionDataProcessor:
    """
    Loads and merges batting + bowling CSVs into a unified player database.
    Handles column-name differences, deduplication, and feature computation.
    """

    def __init__(self, batting_csv: str, bowling_csv: Optional[str] = None):
        self.batting_csv = batting_csv
        self.bowling_csv = bowling_csv
        self.players_db: dict[str, Player] = {}
        self.features_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self._process()

    # ─────────────────────────────────────────
    # Ingestion
    # ─────────────────────────────────────────

    def _load_batting(self) -> pd.DataFrame:
        df = pd.read_csv(self.batting_csv)
        # Normalize player name column
        if "batter_name" in df.columns:
            df = df.rename(columns={"batter_name": "Player"})
        return df

    def _load_bowling(self) -> Optional[pd.DataFrame]:
        if not self.bowling_csv or not Path(self.bowling_csv).exists():
            return None
        df = pd.read_csv(self.bowling_csv)
        # Drop unnamed index column if present
        unnamed = [c for c in df.columns if c.startswith("Unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)
        if "bowler_name" in df.columns:
            df = df.rename(columns={"bowler_name": "Player"})
        return df

    def _merge_sources(self, bat_df: pd.DataFrame, bowl_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge batting and bowling data on (Player, Year, Team).
        Players that appear in both get combined stats.
        Players appearing only in batting get NaN bowling columns and vice versa.
        """
        bat_df = bat_df.copy()
        bat_df["_source"] = "batting"

        if bowl_df is None:
            return bat_df

        bowl_df = bowl_df.copy()
        bowl_df["_source"] = "bowling"

        merge_keys = ["Player", "Year", "Team", "Amount", "Player Origin", "Role"]
        available_merge_keys = [k for k in merge_keys if k in bat_df.columns and k in bowl_df.columns]

        merged = pd.merge(
            bat_df, bowl_df,
            on=available_merge_keys,
            how="outer",
            suffixes=("_bat", "_bowl"),
        )

        # Resolve _source column
        merged["_source"] = merged.apply(
            lambda r: "both" if (pd.notna(r.get("_source_bat")) and pd.notna(r.get("_source_bowl")))
            else (r.get("_source_bat") or r.get("_source_bowl") or "unknown"),
            axis=1,
        )
        merged = merged.drop(columns=["_source_bat", "_source_bowl"], errors="ignore")
        return merged

    # ─────────────────────────────────────────
    # Main Processing Pipeline
    # ─────────────────────────────────────────

    def _process(self):
        bat_df = self._load_batting()
        bowl_df = self._load_bowling()

        df = self._merge_sources(bat_df, bowl_df)

        # Sanitise
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
        df = df[df["Year"] > 0]
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
        df = df[df["Amount"] > 0]
        df["Amount_Cr"] = df["Amount"] / 1e7
        df["Team_Code"] = df["Team"].map(TEAM_NAME_MAP)

        # Normalise Role values so they match PlayerRole enum
        role_map = {
            "Batsman": "Batsman", "Batter": "Batsman",
            "Bowler": "Bowler",
            "All-Rounder": "All-Rounder", "All Rounder": "All-Rounder",
            "Wicket Keeper": "Wicket Keeper", "WK-Batsman": "Wicket Keeper",
            "Wicket-Keeper": "Wicket Keeper",
        }
        df["Role"] = df["Role"].map(role_map).fillna("Batsman")
        df["Player Origin"] = df["Player Origin"].fillna("Indian")

        self.df = df
        self._build_player_database()
        self._compute_features()

    # ─────────────────────────────────────────
    # Player Database
    # ─────────────────────────────────────────

    def _build_player_database(self):
        for name, group in self.df.groupby("Player"):
            group = group.sort_values("Year")
            history = []
            for _, row in group.iterrows():
                history.append({
                    "year": int(row["Year"]),
                    "team": row["Team"],
                    "team_code": row.get("Team_Code", ""),
                    "amount": int(row["Amount"]),
                    "amount_cr": round(row["Amount_Cr"], 2),
                })

            latest = group.iloc[-1]
            max_price = group["Amount"].max()
            appearances = len(group)

            if max_price >= 50_000_000 or appearances >= 4:
                set_type = AuctionSetType.MARQUEE
            elif latest.get("Player Origin") == "Indian" and appearances <= 1:
                set_type = AuctionSetType.UNCAPPED
            else:
                set_type = AuctionSetType.CAPPED

            role_val = latest.get("Role", "Batsman")
            try:
                role = PlayerRole(role_val)
            except ValueError:
                role = PlayerRole.BATTER

            origin_val = latest.get("Player Origin", "Indian")
            try:
                origin = PlayerOrigin(origin_val)
            except ValueError:
                origin = PlayerOrigin.INDIAN

            self.players_db[name] = Player(
                name=name,
                role=role,
                origin=origin,
                base_price=self._estimate_base_price(group),
                historical_prices=history,
                set_type=set_type,
                stats=self._compute_player_stats(name, group),
            )

    def _estimate_base_price(self, player_group: pd.DataFrame) -> int:
        latest_price = player_group.iloc[-1]["Amount"]
        avg_price = player_group["Amount"].mean()
        estimated = min(latest_price, avg_price) * 0.3
        slabs = [2_000_000, 5_000_000, 10_000_000, 15_000_000, 20_000_000,
                 50_000_000, 75_000_000, 100_000_000, 150_000_000, 200_000_000]
        for slab in slabs:
            if estimated <= slab:
                return slab
        return 200_000_000

    def _compute_player_stats(self, name: str, group: pd.DataFrame) -> dict:
        prices = group["Amount"].values
        years = group["Year"].values

        # Batting stats (from batting CSV columns)
        batting_stats = {}
        if "batting_avg" in group.columns:
            batting_stats["batting_avg"] = float(pd.to_numeric(group["batting_avg"], errors="coerce").mean() or 0)
        if "batting_sr" in group.columns:
            batting_stats["batting_sr"] = float(pd.to_numeric(group["batting_sr"], errors="coerce").mean() or 0)
        if "total_runs" in group.columns:
            val = pd.to_numeric(group["total_runs"], errors="coerce").max()
            batting_stats["total_runs"] = int(val) if pd.notna(val) else 0

        # Bowling stats (from bowling CSV columns)
        bowling_stats = {}
        if "wickets" in group.columns:
            val = pd.to_numeric(group["wickets"], errors="coerce").max()
            bowling_stats["wickets"] = int(val) if pd.notna(val) else 0
        if "economy" in group.columns:
            bowling_stats["economy"] = float(pd.to_numeric(group["economy"], errors="coerce").mean() or 0)
        if "bowling_sr" in group.columns:
            bowling_stats["bowling_sr"] = float(pd.to_numeric(group["bowling_sr"], errors="coerce").mean() or 0)
        if "bowler_category" in group.columns:
            cats = group["bowler_category"].dropna().unique().tolist()
            bowling_stats["bowler_category"] = cats[0] if cats else ""

        stats = {
            "auction_appearances": len(group),
            "total_teams": group["Team"].nunique(),
            "avg_price": float(np.mean(prices)),
            "max_price": float(np.max(prices)),
            "min_price": float(np.min(prices)),
            "latest_price": float(prices[-1]),
            "latest_year": int(years[-1]),
            "price_trend": self._compute_price_trend(prices),
            "volatility": float(np.std(prices) / np.mean(prices)) if len(prices) > 1 else 0.0,
            "peak_year": int(years[np.argmax(prices)]),
            "years_since_peak": int(years[-1] - years[np.argmax(prices)]),
            **batting_stats,
            **bowling_stats,
        }

        if len(prices) >= 2:
            recent_trend = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0
            stats["trajectory"] = (
                "rising_star" if recent_trend > 0.3
                else "declining" if recent_trend < -0.3
                else "stable"
            )
        else:
            stats["trajectory"] = "unknown"

        return stats

    def _compute_price_trend(self, prices: np.ndarray) -> float:
        if len(prices) < 2:
            return 0.0
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return float(slope / np.mean(prices))

    def _compute_features(self):
        records = []
        for name, player in self.players_db.items():
            s = player.stats
            records.append({
                "player": name,
                "role": player.role.value,
                "origin": player.origin.value,
                "base_price": player.base_price,
                "auction_appearances": s["auction_appearances"],
                "avg_price_cr": s["avg_price"] / 1e7,
                "max_price_cr": s["max_price"] / 1e7,
                "latest_price_cr": s["latest_price"] / 1e7,
                "price_trend": s["price_trend"],
                "volatility": s["volatility"],
                "trajectory": s["trajectory"],
                "set_type": player.set_type.value,
            })
        self.features_df = pd.DataFrame(records)

    # ─────────────────────────────────────────
    # Query Methods
    # ─────────────────────────────────────────

    def get_player(self, name: str) -> Optional[Player]:
        return self.players_db.get(name)

    def get_players_by_role(self, role: PlayerRole) -> list[Player]:
        return [p for p in self.players_db.values() if p.role == role]

    def get_players_by_set(self, set_type: AuctionSetType) -> list[Player]:
        return [p for p in self.players_db.values() if p.set_type == set_type]

    def get_team_history(self, team_code: str) -> pd.DataFrame:
        return self.df[self.df["Team_Code"] == team_code].sort_values(
            ["Year", "Amount"], ascending=[True, False]
        )

    def get_market_summary(self, year: Optional[int] = None) -> dict:
        data = self.df if year is None else self.df[self.df["Year"] == year]
        if data.empty:
            return {"total_players": 0, "total_spend_cr": 0, "avg_price_cr": 0,
                    "median_price_cr": 0, "max_price_cr": 0, "by_role": {}, "by_origin": {}}
        return {
            "total_players": len(data),
            "total_spend_cr": round(data["Amount"].sum() / 1e7, 2),
            "avg_price_cr": round(data["Amount"].mean() / 1e7, 2),
            "median_price_cr": round(data["Amount"].median() / 1e7, 2),
            "max_price_cr": round(data["Amount"].max() / 1e7, 2),
            "by_role": data.groupby("Role")["Amount"].mean().div(1e7).round(2).to_dict(),
            "by_origin": data.groupby("Player Origin")["Amount"].mean().div(1e7).round(2).to_dict(),
        }

    def generate_auction_pool(self, year: Optional[int] = None) -> list[Player]:
        """
        Generate a player pool.
        If year given, returns players auctioned that year.
        If no year, returns all unique players (deduped to latest appearance).
        """
        if year:
            year_data = self.df[self.df["Year"] == year]
            return [self.players_db[row["Player"]]
                    for _, row in year_data.iterrows()
                    if row["Player"] in self.players_db]

        # All-time pool: latest record per player name
        seen = set()
        pool = []
        for name, player in sorted(
            self.players_db.items(),
            key=lambda x: x[1].stats["latest_year"],
            reverse=True,
        ):
            if name not in seen:
                seen.add(name)
                pool.append(player)
        return pool

    def get_comparable_players(self, player: Player, top_n: int = 5) -> list[dict]:
        candidates = []
        for name, p in self.players_db.items():
            if name == player.name:
                continue
            score = 0
            if p.role == player.role:
                score += 3
            if p.origin == player.origin:
                score += 2
            if player.stats["avg_price"] > 0:
                price_ratio = min(p.stats["avg_price"], player.stats["avg_price"]) / \
                              max(p.stats["avg_price"], player.stats["avg_price"])
                score += price_ratio * 2
            candidates.append({"player": name, "score": score, "data": p})

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]

    def get_price_prediction_context(self, player_name: str) -> str:
        player = self.get_player(player_name)
        if not player:
            return f"No data found for {player_name}"

        s = player.stats
        comparables = self.get_comparable_players(player, top_n=3)

        ctx = f"""## Player Profile: {player.name}
- Role: {player.role.value}
- Origin: {player.origin.value}
- Auction Appearances: {s['auction_appearances']}

## Price History:
"""
        for h in player.historical_prices:
            ctx += f"  {h['year']}: {h['team']} — ₹{h['amount_cr']}Cr\n"

        ctx += f"""
## Analytics:
- Average Price: ₹{s['avg_price']/1e7:.2f}Cr
- Peak Price: ₹{s['max_price']/1e7:.2f}Cr (in {s['peak_year']})
- Latest Price: ₹{s['latest_price']/1e7:.2f}Cr
- Price Trend: {'Rising' if s['price_trend'] > 0.1 else 'Declining' if s['price_trend'] < -0.1 else 'Stable'}
- Trajectory: {s['trajectory']}

## Comparable Players:
"""
        for comp in comparables:
            cp = comp["data"]
            ctx += f"  - {cp.name} ({cp.role.value}, {cp.origin.value}): avg ₹{cp.stats['avg_price']/1e7:.2f}Cr\n"

        return ctx
