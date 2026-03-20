"""
IPL Auction Configuration — Pure stdlib, no external deps.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class AuctionType(str, Enum):
    MEGA = "mega"
    MINI = "mini"

class PlayerRole(str, Enum):
    BATTER = "Batsman"
    BOWLER = "Bowler"
    ALL_ROUNDER = "All-Rounder"
    WICKET_KEEPER = "Wicket Keeper"

class PlayerOrigin(str, Enum):
    INDIAN = "Indian"
    OVERSEAS = "Overseas"

class BidAction(str, Enum):
    BID = "bid"
    PASS = "pass"
    RTM = "rtm"

class AuctionSetType(str, Enum):
    MARQUEE = "marquee"
    CAPPED = "capped"
    UNCAPPED = "uncapped"
    ACCELERATED = "accelerated"

@dataclass
class Player:
    name: str
    role: PlayerRole
    origin: PlayerOrigin
    base_price: int
    age: Optional[int] = None
    ipl_caps: Optional[int] = None
    historical_prices: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    set_type: AuctionSetType = AuctionSetType.CAPPED

@dataclass
class TeamSquad:
    remaining_purse: int
    total_purse: int
    max_squad_size: int
    min_squad_size: int = 18
    max_overseas: int = 8
    rtm_cards: int = 0
    players: list = field(default_factory=list)

    @property
    def current_size(self): return len(self.players)
    @property
    def overseas_count(self): return sum(1 for p in self.players if p.origin == PlayerOrigin.OVERSEAS)
    @property
    def slots_remaining(self): return self.max_squad_size - self.current_size
    @property
    def overseas_slots_remaining(self): return self.max_overseas - self.overseas_count
    @property
    def min_slots_to_fill(self): return max(0, self.min_squad_size - self.current_size)
    @property
    def effective_max_bid(self):
        reserve = max(0, self.min_slots_to_fill - 1) * 2_000_000
        return self.remaining_purse - reserve
    def can_buy_overseas(self): return self.overseas_slots_remaining > 0
    def role_distribution(self):
        d = {}
        for p in self.players:
            d[p.role.value] = d.get(p.role.value, 0) + 1
        return d

@dataclass
class AuctionConfig:
    auction_type: AuctionType
    total_purse: int
    max_squad_size: int
    min_squad_size: int = 18
    max_overseas: int = 8
    rtm_cards_per_team: int = 0
    max_retentions: int = 0
    has_accelerated_round: bool = True

BID_INCREMENTS = [
    (10_000_000, 500_000), (20_000_000, 1_000_000), (50_000_000, 2_500_000),
    (100_000_000, 5_000_000), (200_000_000, 10_000_000), (999_999_999_999, 25_000_000),
]

def get_bid_increment(current_bid: int) -> int:
    for threshold, inc in BID_INCREMENTS:
        if current_bid < threshold:
            return inc
    return 25_000_000

MEGA_AUCTION_CONFIG = AuctionConfig(AuctionType.MEGA, 900_000_000, 25, 18, 8, 2, 4, True)
MINI_AUCTION_CONFIG = AuctionConfig(AuctionType.MINI, 950_000_000, 25, 18, 8, 0, 0, True)

TEAM_NAME_MAP = {
    "Chennai Super Kings": "CSK", "Mumbai Indians": "MI",
    "Royal Challengers Bangalore": "RCB", "Kolkata Knight Riders": "KKR",
    "Rajasthan Royals": "RR", "Delhi Capitals": "DC", "Delhi Daredevils": "DC",
    "Punjab Kings": "PBKS", "Kings XI Punjab": "PBKS", "Sunrisers Hyderabad": "SRH",
    "Gujarat Titans": "GT", "Lucknow Super Giants": "LSG",
    "Rising Pune Supergiant": "RPS", "Gujarat Lions": "GL", "Pune Warriors India": "PWI",
}

ACTIVE_TEAMS = ["CSK", "MI", "RCB", "KKR", "RR", "DC", "PBKS", "SRH", "GT", "LSG"]

TEAM_FULL_NAMES = {
    "CSK": "Chennai Super Kings", "MI": "Mumbai Indians",
    "RCB": "Royal Challengers Bangalore", "KKR": "Kolkata Knight Riders",
    "RR": "Rajasthan Royals", "DC": "Delhi Capitals", "PBKS": "Punjab Kings",
    "SRH": "Sunrisers Hyderabad", "GT": "Gujarat Titans", "LSG": "Lucknow Super Giants",
}

TEAM_DNA = {
    "CSK": {"full_name": "Chennai Super Kings", "philosophy": "Experience over youth. Trust proven performers. MS Dhoni's school: buy match-winners for big moments. Prefer Indian core with selective overseas.", "archetype": "The Wise Elder — patient, calculated, backs experience", "historical_preferences": {"prefer_roles": ["All-Rounder", "Bowler"], "prefer_origin": "balanced", "age_preference": "experienced (28+)", "risk_tolerance": "low", "bidding_style": "patient — enters late, decisive when committed"}, "icon_players": ["MS Dhoni", "Suresh Raina", "Ravindra Jadeja", "Faf du Plessis"]},
    "MI": {"full_name": "Mumbai Indians", "philosophy": "Invest in young Indian talent and develop them. Buy cheap uncapped players who become superstars. Complement with 2-3 marquee overseas. Never overpay in panic.", "archetype": "The Talent Factory — scouts, develops, builds dynasties", "historical_preferences": {"prefer_roles": ["All-Rounder", "Batsman"], "prefer_origin": "Indian-heavy", "age_preference": "young (21-26)", "risk_tolerance": "medium-high on unknowns", "bidding_style": "strategic — knows when to let go, invests in depth"}, "icon_players": ["Rohit Sharma", "Jasprit Bumrah", "Hardik Pandya", "Kieron Pollard"]},
    "RCB": {"full_name": "Royal Challengers Bangalore", "philosophy": "Chase star power and marquee names. Emotional bidding for Indian stars and explosive overseas batters. Prone to overspending on top 3-4 players.", "archetype": "The Star Chaser — bold, emotional, big-name hungry", "historical_preferences": {"prefer_roles": ["Batsman", "All-Rounder"], "prefer_origin": "balanced", "age_preference": "peak (25-32)", "risk_tolerance": "high", "bidding_style": "aggressive — enters early, gets into bidding wars"}, "icon_players": ["Virat Kohli", "AB de Villiers", "Chris Gayle", "Glenn Maxwell"]},
    "KKR": {"full_name": "Kolkata Knight Riders", "philosophy": "Data-driven approach with flair for spin bowling and mystery. Value smart under-the-radar overseas picks. Calculated risks.", "archetype": "The Analyst — data-driven, spin-obsessed, strategic", "historical_preferences": {"prefer_roles": ["Bowler", "All-Rounder"], "prefer_origin": "balanced", "age_preference": "mixed", "risk_tolerance": "medium", "bidding_style": "calculated — uses data, targets specific profiles"}, "icon_players": ["Sunil Narine", "Andre Russell", "Shreyas Iyer", "Gautam Gambhir"]},
    "RR": {"full_name": "Rajasthan Royals", "philosophy": "Moneyball approach — find value where others don't look. Strong analytics. Willing to buy unproven overseas at bargain prices. Youth development.", "archetype": "The Moneyball Pioneer — analytics-first, value-seeking", "historical_preferences": {"prefer_roles": ["All-Rounder", "Bowler"], "prefer_origin": "overseas-leaning", "age_preference": "young-to-mid (22-28)", "risk_tolerance": "high on undervalued players", "bidding_style": "value-hunting — sets strict ceilings, walks away cleanly"}, "icon_players": ["Jos Buttler", "Sanju Samson", "Yashasvi Jaiswal", "Yuzvendra Chahal"]},
    "DC": {"full_name": "Delhi Capitals", "philosophy": "Build a young, exciting core for the future. Attracted to pace bowling and explosive top-order batters.", "archetype": "The Rebuilder — young, aggressive, future-focused", "historical_preferences": {"prefer_roles": ["Batsman", "Bowler"], "prefer_origin": "Indian-heavy", "age_preference": "young (21-27)", "risk_tolerance": "medium-high", "bidding_style": "reactive — watches others, then pounces"}, "icon_players": ["Rishabh Pant", "Anrich Nortje", "Axar Patel", "David Warner"]},
    "PBKS": {"full_name": "Punjab Kings", "philosophy": "Most unpredictable bidders. Known for overpaying and aggressive early spending. Willing to break the bank for overseas all-rounders.", "archetype": "The Wild Card — unpredictable, aggressive, emotional", "historical_preferences": {"prefer_roles": ["All-Rounder", "Batsman"], "prefer_origin": "overseas-heavy", "age_preference": "peak (26-32)", "risk_tolerance": "very high", "bidding_style": "aggressive — first to bid, last to stop"}, "icon_players": ["KL Rahul", "Chris Gayle", "Liam Livingstone", "Shikhar Dhawan"]},
    "SRH": {"full_name": "Sunrisers Hyderabad", "philosophy": "Bowling-first philosophy. Build world-class bowling attack and fill batting around it. Value economy bowlers. Disciplined spending.", "archetype": "The Bowling Machine — disciplined, bowling-first, clinical", "historical_preferences": {"prefer_roles": ["Bowler", "All-Rounder"], "prefer_origin": "balanced", "age_preference": "experienced (27-33)", "risk_tolerance": "low", "bidding_style": "disciplined — strict price ceilings, avoids wars"}, "icon_players": ["David Warner", "Rashid Khan", "Bhuvneshwar Kumar", "Kane Williamson"]},
    "GT": {"full_name": "Gujarat Titans", "philosophy": "New franchise energy — built to win immediately. Aggressive but smart. Invested in Indian match-winners and versatile overseas.", "archetype": "The New Power — ambitious, fearless, win-now mentality", "historical_preferences": {"prefer_roles": ["All-Rounder", "Bowler"], "prefer_origin": "balanced", "age_preference": "peak (25-30)", "risk_tolerance": "medium-high", "bidding_style": "strategic aggression — targets key players hard"}, "icon_players": ["Hardik Pandya", "Rashid Khan", "Shubman Gill", "Mohammed Shami"]},
    "LSG": {"full_name": "Lucknow Super Giants", "philosophy": "New franchise, corporate methodical approach. Heavy investment in top Indian talent with selective overseas buys. Prefer reliability over flair.", "archetype": "The Corporate — methodical, reliable, process-driven", "historical_preferences": {"prefer_roles": ["Batsman", "Bowler"], "prefer_origin": "Indian-heavy", "age_preference": "mid (24-30)", "risk_tolerance": "low-medium", "bidding_style": "methodical — pre-planned targets, disciplined execution"}, "icon_players": ["KL Rahul", "Quinton de Kock", "Mark Wood", "Avesh Khan"]},
}
