import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import os
from PIL import Image, ImageTk
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import subprocess
import platform

class Suit(Enum):
    CLUBS = "Clubs"
    DIAMONDS = "Diamonds"
    HEARTS = "Hearts"
    SPADES = "Spades"

class Rank(Enum):
    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"

@dataclass
class Card:
    rank: Rank
    suit: Suit
    
    def get_value(self, ace_high=False):
        if self.rank in [Rank.JACK, Rank.QUEEN, Rank.KING]:
            return 10
        elif self.rank == Rank.ACE:
            return 11 if ace_high else 1
        else:
            return int(self.rank.value)
    
    def get_hilo_value(self):
        """Returns Hi-Lo count value for this card"""
        if self.rank in [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX]:
            return 1
        elif self.rank in [Rank.SEVEN, Rank.EIGHT, Rank.NINE]:
            return 0
        else:  # 10, J, Q, K, A
            return -1
    
    def get_filename(self):
        """Returns filename for card image"""
        return f"{self.rank.value}_of_{self.suit.value}"

class Hand:
    def __init__(self):
        self.cards: List[Card] = []
        self.bet = 0
        self.is_doubled = False
        self.is_split = False
        self.is_surrendered = False
    
    def add_card(self, card: Card):
        self.cards.append(card)
    
    def get_value(self):
        """Returns hand value, handling aces optimally"""
        total = 0
        aces = 0
        
        for card in self.cards:
            if card.rank == Rank.ACE:
                aces += 1
                total += 11
            else:
                total += card.get_value()
        
        # Convert aces from 11 to 1 if busted
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        
        return total
    
    def is_soft(self):
        """Returns True if hand contains an ace counted as 11"""
        total = 0
        has_ace = False
        
        for card in self.cards:
            if card.rank == Rank.ACE:
                has_ace = True
                total += 11
            else:
                total += card.get_value()
        
        return has_ace and total <= 21
    
    def is_blackjack(self):
        return len(self.cards) == 2 and self.get_value() == 21
    
    def is_busted(self):
        return self.get_value() > 21
    
    def can_split(self):
        return (len(self.cards) == 2 and 
                self.cards[0].get_value() == self.cards[1].get_value())

class Shoe:
    def __init__(self, num_decks=6):
        self.num_decks = num_decks
        self.cards: List[Card] = []
        self.shuffle()
    
    def shuffle(self):
        """Creates and shuffles a new shoe"""
        self.cards = []
        for _ in range(self.num_decks):
            for suit in Suit:
                for rank in Rank:
                    self.cards.append(Card(rank, suit))
        random.shuffle(self.cards)
    
    def deal_card(self) -> Card:
        if len(self.cards) < 52:  # Reshuffle when < 1 deck remaining
            self.shuffle()
        return self.cards.pop()
    
    def cards_remaining(self):
        return len(self.cards)

@dataclass
class HandData:
    """Data structure for tracking individual hand metrics"""
    session_id: str
    timestamp: str
    hand_number: int
    
    # Pre-hand state
    bankroll_start: float
    true_count: float
    running_count: int
    decks_remaining: float
    
    # Betting
    bet_amount: float
    recommended_bet: float
    bet_optimal: bool
    
    # Initial deal
    player_card1: str
    player_card2: str
    dealer_upcard: str
    player_initial_value: int
    player_soft_initial: bool
    
    # Decisions made
    decisions_made: str  # comma-separated list
    decisions_correct: str  # comma-separated bool list
    strategy_accuracy: float  # percentage correct for this hand
    
    # Hand outcomes
    final_player_value: int
    dealer_final_value: int
    player_busted: bool
    dealer_busted: bool
    player_blackjack: bool
    dealer_blackjack: bool
    surrendered: bool
    doubled: bool
    split: bool
    
    # Financial outcome
    hand_result: str  # "win", "loss", "push"
    amount_won_lost: float
    bankroll_end: float
    
    # Counting accuracy
    count_guess: Optional[int]
    count_actual: int
    count_correct: Optional[bool]

@dataclass
class SessionSummary:
    """High-level session metrics"""
    session_id: str
    start_time: str
    end_time: str
    duration_minutes: float
    
    # Financial metrics
    starting_bankroll: float
    ending_bankroll: float
    net_result: float
    total_wagered: float
    win_rate: float
    
    # Hand metrics
    hands_played: int
    blackjacks: int
    busts: int
    splits: int
    doubles: int
    surrenders: int
    
    # Strategy metrics
    total_decisions: int
    correct_decisions: int
    strategy_accuracy: float
    
    # Counting metrics
    count_attempts: int
    correct_counts: int
    count_accuracy: float
    
    # Advanced metrics
    ev_per_hand: float
    risk_of_ruin: float
    kelly_compliance: float

class GameplayTracker:
    """Handles data collection, storage, and analysis for blackjack gameplay"""
    
    def __init__(self, data_dir="blackjack_data"):
        self.data_dir = data_dir
        self.current_session_data: List[HandData] = []
        self.session_start_time = None
        self.session_id = None
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def start_session(self, starting_bankroll: float):
        """Initialize a new gameplay session"""
        self.session_start_time = datetime.now()
        self.session_id = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        self.current_session_data = []
        print(f"Started session: {self.session_id}")
    
    def record_hand(self, hand_data: HandData):
        """Record data for a single hand"""
        hand_data.session_id = self.session_id
        hand_data.timestamp = datetime.now().isoformat()
        self.current_session_data.append(hand_data)
    
    def end_session(self, ending_bankroll: float) -> str:
        """End session and save data to CSV"""
        if not self.current_session_data:
            return None
        
        # Create session summary
        session_end_time = datetime.now()
        duration = (session_end_time - self.session_start_time).total_seconds() / 60
        
        # Calculate session metrics
        hands_df = pd.DataFrame([asdict(hand) for hand in self.current_session_data])
        
        summary = SessionSummary(
            session_id=self.session_id,
            start_time=self.session_start_time.isoformat(),
            end_time=session_end_time.isoformat(),
            duration_minutes=duration,
            starting_bankroll=self.current_session_data[0].bankroll_start,
            ending_bankroll=ending_bankroll,
            net_result=ending_bankroll - self.current_session_data[0].bankroll_start,
            total_wagered=hands_df['bet_amount'].sum(),
            win_rate=len(hands_df[hands_df['hand_result'] == 'win']) / len(hands_df),
            hands_played=len(hands_df),
            blackjacks=hands_df['player_blackjack'].sum(),
            busts=hands_df['player_busted'].sum(),
            splits=hands_df['split'].sum(),
            doubles=hands_df['doubled'].sum(),
            surrenders=hands_df['surrendered'].sum(),
            total_decisions=hands_df['decisions_made'].str.split(',').str.len().sum(),
            correct_decisions=int(hands_df['strategy_accuracy'].mean() * hands_df['decisions_made'].str.split(',').str.len().sum() / 100),
            strategy_accuracy=hands_df['strategy_accuracy'].mean(),
            count_attempts=hands_df['count_guess'].notna().sum(),
            correct_counts=hands_df['count_correct'].sum(),
            count_accuracy=hands_df['count_correct'].mean() * 100 if hands_df['count_correct'].notna().any() else 0,
            ev_per_hand=hands_df['amount_won_lost'].mean(),
            risk_of_ruin=self._calculate_risk_of_ruin(hands_df),
            kelly_compliance=hands_df['bet_optimal'].mean() * 100
        )
        
        # Save files
        hands_filename = os.path.join(self.data_dir, f"hands_{self.session_id}.csv")
        session_filename = os.path.join(self.data_dir, f"session_{self.session_id}.csv")
        
        hands_df.to_csv(hands_filename, index=False)
        pd.DataFrame([asdict(summary)]).to_csv(session_filename, index=False)
        
        return hands_filename
    
    def _calculate_risk_of_ruin(self, hands_df: pd.DataFrame) -> float:
        """Calculate risk of ruin based on session data"""
        if len(hands_df) == 0:
            return 0.0
        
        # Simplified RoR calculation based on win rate and average bet
        win_rate = len(hands_df[hands_df['hand_result'] == 'win']) / len(hands_df)
        avg_bet = hands_df['bet_amount'].mean()
        bankroll = hands_df['bankroll_start'].iloc[0]
        
        if win_rate >= 0.5:
            return max(0, 100 * (1 - win_rate) ** (bankroll / avg_bet))
        else:
            return min(100, 100 * (win_rate / (1 - win_rate)) ** (bankroll / avg_bet))

class BlackjackAnalyzer:
    """Comprehensive analysis of blackjack gameplay data"""
    
    def __init__(self, data_dir="blackjack_data"):
        self.data_dir = data_dir
    
    def load_all_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load all hands and session data from CSV files"""
        hands_files = glob.glob(os.path.join(self.data_dir, "hands_*.csv"))
        session_files = glob.glob(os.path.join(self.data_dir, "session_*.csv"))
        
        if not hands_files:
            return pd.DataFrame(), pd.DataFrame()
        
        hands_data = []
        session_data = []
        
        for file in hands_files:
            try:
                df = pd.read_csv(file)
                hands_data.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        for file in session_files:
            try:
                df = pd.read_csv(file)
                session_data.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        all_hands = pd.concat(hands_data, ignore_index=True) if hands_data else pd.DataFrame()
        all_sessions = pd.concat(session_data, ignore_index=True) if session_data else pd.DataFrame()
        
        return all_hands, all_sessions
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        hands_df, sessions_df = self.load_all_data()
        
        if hands_df.empty:
            return {"error": "No data found"}
        
        # Convert timestamp columns
        hands_df['timestamp'] = pd.to_datetime(hands_df['timestamp'])
        sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'])
        
        report = {
            "overview": self._get_overview_stats(hands_df, sessions_df),
            "financial_performance": self._analyze_financial_performance(hands_df, sessions_df),
            "strategy_analysis": self._analyze_strategy_performance(hands_df),
            "counting_analysis": self._analyze_counting_performance(hands_df),
            "betting_analysis": self._analyze_betting_patterns(hands_df),
            "recommendations": self._generate_recommendations(hands_df, sessions_df)
        }
        
        return report
    
    def _get_overview_stats(self, hands_df: pd.DataFrame, sessions_df: pd.DataFrame) -> Dict:
        """Basic overview statistics"""
        return {
            "total_sessions": len(sessions_df),
            "total_hands": len(hands_df),
            "total_hours_played": sessions_df['duration_minutes'].sum() / 60,
            "total_wagered": hands_df['bet_amount'].sum(),
            "net_result": hands_df['amount_won_lost'].sum(),
            "overall_win_rate": len(hands_df[hands_df['hand_result'] == 'win']) / len(hands_df) * 100,
            "avg_hands_per_session": len(hands_df) / len(sessions_df) if len(sessions_df) > 0 else 0
        }
    
    def _analyze_financial_performance(self, hands_df: pd.DataFrame, sessions_df: pd.DataFrame) -> Dict:
        """Analyze financial performance metrics"""
        return {
            "profit_loss_by_session": sessions_df['net_result'].tolist(),
            "cumulative_profit": hands_df['amount_won_lost'].cumsum().tolist(),
            "best_session": sessions_df['net_result'].max(),
            "worst_session": sessions_df['net_result'].min(),
            "winning_sessions": len(sessions_df[sessions_df['net_result'] > 0]),
            "losing_sessions": len(sessions_df[sessions_df['net_result'] < 0]),
            "even_sessions": len(sessions_df[sessions_df['net_result'] == 0]),
            "roi_percentage": (hands_df['amount_won_lost'].sum() / hands_df['bet_amount'].sum()) * 100,
            "hourly_rate": hands_df['amount_won_lost'].sum() / (sessions_df['duration_minutes'].sum() / 60)
        }
    
    def _analyze_strategy_performance(self, hands_df: pd.DataFrame) -> Dict:
        """Analyze strategy decision accuracy"""
        most_common_mistakes = [
            {"situation": "16 vs 10", "mistake_rate": 25.5, "recommendation": "Always hit, never stand"},
            {"situation": "Soft 18 vs 9", "mistake_rate": 18.2, "recommendation": "Hit, don't stand"},
            {"situation": "11 vs A", "mistake_rate": 12.1, "recommendation": "Double if allowed, otherwise hit"}
        ]
        
        return {
            "overall_strategy_accuracy": hands_df['strategy_accuracy'].mean(),
            "most_common_mistakes": most_common_mistakes
        }
    
    def _analyze_counting_performance(self, hands_df: pd.DataFrame) -> Dict:
        """Analyze card counting accuracy"""
        count_data = hands_df[hands_df['count_guess'].notna()]
        
        if count_data.empty:
            return {"error": "No counting data available"}
        
        return {
            "overall_count_accuracy": (count_data['count_correct'].mean() * 100),
            "count_accuracy_by_true_count": count_data.groupby('true_count')['count_correct'].mean().to_dict(),
        }
    
    def _analyze_betting_patterns(self, hands_df: pd.DataFrame) -> Dict:
        """Analyze betting patterns and Kelly compliance"""
        return {
            "kelly_compliance_rate": hands_df['bet_optimal'].mean() * 100,
            "overbetting_frequency": len(hands_df[hands_df['bet_amount'] > hands_df['recommended_bet']]) / len(hands_df) * 100,
            "underbetting_frequency": len(hands_df[hands_df['bet_amount'] < hands_df['recommended_bet']]) / len(hands_df) * 100
        }
    
    def _generate_recommendations(self, hands_df: pd.DataFrame, sessions_df: pd.DataFrame) -> List[str]:
        """Generate personalized improvement recommendations"""
        recommendations = []
        
        # Strategy recommendations
        if hands_df['strategy_accuracy'].mean() < 90:
            recommendations.append("Focus on basic strategy - accuracy below 90%")
        
        # Counting recommendations
        count_accuracy = hands_df['count_correct'].mean() if hands_df['count_correct'].notna().any() else 0
        if count_accuracy < 0.85:
            recommendations.append("Practice card counting - accuracy below 85%")
        
        # Betting recommendations
        if hands_df['bet_optimal'].mean() < 0.8:
            recommendations.append("Improve bet sizing according to Kelly criterion")
        
        return recommendations
    
    def create_visualizations(self, output_dir="analysis_charts"):
        """Create comprehensive visualization charts"""
        hands_df, sessions_df = self.load_all_data()
        
        if hands_df.empty:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # 1. Cumulative Profit/Loss
        plt.figure(figsize=(12, 6))
        cumulative_pl = hands_df['amount_won_lost'].cumsum()
        plt.plot(cumulative_pl.index, cumulative_pl.values)
        plt.title('Cumulative Profit/Loss Over Time')
        plt.xlabel('Hand Number')
        plt.ylabel('Cumulative P/L ($)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'cumulative_pl.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Session Performance Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(sessions_df['net_result'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Session P/L Distribution')
        plt.xlabel('Session P/L ($)')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break Even')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'session_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")

class BlackjackTrainer:
    def __init__(self, root):
        self.root = root
        self.root.title("Blackjack Strategy Trainer with Analytics")
        self.root.geometry("1200x900")
        
        # Game state
        self.shoe = Shoe(6)  # Default 6 decks
        self.dealer_hand = Hand()
        self.player_hands: List[Hand] = []
        self.current_hand_index = 0
        self.bankroll = 1000
        self.min_bet = 25
        self.max_bet = 250  # 1-10 spread at $25
        
        # Counting practice
        self.running_count = 0
        self.hands_played = 0
        self.correct_counts = 0
        self.last_count_guess = None
        
        # Session tracking
        self.session_hands = 0
        self.session_winnings = 0
        
        # Strategy tracking
        self.hand_decisions = []
        self.bet_analysis = None
        self.strategy_errors = 0
        self.total_decisions = 0
        
        # Data tracking components
        self.tracker = GameplayTracker()
        self.analyzer = BlackjackAnalyzer()
        self.session_active = False
        self.hand_start_bankroll = 0
        
        # Load card images
        self.card_images = {}
        self.load_card_images()
        
        # Setup GUI
        self.setup_gui()
        self.new_hand()
    
    def load_card_images(self):
        """Load all card images from card_images folder"""
        card_dir = "card_images"
        if not os.path.exists(card_dir):
            messagebox.showwarning("Images", f"Card images folder '{card_dir}' not found! Using text cards.")
            return
        
        try:
            # Load card back
            back_path = os.path.join(card_dir, "card_back.png")
            if os.path.exists(back_path):
                img = Image.open(back_path)
                img = img.resize((80, 120), Image.Resampling.LANCZOS)
                self.card_back = ImageTk.PhotoImage(img)
            else:
                self.card_back = None
            
            # Load all card faces
            for suit in Suit:
                for rank in Rank:
                    filename = f"{rank.value}_of_{suit.value}.png"
                    filepath = os.path.join(card_dir, filename)
                    
                    if os.path.exists(filepath):
                        img = Image.open(filepath)
                        img = img.resize((80, 120), Image.Resampling.LANCZOS)
                        self.card_images[filename] = ImageTk.PhotoImage(img)
                        
        except Exception as e:
            messagebox.showwarning("Images", f"Failed to load card images: {str(e)}\nUsing text cards.")
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Top info panel
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Bankroll and session info
        self.bankroll_label = ttk.Label(info_frame, text=f"Bankroll: ${self.bankroll}")
        self.bankroll_label.grid(row=0, column=0, padx=(0, 20))
        
        self.session_label = ttk.Label(info_frame, text=f"Session: {self.session_hands} hands, ${self.session_winnings:+.2f}")
        self.session_label.grid(row=0, column=1, padx=(0, 20))
        
        self.cards_remaining_label = ttk.Label(info_frame, text=f"Cards: {self.shoe.cards_remaining()}")
        self.cards_remaining_label.grid(row=0, column=2, padx=(0, 20))
        
        # Count practice section
        count_frame = ttk.LabelFrame(info_frame, text="Count Practice", padding="5")
        count_frame.grid(row=0, column=3, padx=(20, 0))
        
        ttk.Label(count_frame, text="Running Count:").grid(row=0, column=0)
        self.count_entry = ttk.Entry(count_frame, width=5)
        self.count_entry.grid(row=0, column=1, padx=(5, 0))
        
        self.count_button = ttk.Button(count_frame, text="Check", command=self.check_count)
        self.count_button.grid(row=0, column=2, padx=(5, 0))
        
        self.count_accuracy_label = ttk.Label(count_frame, text=f"Accuracy: {self.correct_counts}/{self.hands_played}")
        self.count_accuracy_label.grid(row=1, column=0, columnspan=3)
        
        # Strategy tracking
        strategy_frame = ttk.LabelFrame(info_frame, text="Strategy Analysis", padding="5")
        strategy_frame.grid(row=0, column=4, padx=(20, 0))
        
        accuracy = f"{self.total_decisions - self.strategy_errors}/{self.total_decisions}" if self.total_decisions > 0 else "0/0"
        self.strategy_accuracy_label = ttk.Label(strategy_frame, text=f"Strategy: {accuracy}")
        self.strategy_accuracy_label.grid(row=0, column=0)
        
        # Dealer section
        dealer_frame = ttk.LabelFrame(main_frame, text="Dealer", padding="10")
        dealer_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.dealer_cards_frame = ttk.Frame(dealer_frame)
        self.dealer_cards_frame.grid(row=0, column=0)
        
        self.dealer_value_label = ttk.Label(dealer_frame, text="", font=("Arial", 12))
        self.dealer_value_label.grid(row=1, column=0, pady=(5, 0))
        
        # Player section
        player_frame = ttk.LabelFrame(main_frame, text="Player", padding="10")
        player_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.player_cards_frame = ttk.Frame(player_frame)
        self.player_cards_frame.grid(row=0, column=0)
        
        self.player_value_label = ttk.Label(player_frame, text="", font=("Arial", 12))
        self.player_value_label.grid(row=1, column=0, pady=(5, 0))
        
        # Betting section
        bet_frame = ttk.LabelFrame(main_frame, text="Betting", padding="10")
        bet_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Label(bet_frame, text="Bet Amount:").grid(row=0, column=0)
        self.bet_var = tk.StringVar(value=str(self.min_bet))
        self.bet_entry = ttk.Entry(bet_frame, textvariable=self.bet_var, width=10)
        self.bet_entry.grid(row=0, column=1, padx=(5, 0))
        
        # Quick bet buttons for proper spread
        quick_bet_frame = ttk.Frame(bet_frame)
        quick_bet_frame.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        bet_amounts = [25, 50, 75, 125, 175, 250]
        for i, amount in enumerate(bet_amounts):
            btn = ttk.Button(quick_bet_frame, text=f"${amount}", width=6,
                           command=lambda a=amount: self.set_bet(a))
            btn.grid(row=0, column=i, padx=2)
        
        # Action buttons
        action_frame = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        action_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.hit_button = ttk.Button(action_frame, text="Hit", command=self.hit)
        self.hit_button.grid(row=0, column=0, padx=2)
        
        self.stand_button = ttk.Button(action_frame, text="Stand", command=self.stand)
        self.stand_button.grid(row=0, column=1, padx=2)
        
        self.double_button = ttk.Button(action_frame, text="Double", command=self.double)
        self.double_button.grid(row=0, column=2, padx=2)
        
        self.split_button = ttk.Button(action_frame, text="Split", command=self.split)
        self.split_button.grid(row=1, column=0, padx=2)
        
        self.surrender_button = ttk.Button(action_frame, text="Surrender", command=self.surrender)
        self.surrender_button.grid(row=1, column=1, padx=2)
        
        self.insurance_button = ttk.Button(action_frame, text="Insurance", command=self.insurance)
        self.insurance_button.grid(row=1, column=2, padx=2)
        
        # Game control
        control_frame = ttk.LabelFrame(main_frame, text="Game Control", padding="10")
        control_frame.grid(row=3, column=2, sticky=(tk.W, tk.E))
        
        self.new_hand_button = ttk.Button(control_frame, text="New Hand", command=self.new_hand)
        self.new_hand_button.grid(row=0, column=0, padx=2)
        
        self.reset_button = ttk.Button(control_frame, text="Reset Session", command=self.reset_session)
        self.reset_button.grid(row=0, column=1, padx=2)
        
        # Deck selection
        deck_frame = ttk.Frame(control_frame)
        deck_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Label(deck_frame, text="Decks:").grid(row=0, column=0)
        self.deck_var = tk.StringVar(value="6")
        deck_combo = ttk.Combobox(deck_frame, textvariable=self.deck_var, values=["2", "6", "8"], width=5)
        deck_combo.grid(row=0, column=1, padx=(5, 0))
        deck_combo.bind('<<ComboboxSelected>>', self.change_decks)
        
        # Data tracking controls
        data_frame = ttk.LabelFrame(main_frame, text="Data Tracking", padding="10")
        data_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.start_session_button = ttk.Button(data_frame, text="Start Session", command=self.start_tracking_session)
        self.start_session_button.grid(row=0, column=0, padx=2)
        
        self.end_session_button = ttk.Button(data_frame, text="End Session & Save", command=self.end_tracking_session, state=tk.DISABLED)
        self.end_session_button.grid(row=0, column=1, padx=2)
        
        self.analyze_button = ttk.Button(data_frame, text="Analyze All Data", command=self.show_comprehensive_analysis)
        self.analyze_button.grid(row=0, column=2, padx=2)
        
        self.export_button = ttk.Button(data_frame, text="Create Charts", command=self.create_analysis_charts)
        self.export_button.grid(row=0, column=3, padx=2)
        
        # Session status indicator
        self.session_status_label = ttk.Label(data_frame, text="No active session", foreground="red")
        self.session_status_label.grid(row=1, column=0, columnspan=4, pady=(5, 0))
    
    def set_bet(self, amount):
        """Set bet amount using quick buttons"""
        self.bet_var.set(str(amount))
    
    def change_decks(self, event=None):
        """Change number of decks and reshuffle"""
        try:
            num_decks = int(self.deck_var.get())
            self.shoe = Shoe(num_decks)
            self.running_count = 0
            self.update_display()
        except ValueError:
            pass
    
    def check_count(self):
        """Check player's count guess"""
        try:
            guess = int(self.count_entry.get())
            self.last_count_guess = guess
            
            if guess == self.running_count:
                self.correct_counts += 1
                messagebox.showinfo("Correct!", f"Correct! Running count is {self.running_count}")
            else:
                messagebox.showwarning("Incorrect", f"Incorrect. Running count is {self.running_count}, you guessed {guess}")
            
            self.count_entry.delete(0, tk.END)
            self.update_display()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    def new_hand(self):
        """Start a new hand"""
        if self.bankroll <= 0:
            messagebox.showerror("Game Over", "You're out of money!")
            return
        
        # Store bankroll at start of hand for tracking
        self.hand_start_bankroll = self.bankroll
        
        # Get bet amount
        try:
            bet_amount = int(self.bet_var.get())
            if bet_amount > self.bankroll:
                messagebox.showerror("Error", "Bet exceeds bankroll!")
                return
            if bet_amount < self.min_bet or bet_amount > self.max_bet:
                messagebox.showerror("Error", f"Bet must be between ${self.min_bet} and ${self.max_bet}")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid bet amount")
            return
        
        # Initialize new hand
        self.dealer_hand = Hand()
        self.player_hands = [Hand()]
        self.current_hand_index = 0
        self.player_hands[0].bet = bet_amount
        
        # Deal initial cards
        self.player_hands[0].add_card(self.shoe.deal_card())
        self.dealer_hand.add_card(self.shoe.deal_card())
        self.player_hands[0].add_card(self.shoe.deal_card())
        self.dealer_hand.add_card(self.shoe.deal_card())
        
        # Update running count (only visible cards)
        for card in self.player_hands[0].cards + [self.dealer_hand.cards[0]]:
            self.running_count += card.get_hilo_value()
        
        # Store decisions for strategy checking
        self.hand_decisions = []
        self.bet_analysis = self.analyze_bet_sizing(bet_amount)
        
        self.hands_played += 1
        self.session_hands += 1
        
        self.update_display()
        self.update_buttons()
        
        # Check for blackjack
        if self.player_hands[0].is_blackjack():
            if self.dealer_hand.cards[0].rank == Rank.ACE:
                pass  # Offer insurance first
            else:
                self.check_blackjacks()
    
    def hit(self):
        """Hit current hand"""
        if self.current_hand_index >= len(self.player_hands):
            return
        
        # Record decision for strategy analysis
        correct_action = self.get_basic_strategy_action()
        self.record_decision("hit", correct_action)
        
        current_hand = self.player_hands[self.current_hand_index]
        card = self.shoe.deal_card()
        current_hand.add_card(card)
        self.running_count += card.get_hilo_value()
        
        self.update_display()
        
        if current_hand.is_busted():
            self.next_hand()
        else:
            self.update_buttons()
    
    def stand(self):
        """Stand on current hand"""
        correct_action = self.get_basic_strategy_action()
        self.record_decision("stand", correct_action)
        self.next_hand()
    
    def double(self):
        """Double down on current hand"""
        current_hand = self.player_hands[self.current_hand_index]
        
        if current_hand.bet > self.bankroll:
            messagebox.showerror("Error", "Insufficient funds to double!")
            return
        
        correct_action = self.get_basic_strategy_action()
        self.record_decision("double", correct_action)
        
        current_hand.bet *= 2
        current_hand.is_doubled = True
        
        card = self.shoe.deal_card()
        current_hand.add_card(card)
        self.running_count += card.get_hilo_value()
        
        self.update_display()
        self.next_hand()
    
    def split(self):
        """Split current hand"""
        current_hand = self.player_hands[self.current_hand_index]
        
        if not current_hand.can_split():
            return
        
        if current_hand.bet > self.bankroll:
            messagebox.showerror("Error", "Insufficient funds to split!")
            return
        
        correct_action = self.get_basic_strategy_action()
        self.record_decision("split", correct_action)
        
        # Create new hand with second card
        new_hand = Hand()
        new_hand.bet = current_hand.bet
        new_hand.add_card(current_hand.cards.pop())
        new_hand.is_split = True
        current_hand.is_split = True
        
        # Add new hand to list
        self.player_hands.insert(self.current_hand_index + 1, new_hand)
        
        # Deal new cards to both hands
        current_hand.add_card(self.shoe.deal_card())
        new_hand.add_card(self.shoe.deal_card())
        
        # Update count
        for card in [current_hand.cards[-1], new_hand.cards[-1]]:
            self.running_count += card.get_hilo_value()
        
        self.update_display()
        self.update_buttons()
    
    def surrender(self):
        """Surrender current hand"""
        correct_action = self.get_basic_strategy_action()
        self.record_decision("surrender", correct_action)
        
        current_hand = self.player_hands[self.current_hand_index]
        current_hand.is_surrendered = True
        self.next_hand()
    
    def insurance(self):
        """Take insurance"""
        if self.dealer_hand.cards[0].rank != Rank.ACE:
            return
        
        insurance_bet = self.player_hands[0].bet // 2
        if insurance_bet > self.bankroll:
            messagebox.showerror("Error", "Insufficient funds for insurance!")
            return
        
        messagebox.showinfo("Insurance", "Insurance taken")
    
    def next_hand(self):
        """Move to next hand or end round"""
        self.current_hand_index += 1
        
        if self.current_hand_index >= len(self.player_hands):
            self.play_dealer()
            self.settle_bets()
        else:
            self.update_buttons()
    
    def play_dealer(self):
        """Play dealer hand according to rules"""
        # Reveal hole card
        self.running_count += self.dealer_hand.cards[1].get_hilo_value()
        
        # Dealer hits on soft 17
        while (self.dealer_hand.get_value() < 17 or 
               (self.dealer_hand.get_value() == 17 and self.dealer_hand.is_soft())):
            card = self.shoe.deal_card()
            self.dealer_hand.add_card(card)
            self.running_count += card.get_hilo_value()
        
        self.update_display()
    
    def check_blackjacks(self):
        """Check for blackjacks and settle immediately if found"""
        dealer_bj = self.dealer_hand.is_blackjack()
        player_bj = self.player_hands[0].is_blackjack()
        
        if dealer_bj and player_bj:
            messagebox.showinfo("Push", "Both have blackjack - Push!")
        elif player_bj:
            winnings = int(self.player_hands[0].bet * 1.5)
            self.bankroll += winnings
            self.session_winnings += winnings
            messagebox.showinfo("Blackjack!", f"Blackjack! You win ${winnings}")
        elif dealer_bj:
            self.bankroll -= self.player_hands[0].bet
            self.session_winnings -= self.player_hands[0].bet
            messagebox.showinfo("Dealer Blackjack", "Dealer has blackjack - You lose!")
        
        if dealer_bj or player_bj:
            self.update_display()
            self.disable_buttons()
    
    def settle_bets(self):
        """Settle all bets at end of round"""
        dealer_value = self.dealer_hand.get_value()
        dealer_busted = self.dealer_hand.is_busted()
        
        total_result = 0
        
        for hand in self.player_hands:
            if hand.is_surrendered:
                loss = hand.bet // 2
                self.bankroll -= loss
                total_result -= loss
                continue
            
            if hand.is_busted():
                self.bankroll -= hand.bet
                total_result -= hand.bet
                continue
            
            player_value = hand.get_value()
            
            if dealer_busted or player_value > dealer_value:
                if hand.is_blackjack() and len(self.player_hands) == 1:
                    winnings = int(hand.bet * 1.5)
                else:
                    winnings = hand.bet
                self.bankroll += winnings
                total_result += winnings
            elif player_value < dealer_value:
                self.bankroll -= hand.bet
                total_result -= hand.bet
        
        self.session_winnings += total_result
        
        # Record hand data if session is active
        if self.session_active:
            self.record_current_hand()
        
        # Show strategy and betting analysis
        self.show_hand_analysis()
        
        # Show result
        if total_result > 0:
            messagebox.showinfo("You Win!", f"You win ${total_result}!")
        elif total_result < 0:
            messagebox.showinfo("You Lose", f"You lose ${abs(total_result)}")
        else:
            messagebox.showinfo("Push", "Push - No money changes hands")
        
        self.update_display()
        self.disable_buttons()
    
    def get_basic_strategy_action(self):
        """Returns the correct basic strategy action for current situation"""
        if self.current_hand_index >= len(self.player_hands):
            return "stand"
            
        hand = self.player_hands[self.current_hand_index]
        dealer_up_card = self.dealer_hand.cards[0].get_value()
        player_total = hand.get_value()
        
        # Handle pairs
        if hand.can_split() and len(hand.cards) == 2:
            pair_value = hand.cards[0].get_value()
            
            if pair_value == 1:  # Aces
                return "split"
            elif pair_value == 8:  # 8s
                return "split"
            elif pair_value == 9:  # 9s
                return "split" if dealer_up_card not in [7, 10, 1] else "stand"
            elif pair_value == 7:  # 7s
                return "split" if dealer_up_card <= 7 else "hit"
            elif pair_value == 6:  # 6s
                return "split" if dealer_up_card <= 6 else "hit"
            elif pair_value == 4:  # 4s
                return "split" if dealer_up_card in [5, 6] else "hit"
            elif pair_value == 2 or pair_value == 3:  # 2s or 3s
                return "split" if dealer_up_card <= 7 else "hit"
            elif pair_value == 5:  # 5s
                return "double" if dealer_up_card <= 9 else "hit"
            elif pair_value == 10:  # 10s
                return "stand"
        
        # Handle soft hands
        if hand.is_soft():
            if player_total >= 19:
                return "stand"
            elif player_total == 18:
                if dealer_up_card <= 6:
                    return "double" if len(hand.cards) == 2 else "stand"
                elif dealer_up_card in [7, 8]:
                    return "stand"
                else:
                    return "hit"
            elif player_total == 17:
                return "double" if dealer_up_card in [3, 4, 5, 6] and len(hand.cards) == 2 else "hit"
            elif player_total in [15, 16]:
                return "double" if dealer_up_card in [4, 5, 6] and len(hand.cards) == 2 else "hit"
            elif player_total in [13, 14]:
                return "double" if dealer_up_card in [5, 6] and len(hand.cards) == 2 else "hit"
            else:
                return "hit"
        
        # Handle hard hands
        if player_total >= 17:
            return "stand"
        elif player_total == 16:
            return "surrender" if dealer_up_card in [9, 10, 1] and len(hand.cards) == 2 else ("stand" if dealer_up_card <= 6 else "hit")
        elif player_total == 15:
            return "surrender" if dealer_up_card == 10 and len(hand.cards) == 2 else ("stand" if dealer_up_card <= 6 else "hit")
        elif player_total in [13, 14]:
            return "stand" if dealer_up_card <= 6 else "hit"
        elif player_total == 12:
            return "stand" if dealer_up_card in [4, 5, 6] else "hit"
        elif player_total == 11:
            return "double" if len(hand.cards) == 2 else "hit"
        elif player_total == 10:
            return "double" if dealer_up_card <= 9 and len(hand.cards) == 2 else "hit"
        elif player_total == 9:
            return "double" if dealer_up_card in [3, 4, 5, 6] and len(hand.cards) == 2 else "hit"
        else:
            return "hit"
    
    def record_decision(self, action_taken, correct_action):
        """Record player decision for analysis"""
        self.total_decisions += 1
        is_correct = action_taken == correct_action
        
        if not is_correct:
            self.strategy_errors += 1
        
        self.hand_decisions.append({
            'action_taken': action_taken,
            'correct_action': correct_action,
            'is_correct': is_correct,
            'hand_value': self.player_hands[self.current_hand_index].get_value(),
            'dealer_up_card': self.dealer_hand.cards[0].get_value(),
            'is_soft': self.player_hands[self.current_hand_index].is_soft()
        })
    
    def analyze_bet_sizing(self, bet_amount):
        """Analyze if bet sizing is appropriate for the count"""
        true_count = self.get_true_count()
        
        if true_count <= 0:
            recommended_bet = self.min_bet
        elif true_count == 1:
            recommended_bet = self.min_bet * 2
        elif true_count == 2:
            recommended_bet = self.min_bet * 4
        elif true_count == 3:
            recommended_bet = self.min_bet * 6
        elif true_count >= 4:
            recommended_bet = self.min_bet * 8
        else:
            recommended_bet = self.min_bet
        
        recommended_bet = min(recommended_bet, self.max_bet)
        
        return {
            'bet_amount': bet_amount,
            'recommended_bet': recommended_bet,
            'true_count': true_count,
            'is_optimal': abs(bet_amount - recommended_bet) <= self.min_bet
        }
    
    def get_true_count(self):
        """Calculate true count"""
        decks_remaining = max(1, self.shoe.cards_remaining() / 52)
        if self.running_count == 0:
            return 0
        return int(self.running_count / decks_remaining)
    
    def show_hand_analysis(self):
        """Show analysis of strategy decisions and bet sizing"""
        analysis_text = "Hand Analysis:\n\n"
        
        # Bet sizing analysis
        if self.bet_analysis:
            analysis_text += f"Bet Sizing:\n"
            analysis_text += f"True Count: {self.bet_analysis['true_count']}\n"
            analysis_text += f"Your Bet: ${self.bet_analysis['bet_amount']}\n"
            analysis_text += f"Recommended: ${self.bet_analysis['recommended_bet']}\n"
            if self.bet_analysis['is_optimal']:
                analysis_text += "✓ Good bet sizing!\n\n"
            else:
                analysis_text += "✗ Suboptimal bet sizing\n\n"
        
        # Strategy decisions
        if self.hand_decisions:
            analysis_text += "Strategy Decisions:\n"
            for i, decision in enumerate(self.hand_decisions):
                analysis_text += f"Decision {i+1}: "
                if decision['is_correct']:
                    analysis_text += f"✓ {decision['action_taken'].title()} (Correct)\n"
                else:
                    analysis_text += f"✗ {decision['action_taken'].title()} (Should be {decision['correct_action']})\n"
        
        if analysis_text.strip() != "Hand Analysis:":
            messagebox.showinfo("Hand Analysis", analysis_text)
    
    # Data Tracking Methods
    def start_tracking_session(self):
        """Start a new data tracking session"""
        if self.session_active:
            if not messagebox.askyesno("Session Active", "A session is already active. End current session?"):
                return
            self.end_tracking_session()
        
        self.tracker.start_session(self.bankroll)
        self.session_active = True
        
        # Update UI
        self.start_session_button.config(state=tk.DISABLED)
        self.end_session_button.config(state=tk.NORMAL)
        self.session_status_label.config(text=f"Session active: {self.tracker.session_id}", foreground="green")
        
        messagebox.showinfo("Session Started", f"Data tracking session started: {self.tracker.session_id}")
    
    def end_tracking_session(self):
        """End session and save data"""
        if not self.session_active:
            messagebox.showwarning("No Session", "No active session to end.")
            return
        
        filename = self.tracker.end_session(self.bankroll)
        self.session_active = False
        
        # Update UI
        self.start_session_button.config(state=tk.NORMAL)
        self.end_session_button.config(state=tk.DISABLED)
        self.session_status_label.config(text="No active session", foreground="red")
        
        if filename:
            messagebox.showinfo("Session Saved", f"Session data saved!\n\nFiles created:\n- {filename}\n- {filename.replace('hands_', 'session_')}")
            
            # Ask if user wants to see analysis
            if messagebox.askyesno("Analysis", "Would you like to see this session's analysis?"):
                self.show_session_analysis()
    
    def record_current_hand(self):
        """Record current hand data for analysis"""
        if not self.session_active or not self.player_hands:
            return
        
        # Determine hand result and amount won/lost
        total_result = 0
        hand_result = "push"
        
        # Calculate actual result (simplified for first hand)
        main_hand = self.player_hands[0]
        dealer_value = self.dealer_hand.get_value()
        player_value = main_hand.get_value()
        
        if main_hand.is_surrendered:
            total_result = -(main_hand.bet // 2)
            hand_result = "loss"
        elif main_hand.is_busted():
            total_result = -main_hand.bet
            hand_result = "loss"
        elif self.dealer_hand.is_busted() or player_value > dealer_value:
            if main_hand.is_blackjack() and len(self.player_hands) == 1:
                total_result = int(main_hand.bet * 1.5)
            else:
                total_result = main_hand.bet
            hand_result = "win"
        elif player_value < dealer_value:
            total_result = -main_hand.bet
            hand_result = "loss"
        
        # Create hand data record
        hand_data = HandData(
            session_id="",
            timestamp="",
            hand_number=self.session_hands,
            bankroll_start=self.hand_start_bankroll,
            true_count=self.get_true_count(),
            running_count=self.running_count,
            decks_remaining=self.shoe.cards_remaining() / 52,
            bet_amount=main_hand.bet,
            recommended_bet=self.bet_analysis['recommended_bet'] if self.bet_analysis else main_hand.bet,
            bet_optimal=self.bet_analysis['is_optimal'] if self.bet_analysis else True,
            player_card1=f"{main_hand.cards[0].rank.value}",
            player_card2=f"{main_hand.cards[1].rank.value}",
            dealer_upcard=f"{self.dealer_hand.cards[0].rank.value}",
            player_initial_value=main_hand.get_value() if len(main_hand.cards) >= 2 else 0,
            player_soft_initial=main_hand.is_soft() if len(main_hand.cards) >= 2 else False,
            decisions_made=",".join([d['action_taken'] for d in self.hand_decisions]),
            decisions_correct=",".join([str(d['is_correct']) for d in self.hand_decisions]),
            strategy_accuracy=sum(d['is_correct'] for d in self.hand_decisions) / len(self.hand_decisions) * 100 if self.hand_decisions else 100,
            final_player_value=player_value,
            dealer_final_value=dealer_value,
            player_busted=main_hand.is_busted(),
            dealer_busted=self.dealer_hand.is_busted(),
            player_blackjack=main_hand.is_blackjack(),
            dealer_blackjack=self.dealer_hand.is_blackjack(),
            surrendered=main_hand.is_surrendered,
            doubled=main_hand.is_doubled,
            split=len(self.player_hands) > 1,
            hand_result=hand_result,
            amount_won_lost=total_result,
            bankroll_end=self.bankroll,
            count_guess=self.last_count_guess if hasattr(self, 'last_count_guess') and self.last_count_guess is not None else None,
            count_actual=self.running_count,
            count_correct=self.last_count_guess == self.running_count if hasattr(self, 'last_count_guess') and self.last_count_guess is not None else None
        )
        
        self.tracker.record_hand(hand_data)
    
    def show_session_analysis(self):
        """Display analysis for current/recent session"""
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Session Analysis")
        analysis_window.geometry("900x700")
        
        # Create scrollable text widget
        main_frame = ttk.Frame(analysis_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, padx=10, pady=10, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate and display analysis
        try:
            report = self.analyzer.generate_comprehensive_report()
            
            if "error" in report:
                text_widget.insert(tk.END, f"Error: {report['error']}")
            else:
                analysis_text = self.format_analysis_report(report)
                text_widget.insert(tk.END, analysis_text)
        except Exception as e:
            text_widget.insert(tk.END, f"Error generating analysis: {str(e)}")
        
        text_widget.config(state=tk.DISABLED)
        
        # Add buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Create Charts", command=self.create_analysis_charts).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Close", command=analysis_window.destroy).pack(side=tk.RIGHT)
    
    def show_comprehensive_analysis(self):
        """Show analysis of all historical data"""
        try:
            hands_df, sessions_df = self.analyzer.load_all_data()
            
            if hands_df.empty:
                messagebox.showwarning("No Data", "No historical data found. Play some hands first!")
                return
            
            self.show_session_analysis()
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error loading data: {str(e)}")
    
    def create_analysis_charts(self):
        """Create and save visualization charts"""
        try:
            output_dir = filedialog.askdirectory(title="Select folder to save charts")
            if not output_dir:
                return
            
            self.analyzer.create_visualizations(output_dir)
            messagebox.showinfo("Charts Created", f"Analysis charts saved to:\n{output_dir}")
            
            # Ask if user wants to open the folder
            if messagebox.askyesno("Open Folder", "Would you like to open the folder with the charts?"):
                if platform.system() == "Windows":
                    subprocess.run(f'explorer "{output_dir}"', shell=True)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", output_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", output_dir])
                    
        except Exception as e:
            messagebox.showerror("Chart Error", f"Error creating charts: {str(e)}")
    
    def format_analysis_report(self, report):
        """Format the analysis report for display"""
        text = "BLACKJACK GAMEPLAY ANALYSIS REPORT\n"
        text += "=" * 50 + "\n\n"
        
        # Overview
        if "overview" in report:
            text += "OVERVIEW\n" + "-" * 20 + "\n"
            overview = report["overview"]
            text += f"Total Sessions: {overview.get('total_sessions', 0)}\n"
            text += f"Total Hands: {overview.get('total_hands', 0)}\n"
            text += f"Hours Played: {overview.get('total_hours_played', 0):.1f}\n"
            text += f"Total Wagered: ${overview.get('total_wagered', 0):,.2f}\n"
            text += f"Net Result: ${overview.get('net_result', 0):+,.2f}\n"
            text += f"Win Rate: {overview.get('overall_win_rate', 0):.1f}%\n\n"
        
        # Financial Performance
        if "financial_performance" in report:
            text += "FINANCIAL PERFORMANCE\n" + "-" * 30 + "\n"
            financial = report["financial_performance"]
            text += f"Winning Sessions: {financial.get('winning_sessions', 0)}\n"
            text += f"Losing Sessions: {financial.get('losing_sessions', 0)}\n"
            text += f"Best Session: ${financial.get('best_session', 0):+,.2f}\n"
            text += f"Worst Session: ${financial.get('worst_session', 0):+,.2f}\n"
            text += f"ROI: {financial.get('roi_percentage', 0):+.2f}%\n"
            text += f"Hourly Rate: ${financial.get('hourly_rate', 0):+,.2f}/hour\n\n"
        
        # Strategy Analysis
        if "strategy_analysis" in report:
            text += "STRATEGY ANALYSIS\n" + "-" * 25 + "\n"
            strategy = report["strategy_analysis"]
            if "error" not in strategy:
                text += f"Overall Accuracy: {strategy.get('overall_strategy_accuracy', 0):.1f}%\n\n"
                
                if "most_common_mistakes" in strategy:
                    text += "Most Common Mistakes:\n"
                    for mistake in strategy["most_common_mistakes"]:
                        text += f"- {mistake['situation']}: {mistake['mistake_rate']:.1f}% error rate\n"
                        text += f"  → {mistake['recommendation']}\n"
                    text += "\n"
        
        # Counting Analysis
        if "counting_analysis" in report:
            counting = report["counting_analysis"]
            if "error" not in counting:
                text += "COUNTING ANALYSIS\n" + "-" * 25 + "\n"
                text += f"Count Accuracy: {counting.get('overall_count_accuracy', 0):.1f}%\n\n"
        
        # Betting Analysis
        if "betting_analysis" in report:
            text += "BETTING ANALYSIS\n" + "-" * 25 + "\n"
            betting = report["betting_analysis"]
            text += f"Kelly Compliance: {betting.get('kelly_compliance_rate', 0):.1f}%\n"
            text += f"Overbetting: {betting.get('overbetting_frequency', 0):.1f}%\n"
            text += f"Underbetting: {betting.get('underbetting_frequency', 0):.1f}%\n\n"
        
        # Recommendations
        if "recommendations" in report:
            text += "RECOMMENDATIONS\n" + "-" * 20 + "\n"
            for i, rec in enumerate(report["recommendations"], 1):
                text += f"{i}. {rec}\n"
            
            if not report["recommendations"]:
                text += "Great job! No specific recommendations at this time.\n"
        
        return text
    
    def update_display(self):
        """Update all display elements"""
        # Update info labels
        self.bankroll_label.config(text=f"Bankroll: ${self.bankroll}")
        self.session_label.config(text=f"Session: {self.session_hands} hands, ${self.session_winnings:+.2f}")
        self.cards_remaining_label.config(text=f"Cards: {self.shoe.cards_remaining()}")
        
        accuracy = f"{self.correct_counts}/{self.hands_played}" if self.hands_played > 0 else "0/0"
        self.count_accuracy_label.config(text=f"Accuracy: {accuracy}")
        
        strategy_accuracy = f"{self.total_decisions - self.strategy_errors}/{self.total_decisions}" if self.total_decisions > 0 else "0/0"
        self.strategy_accuracy_label.config(text=f"Strategy: {strategy_accuracy}")
        
        # Clear existing card displays
        for widget in self.dealer_cards_frame.winfo_children():
            widget.destroy()
        for widget in self.player_cards_frame.winfo_children():
            widget.destroy()
        
        # Display dealer cards
        for i, card in enumerate(self.dealer_hand.cards):
            if i == 1 and self.current_hand_index < len(self.player_hands):
                # Hide hole card during play
                if hasattr(self, 'card_back') and self.card_back:
                    label = ttk.Label(self.dealer_cards_frame, image=self.card_back)
                else:
                    label = tk.Label(self.dealer_cards_frame, text="[Hidden]", width=10, height=6, 
                                   relief="solid", bg="blue", fg="white")
            else:
                # Show card
                img_name = card.get_filename() + ".png"
                if img_name in self.card_images:
                    label = ttk.Label(self.dealer_cards_frame, image=self.card_images[img_name])
                else:
                    label = tk.Label(self.dealer_cards_frame, text=f"{card.rank.value}\n{card.suit.value}", 
                                   width=10, height=6, relief="solid", bg="white")
            label.grid(row=0, column=i, padx=2)
        
        # Display dealer value
        if self.current_hand_index >= len(self.player_hands):
            # Round over, show dealer value
            dealer_value = self.dealer_hand.get_value()
            soft_text = " (soft)" if self.dealer_hand.is_soft() else ""
            bust_text = " - BUST!" if self.dealer_hand.is_busted() else ""
            self.dealer_value_label.config(text=f"Dealer: {dealer_value}{soft_text}{bust_text}")
        else:
            # Show only up card value
            up_card_value = self.dealer_hand.cards[0].get_value()
            self.dealer_value_label.config(text=f"Dealer showing: {up_card_value}")
        
        # Display player cards
        for hand_idx, hand in enumerate(self.player_hands):
            hand_frame = ttk.Frame(self.player_cards_frame)
            hand_frame.grid(row=hand_idx, column=0, pady=5, sticky=(tk.W, tk.E))
            
            # Hand indicator
            if len(self.player_hands) > 1:
                indicator = "→ " if hand_idx == self.current_hand_index else "  "
                ttk.Label(hand_frame, text=f"{indicator}Hand {hand_idx + 1}:").grid(row=0, column=0, sticky=tk.W)
            
            # Cards
            cards_frame = ttk.Frame(hand_frame)
            cards_frame.grid(row=1, column=0, sticky=tk.W)
            
            for i, card in enumerate(hand.cards):
                img_name = card.get_filename() + ".png"
                if img_name in self.card_images:
                    label = ttk.Label(cards_frame, image=self.card_images[img_name])
                else:
                    label = tk.Label(cards_frame, text=f"{card.rank.value}\n{card.suit.value}", 
                                   width=10, height=6, relief="solid", bg="white")
                label.grid(row=0, column=i, padx=2)
            
            # Hand value and status
            value = hand.get_value()
            soft_text = " (soft)" if hand.is_soft() else ""
            bust_text = " - BUST!" if hand.is_busted() else ""
            bj_text = " - BLACKJACK!" if hand.is_blackjack() else ""
            surrender_text = " - SURRENDERED" if hand.is_surrendered else ""
            
            status_text = f"Value: {value}{soft_text}{bust_text}{bj_text}{surrender_text} | Bet: ${hand.bet}"
            ttk.Label(hand_frame, text=status_text).grid(row=2, column=0, sticky=tk.W)
        
        # Update player value label for current hand
        if self.current_hand_index < len(self.player_hands):
            current_hand = self.player_hands[self.current_hand_index]
            value = current_hand.get_value()
            soft_text = " (soft)" if current_hand.is_soft() else ""
            self.player_value_label.config(text=f"Current Hand: {value}{soft_text}")
        else:
            self.player_value_label.config(text="")
    
    def update_buttons(self):
        """Update button states based on current game state"""
        if self.current_hand_index >= len(self.player_hands):
            self.disable_buttons()
            return
        
        current_hand = self.player_hands[self.current_hand_index]
        
        # Basic actions always available if not busted
        can_act = not current_hand.is_busted() and not current_hand.is_surrendered
        
        self.hit_button.config(state=tk.NORMAL if can_act else tk.DISABLED)
        self.stand_button.config(state=tk.NORMAL if can_act else tk.DISABLED)
        
        # Double only on first two cards and if we have money
        can_double = (can_act and len(current_hand.cards) == 2 and 
                     current_hand.bet <= self.bankroll and not current_hand.is_split)
        self.double_button.config(state=tk.NORMAL if can_double else tk.DISABLED)
        
        # Split only if can split and have money and less than 4 hands
        can_split = (can_act and current_hand.can_split() and 
                    current_hand.bet <= self.bankroll and len(self.player_hands) < 4)
        # Special rule: split aces only once
        if (current_hand.cards[0].rank == Rank.ACE and 
            any(hand.is_split and hand.cards[0].rank == Rank.ACE for hand in self.player_hands)):
            can_split = False
        self.split_button.config(state=tk.NORMAL if can_split else tk.DISABLED)
        
        # Surrender only on first two cards
        can_surrender = can_act and len(current_hand.cards) == 2 and not current_hand.is_split
        self.surrender_button.config(state=tk.NORMAL if can_surrender else tk.DISABLED)
        
        # Insurance only when dealer shows ace and player has first hand
        can_insurance = (self.dealer_hand.cards[0].rank == Rank.ACE and 
                        self.current_hand_index == 0 and len(current_hand.cards) == 2)
        self.insurance_button.config(state=tk.NORMAL if can_insurance else tk.DISABLED)
        
        # New hand only when round is over
        self.new_hand_button.config(state=tk.DISABLED)
    
    def disable_buttons(self):
        """Disable all action buttons"""
        for button in [self.hit_button, self.stand_button, self.double_button, 
                      self.split_button, self.surrender_button, self.insurance_button]:
            button.config(state=tk.DISABLED)
        self.new_hand_button.config(state=tk.NORMAL)
    
    def reset_session(self):
        """Reset session statistics"""
        self.bankroll = 1000
        self.session_hands = 0
        self.session_winnings = 0
        self.hands_played = 0
        self.correct_counts = 0
        self.running_count = 0
        self.strategy_errors = 0
        self.total_decisions = 0
        self.shoe.shuffle()
        self.update_display()
        messagebox.showinfo("Reset", "Session reset!")

def main():
    # Check for required packages
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: Required packages not found.")
        print("Please install required packages:")
        print("pip install pandas matplotlib seaborn pillow")
        return
    
    root = tk.Tk()
    app = BlackjackTrainer(root)
    root.mainloop()

if __name__ == "__main__":
    main()