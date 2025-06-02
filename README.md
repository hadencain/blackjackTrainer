# Blackjack Strategy Trainer with Analytics

An application for practice with blackjack basic strategy, card counting, and analyzing gameplay performance through detailed data tracking and visualization.

## Features

###  Core Gameplay
- **Blackjack Rules**: Full implementation of standard casino blackjack rules
- **Multi-deck Support**: Play with 2, 6, or 8 deck shoes
- **Complete Action Set**: Hit, stand, double down, split, surrender, and insurance
- **Multiple Hand Support**: Handle split hands with proper game flow

###  Strategy Training
- **Real-time Strategy Analysis**: Get instant feedback on basic strategy decisions
- **Decision Tracking**: Monitor accuracy across all gameplay decisions
- **Common Mistake Identification**: Learn from the most frequent strategy errors
- **Optimal Play Recommendations**: Receive guidance for improvement

### Card Counting
- **Hi-Lo Count System**: Practice the most popular card counting method
- **Running Count Tracking**: Test your counting accuracy throughout sessions
- **True Count Calculations**: Automatic conversion based on remaining decks
- **Count-based Bet Sizing**: Learn proper betting strategies for different counts

### Analytics
- **Session Data Recording**: Comprehensive tracking of every hand played
- **Performance Metrics**: Win rates, strategy accuracy, counting precision
- **Financial Analysis**: ROI, hourly rates, bankroll management insights
- **Visual Reports**: Charts and graphs showing performance trends
- **Historical Data**: Long-term analysis across multiple sessions


## Installation

### Prerequisites
- Python 3.7 or higher
- Required Python packages (install via pip):

```bash
pip install pandas matplotlib seaborn pillow numpy
```

### Setup
1. **Clone or download** the repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Optional: Add card images**
   - Create a `card_images` folder in the application directory
   - Add PNG files for each card named as: `[RANK]_of_[SUIT].png`
   - Include `card_back.png` for face-down cards
   - Example: `A_of_Spades.png`, `10_of_Hearts.png`

### Running the Application
```bash/CMD
python blackjack2.py
```

## Usage Guide

### Starting a Session
1. **Launch the application**
2. **Set your preferred number of decks** (2, 6, or 8)
3. **Click "Start Session"** to begin data tracking
4. **Place your bet** using the amount field or quick-bet buttons
5. **Click "New Hand"** to deal cards

### Playing Hands
- **Make decisions** using the action buttons (Hit, Stand, Double, etc.)
- **Practice counting** by entering your running count guess
- **Review feedback** after each hand showing strategy and betting analysis
- **Continue playing** to build your session data

### Analyzing Performance
- **End your session** to save data and view session summary
- **Click "Analyze All Data"** to see comprehensive historical analysis
- **Create charts** to visualize your performance trends
- **Review recommendations** for areas of improvement

### Data Management
- Session data is automatically saved to CSV files in the `blackjack_data` folder
- Files include detailed hand-by-hand data and session summaries
- Charts and visualizations can be exported to any folder

## Strategy Implementation

The application implements mathematically optimal basic strategy based on:
- **Player hand value** (hard totals, soft totals, pairs)
- **Dealer upcard**
- **Available actions** (considering doubling/splitting restrictions)

### Default Rule Set
This trainer uses a standard set of blackjack rules commonly found in many casinos:
- **Dealer hits soft 17**
- **Late surrender allowed**
- **Double down on any two cards**
- **Double after split allowed**
- **Split up to 4 hands (3 additional splits)**
- **Split aces receive one card only**
- **No re-splitting of aces**
- **Blackjack pays 3:2**
- **Insurance available when dealer shows ace**

### Important Note: Casino Rule Variations

**Real casinos implement significantly different rule variations that can substantially impact optimal strategy and house edge.** This trainer uses one common rule set, but players should be aware that casino conditions vary widely:

#### Common Rule Variations You May Encounter:

**Dealer Rules:**
- Dealer **stands** on soft 17 (better for player, -0.2% house edge)
- Dealer **hits** on soft 17 (current default, standard rule)

**Surrender Options:**
- **No surrender** allowed (increases house edge by ~0.1%)
- **Early surrender** (rare, very favorable to player)
- **Late surrender only** (current default)

**Doubling Restrictions:**
- Double on **10 and 11 only** (worse for player, +0.3% house edge)
- Double on **any two cards** (current default)
- **No doubling after split** (+0.1% house edge)

**Splitting Rules:**
- **No re-splitting** allowed (slight house edge increase)
- **Re-split aces allowed** (rare, favorable to player)
- Different **maximum split hands** (2, 3, or 4 total hands)

**Blackjack Payouts:**
- **6:5 blackjack** (terrible for player, +1.4% house edge!)
- **3:2 blackjack** (current default, standard)
- **2:1 blackjack** (rare promotion, very favorable)

**Other Variations:**
- **European No-Hole-Card** (dealer doesn't check for blackjack)
- **Peek/No-Peek** rules for dealer blackjack
- **Insurance side bet** availability
- **Charlie rules** (automatic win on 5+ cards)

#### Impact on Strategy and House Edge

These rule variations can significantly affect:
- **Basic strategy decisions** (especially with dealer soft 17 variations)
- **House edge** (ranging from ~0.3% to over 2% depending on rules)
- **Optimal betting strategies** for card counters
- **Expected value** of various plays



### Betting Strategy
Bet sizing recommendations follow the Kelly Criterion with Hi-Lo count correlation:
- **True Count ≤ 0**: Minimum bet
- **True Count = 1**: 2x minimum bet
- **True Count = 2**: 4x minimum bet
- **True Count = 3**: 6x minimum bet
- **True Count ≥ 4**: 8x minimum bet (capped at table maximum)

*Note: Betting strategies should also be adjusted based on specific rule variations and penetration levels.*

## Data Analysis Features

### Hand-Level Metrics
- Initial cards and dealer upcard
- All decisions made and their correctness
- Final hand values and outcomes
- Betting analysis and count accuracy
- Financial results

### Session-Level Analytics
- Win/loss rates and financial performance
- Strategy accuracy across all decisions
- Card counting precision
- Risk of ruin calculations
- Kelly Criterion compliance

### Comprehensive Reports
- **Financial Performance**: ROI, hourly rates, session profitability
- **Strategy Analysis**: Decision accuracy, common mistakes
- **Counting Performance**: Count accuracy by true count levels
- **Betting Patterns**: Optimal sizing compliance, over/under betting
- **Personalized Recommendations**: Targeted improvement suggestions

## File Structure

```
blackjack-trainer/
├── blackjack2.py              # Main application file
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── card_images/              # Optional card image folder
│   ├── A_of_Spades.png
│   ├── card_back.png
│   └── ...
└── blackjack_data/          # Generated data folder
    ├── hands_YYYYMMDD_HHMMSS.csv
    ├── session_YYYYMMDD_HHMMSS.csv
    └── ...
```

## Technical Details

### Built With
- **Python 3.7+**: Core application language
- **tkinter**: GUI framework
- **pandas**: Data manipulation and analysis
- **matplotlib/seaborn**: Data visualization
- **PIL (Pillow)**: Image processing for card display
- **numpy**: Numerical computations

### Key Classes
- **`Card`**: Individual playing card with Hi-Lo values
- **`Hand`**: Player/dealer hand with game logic
- **`Shoe`**: Multi-deck card shoe with shuffling
- **`GameplayTracker`**: Session data recording
- **`BlackjackAnalyzer`**: Historical data analysis
- **`BlackjackTrainer`**: Main GUI application

### Data Export Format
All data is saved in CSV format for easy analysis in external tools:
- Compatible with Excel, R, Python pandas
- Detailed schema documentation included in data files
- Time-stamped sessions for historical tracking


## Troubleshooting

### Common Issues

**Card images not displaying:**
- Ensure `card_images` folder exists
- Verify PNG files are named correctly
- Check file permissions

**Analysis errors:**
- Confirm all required packages are installed
- Verify CSV files aren't corrupted
- Check available disk space for data files

**Performance issues:**
- Large datasets may slow analysis
- Consider archiving old session data
- Ensure adequate system memory

### Dependencies Installation Issues
```bash
# If pip install fails, try:
python -m pip install --upgrade pip
pip install --user pandas matplotlib seaborn pillow numpy

# For conda users:
conda install pandas matplotlib seaborn pillow numpy
```

---
