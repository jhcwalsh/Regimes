"""
Central configuration for the Regimes project.
Replication of: Mulliner, Harvey, Xia, Fang & Van Hemert (JPM, Feb 2026)
"""

# ---------------------------------------------------------------------------
# FRED API
# ---------------------------------------------------------------------------
# Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
# Set via environment variable or replace the empty string below.
import os

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ---------------------------------------------------------------------------
# FRED series IDs for the seven state variables
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "yield_10yr":  "GS10",        # 10-Year Treasury Constant Maturity Rate
    "tbill_3m":    "TB3MS",       # 3-Month Treasury Bill Secondary Market Rate
    "oil":         "DCOILWTICO",  # WTI Crude Oil Price ($/barrel)
    "copper":      "PCOPPUSDM",   # Copper Price (USD/metric ton, monthly)
    "vix":         "VIXCLS",      # CBOE VIX (from 1990)
}

# S&P 500 and daily equity/bond data pulled via yfinance
SP500_TICKER   = "^GSPC"
BOND_TICKER    = "^TNX"          # 10-yr yield proxy for daily correlation calc

# ---------------------------------------------------------------------------
# Transformation parameters (paper Section: Economic State Variables)
# ---------------------------------------------------------------------------
DIFF_MONTHS        = 12          # 12-month (annual) difference
ZSCORE_WINDOW_YRS  = 10          # Rolling window for Z-score denominator (years)
WINSOR_LIMIT       = 3.0         # Winsorize to ±3
CORR_LOOKBACK_YRS  = 3           # Rolling stock-bond correlation window (years)

# ---------------------------------------------------------------------------
# Similarity / regime selection parameters
# ---------------------------------------------------------------------------
EXCLUDE_RECENT_MONTHS = 36       # Exclude last N months to avoid momentum loading
QUANTILE_SIMILAR      = 0.20     # Top fraction = most similar (Q1, paper uses 15-20%)
QUANTILE_DISSIMILAR   = 0.20     # Bottom fraction = most dissimilar (Q5)
N_DISPLAY_SIMILAR     = 20       # Number of similar periods to display

# ---------------------------------------------------------------------------
# Regime shift EWMA parameters (paper Exhibit 9)
# ---------------------------------------------------------------------------
EWMA_LOOKBACK_YEARS = [1, 2, 3, 4]  # Four lookback windows in years

# ---------------------------------------------------------------------------
# Portfolio construction parameters
# ---------------------------------------------------------------------------
RISK_AVERSION          = 2.5     # Black-Litterman risk aversion coefficient (delta)
BL_TAU                 = 0.05    # BL uncertainty scaling parameter (tau)
REBALANCE_FREQ_MONTHS  = 1       # Rebalance frequency (1 = monthly)
MAX_STRATEGY_WEIGHT    = 0.40    # Maximum weight per strategy
MIN_STRATEGY_WEIGHT    = 0.00    # Minimum weight (no short selling)
HF_HORIZONS            = [1, 3, 6, 12]   # Forward return horizons for regime-conditional stats
UNSMOOTH_RETURNS       = True    # Apply Geltner (1994) unsmoothing to HF returns
BACKTEST_START         = "1994-01-31"    # Default backtest start (matches HFRI history)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
