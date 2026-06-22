"""
G20 Major Market Indices + Gold, Silver, REITs — Normalized Chart
==================================================================
Requirements:
  pip install yfinance matplotlib pandas numpy

Usage:
  python g20_market_chart.py

Customization:
  - BASE_VALUE : normalization base (default 100)
  - START_DATE / END_DATE : override date range (None = auto)
  - The script auto-detects the shortest available series and sets
    the base date to the first common trading day across all tickers.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import yfinance as yf
from datetime import date, timedelta

# ══════════════════════════════════════════════════
#  USER SETTINGS  — edit these freely
# ══════════════════════════════════════════════════
BASE_VALUE  = 100          # Normalisation base (e.g. 100, 1000, 10000)
START_DATE  = "2024-01-01" # Fetch start — set None to use 2 years ago
END_DATE    = None         # Fetch end   — set None to use today
JPY_BASE    = True         # True = convert all prices to JPY
# ══════════════════════════════════════════════════

# ── Date range ────────────────────────────────────
today = date.today()
if END_DATE is None:
    END_DATE = today.strftime("%Y-%m-%d")
if START_DATE is None:
    START_DATE = (today - timedelta(days=365*2 + 60)).strftime("%Y-%m-%d")

# ── Tickers ───────────────────────────────────────
# Format: "Display Label": ("ticker", "currency_of_ticker")
TICKERS = {
    # Equities — G20 + key markets
    "Nikkei 225 (JP)":        ("^N225",       "JPY"),
    "S&P 500 (US)":           ("^GSPC",       "USD"),
    "DAX (DE)":               ("^GDAXI",      "EUR"),
    "FTSE 100 (UK)":          ("^FTSE",       "GBP"),
    "CAC 40 (FR)":            ("^FCHI",       "EUR"),
    "Shanghai Comp. (CN)":    ("000001.SS",   "CNY"),
    "SENSEX (IN)":            ("^BSESN",      "INR"),
    "KOSPI (KR)":             ("^KS11",       "KRW"),
    "IBOVESPA (BR)":          ("^BVSP",       "BRL"),
    "ASX 200 (AU)":           ("^AXJO",       "AUD"),
    "TSX (CA)":               ("^GSPTSE",     "CAD"),
    "FTSE MIB (IT)":          ("FTSEMIB.MI",  "EUR"),
    # Commodities
    "Gold (JPY/g)":           ("GC=F",        "USD"),  # USD/oz → JPY/g
    "Silver (JPY/g)":         ("SI=F",        "USD"),  # USD/oz → JPY/g
    # REITs
    "J-REIT Index (JP)":      ("1343.T",      "JPY"),  # NEXT FUNDS J-REIT ETF
    "US REIT (VNQ, JPY)":     ("VNQ",         "USD"),
    "Asia REIT (3269.T, JPY)":("3269.T",      "JPY"),  # Hoshino Resorts REIT
}

FX_TICKERS = {
    "USD": "JPY=X",
    "EUR": "EURJPY=X",
    "GBP": "GBPJPY=X",
    "AUD": "AUDJPY=X",
    "CAD": "CADJPY=X",
    "CNY": "CNYJPY=X",
    "INR": "INRJPY=X",
    "KRW": "KRWJPY=X",
    "BRL": "BRLJPY=X",
    "JPY": None,
}

TROY_OZ_TO_GRAM = 31.1035  # 1 troy oz = 31.1035 g

GROUPS = {
    "Equities — Developed":  ["Nikkei 225 (JP)","S&P 500 (US)","DAX (DE)","FTSE 100 (UK)",
                               "CAC 40 (FR)","ASX 200 (AU)","TSX (CA)","FTSE MIB (IT)"],
    "Equities — Emerging":   ["Shanghai Comp. (CN)","SENSEX (IN)","KOSPI (KR)","IBOVESPA (BR)"],
    "Commodities":           ["Gold (JPY/g)","Silver (JPY/g)"],
    "REITs":                 ["J-REIT Index (JP)","US REIT (VNQ, JPY)","Asia REIT (3269.T, JPY)"],
}

GROUP_COLORS = {
    "Equities — Developed": ["#1f4e79","#2e75b6","#5ba3d9","#9dc3e6",
                              "#c55a11","#ed7d31","#ffc000","#70ad47"],
    "Equities — Emerging":  ["#c00000","#ff4444","#ff9999","#ffcccc"],
    "Commodities":          ["#b8860b","#999999"],
    "REITs":                ["#375623","#70ad47","#a9d18e"],
}

# ── Download price data ───────────────────────────
print("Downloading price data...")
all_tickers = list({t for t, _ in TICKERS.values()})
fx_needed   = list({FX_TICKERS[c] for _, c in TICKERS.values() if c != "JPY"})

raw_prices = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                          progress=True, auto_adjust=True)["Close"]
raw_fx     = yf.download(fx_needed,   start=START_DATE, end=END_DATE,
                          progress=False, auto_adjust=True)["Close"]

# Ensure DataFrame even for single ticker
if isinstance(raw_prices, pd.Series):
    raw_prices = raw_prices.to_frame(name=all_tickers[0])
if isinstance(raw_fx, pd.Series):
    raw_fx = raw_fx.to_frame(name=fx_needed[0])

# ── Build JPY-denominated monthly series ─────────
monthly_series = {}

for label, (ticker, currency) in TICKERS.items():
    if ticker not in raw_prices.columns:
        print(f"  [SKIP] {label}: ticker not found")
        continue

    price = raw_prices[ticker].dropna()
    if price.empty:
        print(f"  [SKIP] {label}: no data")
        continue

    # Convert to JPY
    if currency != "JPY":
        fx_ticker = FX_TICKERS[currency]
        if fx_ticker not in raw_fx.columns:
            print(f"  [SKIP] {label}: FX rate not found")
            continue
        fx = raw_fx[fx_ticker].reindex(price.index, method="ffill").dropna()
        price = price.reindex(fx.index).dropna() * fx
    
    # Gold / Silver: USD/oz → JPY/g
    if label in ("Gold (JPY/g)", "Silver (JPY/g)"):
        price = price / TROY_OZ_TO_GRAM

    # Resample to month-end
    monthly = price.resample("ME").last().dropna()
    if len(monthly) < 3:
        print(f"  [SKIP] {label}: insufficient data ({len(monthly)} months)")
        continue

    monthly_series[label] = monthly
    print(f"  [OK]   {label}: {monthly.index[0].date()} to {monthly.index[-1].date()} ({len(monthly)} months)")

if not monthly_series:
    raise RuntimeError("No data downloaded. Check your internet connection and yfinance installation.")

# ── Find common start date (shortest series) ──────
first_dates = {k: s.index[0] for k, s in monthly_series.items()}
common_start = max(first_dates.values())  # latest first date = common start

shortest = max(first_dates, key=lambda k: first_dates[k])
print(f"\nShortest series: '{shortest}' starting {first_dates[shortest].date()}")
print(f"Common base date set to: {common_start.date()} (BASE_VALUE = {BASE_VALUE})")

# ── Normalise from common_start ───────────────────
normalised = {}
for label, series in monthly_series.items():
    clipped = series[series.index >= common_start]
    if clipped.empty:
        continue
    base = clipped.iloc[0]
    normalised[label] = (clipped / base * BASE_VALUE)

# ── Plot ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(20, 13))
fig.patch.set_facecolor("#f8f9fa")

base_label = common_start.strftime("%b %Y")
fig.suptitle(
    f"G20 Major Markets: Equities, Gold, Silver & REITs  |  "
    f"Base = {BASE_VALUE} ({base_label})  |  JPY-denominated",
    fontsize=14, fontweight="bold", y=0.99
)

axes_flat = axes.flatten()

for ax_i, (group_name, labels) in enumerate(GROUPS.items()):
    ax = axes_flat[ax_i]
    ax.set_facecolor("#ffffff")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#cccccc")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#e0e0e0", alpha=0.8)
    ax.grid(axis="x", linestyle=":",  linewidth=0.4, color="#e8e8e8", alpha=0.6)

    colors = GROUP_COLORS[group_name]
    plotted = 0

    for j, label in enumerate(labels):
        if label not in normalised:
            continue
        s = normalised[label]
        color = colors[j % len(colors)]
        lw = 2.2 if j < 2 else 1.6
        ax.plot(s.index, s.values, label=label, color=color,
                linewidth=lw, alpha=0.92)
        # Annotate last value
        ax.annotate(f"{s.iloc[-1]:.1f}",
                    xy=(s.index[-1], s.iloc[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7.5, color=color, fontweight="bold", va="center")
        plotted += 1

    ax.axhline(BASE_VALUE, color="#888", linewidth=0.9, linestyle="--", alpha=0.6)

    ax.set_title(group_name, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel(f"Index ({base_label} = {BASE_VALUE})", fontsize=9, color="#444")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=8)

    if plotted > 0:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85,
                  edgecolor="#cccccc", fancybox=True)
    else:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", va="center", color="#888", fontsize=11)

# ── Footer ────────────────────────────────────────
fig.text(0.01, 0.005,
         f"Source: Yahoo Finance via yfinance  |  Month-end prices  |  "
         f"Generated {today}  |  Gold & Silver converted to JPY/g",
         fontsize=7.5, color="#777")

plt.tight_layout(rect=[0, 0.015, 1, 0.975])

out = "g20_market_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"\nSaved: {out}")