"""
G20 Major Markets — Interactive Chart
======================================
pip install yfinance matplotlib pandas numpy mplcursors

Run:  python g20_market_chart.py
"""

import warnings, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.widgets import CheckButtons, Button, TextBox
from datetime import date, timedelta
import yfinance as yf

# ══════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════
BASE_VALUE  = 100
FETCH_START = "2020-01-01"   # fetch extra history for flexible base-date
FETCH_END   = date.today().strftime("%Y-%m-%d")

TICKERS = {
    "Nikkei 225":        ("^N225",      "JPY"),
    "S&P 500":           ("^GSPC",      "USD"),
    "DAX":               ("^GDAXI",     "EUR"),
    "FTSE 100":          ("^FTSE",      "GBP"),
    "CAC 40":            ("^FCHI",      "EUR"),
    "Shanghai Comp.":    ("000001.SS",  "CNY"),
    "SENSEX":            ("^BSESN",     "INR"),
    "KOSPI":             ("^KS11",      "KRW"),
    "IBOVESPA":          ("^BVSP",      "BRL"),
    "ASX 200":           ("^AXJO",      "AUD"),
    "TSX":               ("^GSPTSE",    "CAD"),
    "FTSE MIB":          ("FTSEMIB.MI", "EUR"),
    "Gold (JPY/g)":      ("GC=F",       "USD"),
    "Silver (JPY/g)":    ("SI=F",       "USD"),
    "J-REIT":            ("1343.T",     "JPY"),
    "US REIT (VNQ)":     ("VNQ",        "USD"),
}

FX_MAP = {
    "USD": "JPY=X", "EUR": "EURJPY=X", "GBP": "GBPJPY=X",
    "AUD": "AUDJPY=X", "CAD": "CADJPY=X", "CNY": "CNYJPY=X",
    "INR": "INRJPY=X", "KRW": "KRWJPY=X", "BRL": "BRLJPY=X",
    "JPY": None,
}

COLORS = [
    "#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd",
    "#8c564b","#e377c2","#17becf","#bcbd22","#7f7f7f",
    "#393b79","#637939","#843c39","#7b4173","#5254a3",
    "#3182bd",
]

TROY_TO_G = 31.1035

# ══════════════════════════════════════════════
#  1. DOWNLOAD DATA
# ══════════════════════════════════════════════
print("Downloading price data…")

all_t  = list({t for t,_ in TICKERS.values()})
fx_t   = list({FX_MAP[c] for _,c in TICKERS.values() if c != "JPY"})

raw_p  = yf.download(all_t,  start=FETCH_START, end=FETCH_END, progress=True,  auto_adjust=True)["Close"]
raw_fx = yf.download(fx_t,   start=FETCH_START, end=FETCH_END, progress=False, auto_adjust=True)["Close"]

if isinstance(raw_p,  pd.Series): raw_p  = raw_p.to_frame(name=all_t[0])
if isinstance(raw_fx, pd.Series): raw_fx = raw_fx.to_frame(name=fx_t[0])

# build monthly JPY series
monthly = {}
for label, (ticker, ccy) in TICKERS.items():
    if ticker not in raw_p.columns:
        print(f"  [SKIP] {label}")
        continue
    s = raw_p[ticker].dropna()
    if s.empty: continue
    if ccy != "JPY":
        fx_col = FX_MAP[ccy]
        if fx_col not in raw_fx.columns: continue
        fx = raw_fx[fx_col].reindex(s.index, method="ffill").dropna()
        s  = s.reindex(fx.index).dropna() * fx
    if label in ("Gold (JPY/g)", "Silver (JPY/g)"):
        s = s / TROY_TO_G
    m = s.resample("ME").last().dropna()
    if len(m) >= 3:
        monthly[label] = m
        print(f"  [OK]  {label}: {m.index[0].date()} — {m.index[-1].date()}")

if not monthly:
    sys.exit("No data. Check internet / yfinance installation.")

# ══════════════════════════════════════════════
#  2. COMMON DATE LOGIC
# ══════════════════════════════════════════════
# The earliest possible base date = start of the shortest series
first_dates   = {k: v.index[0] for k,v in monthly.items()}
shortest_key  = max(first_dates, key=lambda k: first_dates[k])
EARLIEST_BASE = first_dates[shortest_key]     # can't go earlier than this

print(f"\nShortest series : '{shortest_key}' from {EARLIEST_BASE.date()}")

labels_list = list(monthly.keys())
# default: show Nikkei + S&P + Gold
DEFAULT_ON = {"Nikkei 225", "S&P 500", "Gold (JPY/g)"}

# ══════════════════════════════════════════════
#  3. FIGURE LAYOUT
# ══════════════════════════════════════════════
fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor("#f8f9fa")

# --- axes proportions ---
# left panel: checkboxes  |  right: chart  |  top: controls
ax_chart  = fig.add_axes([0.22, 0.13, 0.76, 0.74])  # main chart
ax_checks = fig.add_axes([0.01, 0.13, 0.19, 0.74])  # checkbox panel
ax_title  = fig.add_axes([0.22, 0.89, 0.76, 0.08])  # title area (no frame)
ax_title.axis("off")

ax_chart.set_facecolor("#ffffff")
ax_chart.spines[["top","right"]].set_visible(False)
ax_chart.spines[["left","bottom"]].set_color("#cccccc")
ax_chart.grid(axis="y", linestyle="--", lw=0.6, color="#e0e0e0", alpha=0.8)
ax_chart.grid(axis="x", linestyle=":",  lw=0.4, color="#e8e8e8", alpha=0.6)

ax_checks.set_facecolor("#f0f0f0")
ax_checks.set_title("Select indices", fontsize=9, pad=4, color="#333")

# --- TextBox for base-date ---
ax_tb_label = fig.add_axes([0.22, 0.05, 0.10, 0.05])
ax_tb_label.axis("off")
ax_tb_label.text(1.0, 0.5, "Base date\n(YYYY-MM):",
                 ha="right", va="center", fontsize=8, color="#444")

ax_textbox = fig.add_axes([0.33, 0.055, 0.13, 0.04])
textbox = TextBox(ax_textbox, "", initial=EARLIEST_BASE.strftime("%Y-%m"))

ax_base_info = fig.add_axes([0.47, 0.05, 0.30, 0.05])
ax_base_info.axis("off")
base_info_txt = ax_base_info.text(
    0.0, 0.5,
    f"Earliest available: {EARLIEST_BASE.strftime('%Y-%m')}  "
    f"(limited by '{shortest_key}')",
    va="center", fontsize=7.5, color="#666"
)

# --- Base-value TextBox ---
ax_bv_label = fig.add_axes([0.77, 0.05, 0.08, 0.05])
ax_bv_label.axis("off")
ax_bv_label.text(1.0, 0.5, "Base value:", ha="right", va="center",
                 fontsize=8, color="#444")

ax_bv_box = fig.add_axes([0.86, 0.055, 0.07, 0.04])
bv_textbox = TextBox(ax_bv_box, "", initial=str(BASE_VALUE))

# ══════════════════════════════════════════════
#  4. STATE
# ══════════════════════════════════════════════
state = {
    "base_date":  EARLIEST_BASE,
    "base_value": BASE_VALUE,
    "active":     {lb: (lb in DEFAULT_ON) for lb in labels_list},
    "lines":      {},
    "legend":     None,
}

# ══════════════════════════════════════════════
#  5. DRAW / REDRAW CHART
# ══════════════════════════════════════════════
def get_normalised(label):
    """Return normalised series from state['base_date'], or None."""
    s = monthly[label]
    bd = state["base_date"]
    bv = state["base_value"]
    clipped = s[s.index >= bd]
    if clipped.empty:
        return None
    base = clipped.iloc[0]
    if base == 0:
        return None
    return clipped / base * bv

def redraw():
    ax_chart.cla()
    ax_chart.set_facecolor("#ffffff")
    ax_chart.spines[["top","right"]].set_visible(False)
    ax_chart.spines[["left","bottom"]].set_color("#cccccc")
    ax_chart.grid(axis="y", linestyle="--", lw=0.6, color="#e0e0e0", alpha=0.8)
    ax_chart.grid(axis="x", linestyle=":",  lw=0.4, color="#e8e8e8", alpha=0.6)

    bd  = state["base_date"]
    bv  = state["base_value"]
    plotted = []

    for i, label in enumerate(labels_list):
        if not state["active"][label]:
            continue
        ns = get_normalised(label)
        if ns is None:
            continue
        color = COLORS[i % len(COLORS)]
        line, = ax_chart.plot(ns.index, ns.values,
                              label=label, color=color,
                              linewidth=2.0, alpha=0.9)
        # annotate last value
        ax_chart.annotate(f"{ns.iloc[-1]:.1f}",
                          xy=(ns.index[-1], ns.iloc[-1]),
                          xytext=(5, 0), textcoords="offset points",
                          fontsize=7, color=color, fontweight="bold", va="center")
        plotted.append(label)

    # base line
    ax_chart.axhline(bv, color="#999", linewidth=0.9, linestyle="--", alpha=0.7)

    ax_chart.set_ylabel(f"Index  ({bd.strftime('%Y-%m')} = {bv})",
                        fontsize=9, color="#444")
    ax_chart.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax_chart.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax_chart.get_xticklabels(), rotation=30, fontsize=8)
    plt.setp(ax_chart.get_yticklabels(), fontsize=8)

    if plotted:
        ax_chart.legend(loc="upper left", fontsize=8,
                        framealpha=0.88, edgecolor="#ccc", fancybox=True)

    # update title
    ax_title.cla(); ax_title.axis("off")
    ax_title.text(0.0, 0.7,
                  "G20 Major Markets: Equities · Gold · Silver · REITs",
                  fontsize=13, fontweight="bold", color="#1a1a2e")
    ax_title.text(0.0, 0.1,
                  f"JPY-denominated · Month-end · Base {bd.strftime('%Y-%m')} = {bv} "
                  f"· Source: Yahoo Finance",
                  fontsize=8, color="#666")

    fig.canvas.draw_idle()

# ══════════════════════════════════════════════
#  6. CHECKBOXES
# ══════════════════════════════════════════════
check_init = [state["active"][lb] for lb in labels_list]
chk = CheckButtons(ax_checks, labels_list, check_init)

# style checkbox labels
for text_obj in chk.labels:
    text_obj.set_fontsize(8)
    text_obj.set_color("#222")

def on_check(clicked_label):
    state["active"][clicked_label] = not state["active"][clicked_label]
    redraw()

chk.on_clicked(on_check)

# ══════════════════════════════════════════════
#  7. BASE-DATE TEXTBOX
# ══════════════════════════════════════════════
def on_basedate_submit(text):
    text = text.strip()
    # accept YYYY-MM or YYYY-MM-DD
    for fmt in ("%Y-%m", "%Y-%m-%d", "%Y/%m", "%Y/%m/%d"):
        try:
            parsed = pd.to_datetime(text, format=fmt)
            break
        except ValueError:
            parsed = None
    if parsed is None:
        ax_base_info.cla(); ax_base_info.axis("off")
        ax_base_info.text(0, 0.5, "Invalid date format. Use YYYY-MM.",
                          va="center", fontsize=7.5, color="red")
        fig.canvas.draw_idle()
        return

    parsed = parsed.normalize()
    # clamp to EARLIEST_BASE
    if parsed < EARLIEST_BASE:
        parsed = EARLIEST_BASE
        textbox.set_val(EARLIEST_BASE.strftime("%Y-%m"))
        ax_base_info.cla(); ax_base_info.axis("off")
        ax_base_info.text(0, 0.5,
                          f"Clamped to earliest available: "
                          f"{EARLIEST_BASE.strftime('%Y-%m')} ('{shortest_key}')",
                          va="center", fontsize=7.5, color="#b85c00")
        fig.canvas.draw_idle()
    else:
        ax_base_info.cla(); ax_base_info.axis("off")
        ax_base_info.text(0, 0.5,
                          f"Earliest available: {EARLIEST_BASE.strftime('%Y-%m')} "
                          f"(limited by '{shortest_key}')",
                          va="center", fontsize=7.5, color="#666")
        fig.canvas.draw_idle()

    state["base_date"] = parsed
    redraw()

textbox.on_submit(on_basedate_submit)

# ══════════════════════════════════════════════
#  8. BASE-VALUE TEXTBOX
# ══════════════════════════════════════════════
def on_basevalue_submit(text):
    try:
        v = float(text.strip())
        if v <= 0: raise ValueError
        state["base_value"] = v
        redraw()
    except ValueError:
        pass

bv_textbox.on_submit(on_basevalue_submit)

# ══════════════════════════════════════════════
#  9. SELECT-ALL / CLEAR BUTTONS
# ══════════════════════════════════════════════
ax_btn_all   = fig.add_axes([0.01, 0.06, 0.09, 0.04])
ax_btn_clear = fig.add_axes([0.11, 0.06, 0.09, 0.04])

btn_all   = Button(ax_btn_all,   "Select All",  color="#ddeeff", hovercolor="#bbddff")
btn_clear = Button(ax_btn_clear, "Clear All",   color="#ffeedd", hovercolor="#ffccbb")

def on_select_all(event):
    for lb in labels_list:
        if not state["active"][lb]:
            state["active"][lb] = True
            # sync checkbox widget
            idx = labels_list.index(lb)
            chk.set_active(idx)   # toggles the visual, but also fires on_check again
    redraw()

def on_clear_all(event):
    for lb in labels_list:
        if state["active"][lb]:
            state["active"][lb] = False
            idx = labels_list.index(lb)
            chk.set_active(idx)
    redraw()

# Use a simpler approach: directly toggle checkboxes via rectangles
def select_all(event):
    for i, lb in enumerate(labels_list):
        if not state["active"][lb]:
            chk.set_active(i)

def clear_all(event):
    for i, lb in enumerate(labels_list):
        if state["active"][lb]:
            chk.set_active(i)

btn_all.on_clicked(select_all)
btn_clear.on_clicked(clear_all)

# ══════════════════════════════════════════════
#  10. SAVE BUTTON
# ══════════════════════════════════════════════
ax_btn_save = fig.add_axes([0.87, 0.01, 0.11, 0.04])
btn_save    = Button(ax_btn_save, "Save PNG", color="#ddeedd", hovercolor="#bbddbb")

def save_chart(event):
    fname = f"g20_chart_{state['base_date'].strftime('%Y%m')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {fname}")

btn_save.on_clicked(save_chart)

# ══════════════════════════════════════════════
#  11. INITIAL DRAW
# ══════════════════════════════════════════════
redraw()

fig.text(0.01, 0.01,
         f"Data: Yahoo Finance · Generated {date.today()} · All prices in JPY",
         fontsize=6.5, color="#aaa")

plt.show()