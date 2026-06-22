"""
G20 Major Markets — Interactive Chart  v3
==========================================
pip install yfinance matplotlib pandas numpy

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
BASE_VALUE   = 100
FETCH_START  = "2000-01-01"
FETCH_END    = date.today().strftime("%Y-%m-%d")

# Resample frequency: "Daily" / "Weekly" / "Monthly"
RESAMPLE_FREQ = "Weekly"   # default; must match a key in FREQ_CODES below

TICKERS = {
    # label                 ticker(s)                          currency
    "Nikkei 225":         ("^N225",                         "JPY"),
    # TOPIX can be unstable on Yahoo depending on the symbol.
    # Use the index first, then fall back to alternatives.
    "TOPIX":              (("^TOPX", "^TPX", "1306.T"),     "JPY"),
    "S&P 500":            ("^GSPC",                         "USD"),
    "DAX":             ("^GDAXI",     "EUR"),
    "FTSE 100":        ("^FTSE",      "GBP"),
    "CAC 40":          ("^FCHI",      "EUR"),
    "Shanghai Comp.":  ("000001.SS",  "CNY"),
    "SENSEX":          ("^BSESN",     "INR"),
    "KOSPI":           ("^KS11",      "KRW"),
    "IBOVESPA":        ("^BVSP",      "BRL"),
    "ASX 200":         ("^AXJO",      "AUD"),
    "TSX":             ("^GSPTSE",    "CAD"),
    "FTSE MIB":        ("FTSEMIB.MI", "EUR"),
    "Gold (JPY/g)":   ("GC=F",       "USD"),
    "Silver (JPY/g)": ("SI=F",       "USD"),
    "J-REIT":          ("1343.T",     "JPY"),
    "US REIT (VNQ)":   ("VNQ",        "USD"),
}

FX_MAP = {
    "USD": "JPY=X",    "EUR": "EURJPY=X", "GBP": "GBPJPY=X",
    "AUD": "AUDJPY=X", "CAD": "CADJPY=X", "CNY": "CNYJPY=X",
    "INR": "INRJPY=X", "KRW": "KRWJPY=X", "BRL": "BRLJPY=X",
    "JPY": None,
}

def ticker_candidates(spec):
    """Return a list of ticker strings from a TICKERS entry."""
    if isinstance(spec, (tuple, list)):
        if len(spec) == 2 and isinstance(spec[0], str):
            return [spec[0]]
        if len(spec) == 2 and isinstance(spec[0], (tuple, list)):
            return list(spec[0])
        return list(spec)
    return [spec]

COLORS = [
    "#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd",
    "#8c564b","#e377c2","#17becf","#bcbd22","#7f7f7f",
    "#393b79","#637939","#843c39","#7b4173","#5254a3",
    "#3182bd","#6baed6",
]

TROY_TO_G   = 31.1035
DEFAULT_ON  = {"Nikkei 225", "S&P 500", "Gold (JPY/g)"}

FREQ_OPTIONS  = ["Daily", "Weekly", "Monthly"]
FREQ_CODES    = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}

# ══════════════════════════════════════════════
#  1. DOWNLOAD
# ══════════════════════════════════════════════
print("Downloading price data…")

all_t  = sorted({t for spec, _ in TICKERS.values() for t in ticker_candidates(spec)})
fx_t   = sorted({FX_MAP[c] for _, c in TICKERS.values() if c != "JPY" and FX_MAP[c] is not None})

raw_p  = yf.download(all_t, start=FETCH_START, end=FETCH_END,
                     progress=True, auto_adjust=True)["Close"]
raw_fx = yf.download(fx_t, start=FETCH_START, end=FETCH_END,
                     progress=False, auto_adjust=True)["Close"]

if isinstance(raw_p, pd.Series):
    raw_p = raw_p.to_frame(name=all_t[0])
if isinstance(raw_fx, pd.Series):
    raw_fx = raw_fx.to_frame(name=fx_t[0])

# build daily JPY series (resample later per frequency)
daily = {}
for label, (ticker_spec, ccy) in TICKERS.items():
    picked = None
    for ticker in ticker_candidates(ticker_spec):
        if ticker in raw_p.columns and not raw_p[ticker].dropna().empty:
            picked = ticker
            break
    if picked is None:
        print(f"  [SKIP] {label} — no usable price series found")
        continue

    s = raw_p[picked].dropna()
    if s.empty:
        print(f"  [SKIP] {label} — empty")
        continue

    if ccy != "JPY":
        fx_col = FX_MAP[ccy]
        if fx_col not in raw_fx.columns:
            print(f"  [SKIP] {label} — FX missing")
            continue
        fx = raw_fx[fx_col].reindex(s.index, method="ffill").dropna()
        s  = s.reindex(fx.index).dropna() * fx

    # Convert gold/silver futures from oz to gram after JPY conversion
    if label in ("Gold (JPY/g)", "Silver (JPY/g)"):
        s = s / TROY_TO_G

    daily[label] = s
    print(f"  [OK]  {label} [{picked}]: {s.index[0].date()} — {s.index[-1].date()}  ({len(s)} days)")

if not daily:
    sys.exit("No data downloaded. Check internet connection.")

labels_list = list(daily.keys())

# ══════════════════════════════════════════════
#  2. HELPERS
# ══════════════════════════════════════════════
def resample_series(freq_code):
    """Resample all daily series to freq_code and return dict."""
    out = {}
    for lb, s in daily.items():
        if freq_code == "D":
            r = s.copy()
        elif freq_code == "W":
            r = s.resample("W-FRI").last().dropna()
        else:  # ME
            r = s.resample("ME").last().dropna()
        if len(r) >= 2:
            out[lb] = r
    return out

def common_start(resampled):
    """Latest first-date across all series currently available."""
    first = {k: v.index[0] for k, v in resampled.items()}
    key = max(first, key=lambda k: first[k])
    return first[key], key

def get_normalised(label, resampled, base_date, base_value):
    """Normalise from the first available value on/after base_date.

    Returns
    -------
    (series, reason)
        reason is None when the series can be drawn.
    """
    if label not in resampled:
        return None, "not available at this frequency"
    s = resampled[label]
    clipped = s[s.index >= pd.Timestamp(base_date)]
    if clipped.empty:
        return None, "no data on/after selected base date"
    base = clipped.iloc[0]
    if base == 0:
        return None, "base value is zero"
    return clipped / base * base_value, None


def parse_date_text(text):
    """Parse YYYY-MM / YYYY-MM-DD / slashed variants into a Timestamp."""
    text = text.strip()
    for fmt in ("%Y-%m", "%Y-%m-%d", "%Y/%m", "%Y/%m/%d"):
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(text)
    except Exception:
        return None


# ══════════════════════════════════════════════
#  3. INITIAL STATE
# ══════════════════════════════════════════════
resampled = resample_series(FREQ_CODES[RESAMPLE_FREQ])
common_earliest, shortest_key = common_start(resampled)

state = {
    "freq":       RESAMPLE_FREQ,
    "resampled":  resampled,
    "common_earliest": common_earliest,
    "shortest":   shortest_key,
    "base_date":  common_earliest,
    "base_value": BASE_VALUE,
    "end_date":   pd.Timestamp(FETCH_END),
    "active":     {lb: (lb in DEFAULT_ON) for lb in labels_list},
}

# ══════════════════════════════════════════════
#  4. FIGURE  (tall enough for bottom controls)
# ══════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#f8f9fa")

# Layout (left=checks, right=chart, bottom=controls)
#   [title bar]
#   [checks | chart         ]
#   [checks | x-axis labels ]
#   [controls row 1         ]
#   [controls row 2         ]
#   [footer                 ]

AX_CHART  = [0.25, 0.20, 0.73, 0.70]   # chart leaves room below for labels
AX_CHECKS = [0.01, 0.20, 0.22, 0.70]
AX_TITLE  = [0.25, 0.91, 0.73, 0.07]

ax_chart  = fig.add_axes(AX_CHART)
ax_checks = fig.add_axes(AX_CHECKS)
ax_title  = fig.add_axes(AX_TITLE);  ax_title.axis("off")
ax_checks.set_facecolor("#f0f0f0")
ax_checks.set_title("Select indices", fontsize=13, pad=9, color="#333", fontweight="bold")

# ── control row 1: base-date  +  base-value  (y≈0.12)
y1 = 0.125
# Base date label + box
ax_bd_lbl = fig.add_axes([0.21, y1, 0.09, 0.04]); ax_bd_lbl.axis("off")
ax_bd_lbl.text(1.0, 0.5, "Base date (YYYY-MM):", ha="right", va="center",
               fontsize=8, color="#444")
ax_bd_box = fig.add_axes([0.31, y1+0.005, 0.10, 0.033])
tb_date   = TextBox(ax_bd_box, "", initial=common_earliest.strftime("%Y-%m"))

# common_earliest-info text
ax_bd_inf = fig.add_axes([0.42, y1, 0.16, 0.04]); ax_bd_inf.axis("off")
bd_info   = ax_bd_inf.text(0, 0.5,
    f"Common start: {common_earliest.strftime('%Y-%m')}  (limited by '{shortest_key}')",
    va="center", fontsize=7.1, color="#666")

# End date label + box
ax_ed_lbl = fig.add_axes([0.58, y1, 0.06, 0.04]); ax_ed_lbl.axis("off")
ax_ed_lbl.text(1.0, 0.5, "End date:", ha="right", va="center",
               fontsize=8, color="#444")
ax_ed_box = fig.add_axes([0.645, y1+0.005, 0.10, 0.033])
tb_edate  = TextBox(ax_ed_box, "", initial=pd.Timestamp(FETCH_END).strftime("%Y-%m"))

# Base value label + box
ax_bv_lbl = fig.add_axes([0.76, y1, 0.07, 0.04]); ax_bv_lbl.axis("off")
ax_bv_lbl.text(1.0, 0.5, "Base value:", ha="right", va="center",
               fontsize=8, color="#444")
ax_bv_box = fig.add_axes([0.84, y1+0.005, 0.07, 0.033])
tb_bval   = TextBox(ax_bv_box, "", initial=str(BASE_VALUE))

# ── control row 2: freq buttons  +  select/clear  +  save  (y≈0.07)
y2 = 0.07
freq_axes = []
freq_btns = []
freq_x    = 0.21
for i, fname in enumerate(FREQ_OPTIONS):
    a = fig.add_axes([freq_x + i*0.075, y2, 0.068, 0.037])
    clr  = "#c8dff5" if fname == RESAMPLE_FREQ else "#eeeeee"
    b    = Button(a, fname, color=clr, hovercolor="#aaccee")
    b.label.set_fontsize(8)
    freq_axes.append(a); freq_btns.append(b)

ax_all   = fig.add_axes([0.455, y2, 0.08, 0.037])
ax_clear = fig.add_axes([0.540, y2, 0.08, 0.037])
ax_save  = fig.add_axes([0.870, y2, 0.10, 0.037])
btn_all   = Button(ax_all,   "Select All",  color="#ddeeff", hovercolor="#bbddff")
btn_clear = Button(ax_clear, "Clear All",   color="#ffeedd", hovercolor="#ffccbb")
btn_save  = Button(ax_save,  "Save PNG",    color="#ddeedd", hovercolor="#bbddbb")
for b in (btn_all, btn_clear, btn_save):
    b.label.set_fontsize(8)

# ══════════════════════════════════════════════
#  5. CHECKBOXES
# ══════════════════════════════════════════════
check_init = [state["active"][lb] for lb in labels_list]
chk = CheckButtons(ax_checks, labels_list, check_init)
for t in chk.labels:
    t.set_fontsize(10); t.set_color("#222")

# Make the checkbox labels easier to read across Matplotlib versions
# (Older Matplotlib builds do not expose chk.rectangles.)
for label in chk.labels:
    label.set_fontsize(10)
    label.set_color("#222")

# ══════════════════════════════════════════════
#  6. REDRAW
# ══════════════════════════════════════════════
def redraw():
    ax_chart.cla()
    ax_chart.set_facecolor("#ffffff")
    ax_chart.spines[["top","right"]].set_visible(False)
    ax_chart.spines[["left","bottom"]].set_color("#cccccc")
    ax_chart.grid(axis="y", linestyle="--", lw=0.6, color="#e0e0e0", alpha=0.8)
    ax_chart.grid(axis="x", linestyle=":",  lw=0.4, color="#e8e8e8", alpha=0.6)

    rs  = state["resampled"]
    bd  = state["base_date"]
    ed  = state["end_date"]
    bv  = state["base_value"]

    plotted = []
    missing = []

    for i, label in enumerate(labels_list):
        if not state["active"][label]:
            continue
        ns, reason = get_normalised(label, rs, bd, bv)
        if ns is None:
            missing.append(f"{label} ({reason})")
            continue
        ns = ns[ns.index <= ed]
        if ns.empty:
            missing.append(f"{label} (no data on/before selected end date)")
            continue
        color = COLORS[i % len(COLORS)]
        lw    = 1.8 if len(ns) > 500 else 2.0
        ax_chart.plot(ns.index, ns.values, label=label,
                      color=color, linewidth=lw, alpha=0.9)
        ax_chart.annotate(f"{ns.iloc[-1]:.1f}",
                          xy=(ns.index[-1], ns.iloc[-1]),
                          xytext=(5, 0), textcoords="offset points",
                          fontsize=7, color=color, fontweight="bold", va="center")
        plotted.append(label)

    ax_chart.axhline(bv, color="#999", lw=0.9, linestyle="--", alpha=0.6)
    ax_chart.set_ylabel(f"Index  ({pd.Timestamp(bd).strftime('%Y-%m-%d')} → {pd.Timestamp(ed).strftime('%Y-%m-%d')} = {bv:.4g})",
                        fontsize=9, color="#444")
    ax_chart.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # smart x-axis formatting by data span
    if plotted:
        all_ns = [get_normalised(lb, rs, bd, bv)[0] for lb in plotted]
        spans  = [(ns.index[-1] - ns.index[0]).days for ns in all_ns if ns is not None]
        span   = max(spans) if spans else 365
    else:
        span = 365

    if span <= 180:          # ≤6 months → monthly ticks
        ax_chart.xaxis.set_major_locator(mdates.MonthLocator())
        ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    elif span <= 730:        # ≤2 years → quarterly
        ax_chart.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    elif span <= 1825:       # ≤5 years → half-yearly
        ax_chart.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]))
        ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    else:                    # >5 years → yearly
        ax_chart.xaxis.set_major_locator(mdates.YearLocator())
        ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.setp(ax_chart.get_xticklabels(), rotation=35, fontsize=8, ha="right")
    plt.setp(ax_chart.get_yticklabels(), fontsize=8)

    if plotted:
        ax_chart.legend(loc="upper left", fontsize=8,
                        framealpha=0.88, edgecolor="#ccc", fancybox=True)

    # title bar
    ax_title.cla()
    ax_title.axis("off")
    ax_title.text(0, 0.78,
                  "G20 Major Markets: Equities · Gold · Silver · REITs",
                  fontsize=13, fontweight="bold", color="#1a1a2e")
    ax_title.text(0, 0.28,
                  f"JPY-denominated · {state['freq']} data · "
                  f"Base {pd.Timestamp(bd).strftime('%Y-%m-%d')} = {bv:.4g} · "
                  f"Source: Yahoo Finance",
                  fontsize=8, color="#666")

    if missing:
        short = ", ".join(missing[:4])
        if len(missing) > 4:
            short += f" … (+{len(missing) - 4} more)"
        ax_title.text(0, -0.15,
                      f"Not drawn at this base date: {short}",
                      fontsize=7.3, color="#b00020", clip_on=False)
        bd_info.set_text(f"Selected: {pd.Timestamp(bd).strftime('%Y-%m-%d')} → {pd.Timestamp(ed).strftime('%Y-%m-%d')}  |  Missing: {len(missing)}")
        bd_info.set_color("#b00020")
    else:
        bd_info.set_text(f"Selected: {pd.Timestamp(bd).strftime('%Y-%m-%d')} → {pd.Timestamp(ed).strftime('%Y-%m-%d')}  |  All selected series drawable")
        bd_info.set_color("#666")

    fig.canvas.draw_idle()

# ══════════════════════════════════════════════
#  7. CALLBACKS
# ══════════════════════════════════════════════
def on_check(clicked):
    state["active"][clicked] = not state["active"][clicked]
    redraw()
chk.on_clicked(on_check)

def on_date_submit(text):
    parsed = parse_date_text(text)
    if parsed is None:
        bd_info.set_text("Invalid format — use YYYY-MM or YYYY-MM-DD")
        bd_info.set_color("red")
        fig.canvas.draw_idle()
        return

    if parsed > state["end_date"]:
        parsed = state["end_date"]
        tb_date.set_val(parsed.strftime("%Y-%m-%d"))
        bd_info.set_text("Base date clamped to end date")
        bd_info.set_color("#b85c00")
    else:
        bd_info.set_color("#666")

    state["base_date"] = parsed
    redraw()
tb_date.on_submit(on_date_submit)

def on_bval_submit(text):
    try:
        v = float(text.strip())
        if v > 0:
            state["base_value"] = v
            redraw()
    except ValueError:
        pass
tb_bval.on_submit(on_bval_submit)

def on_end_date_submit(text):
    parsed = parse_date_text(text)
    if parsed is None:
        bd_info.set_text("Invalid end date — use YYYY-MM or YYYY-MM-DD")
        bd_info.set_color("red")
        fig.canvas.draw_idle()
        return

    if parsed < state["base_date"]:
        state["end_date"] = state["base_date"]
        tb_edate.set_val(state["base_date"].strftime("%Y-%m-%d"))
        bd_info.set_text("End date clamped to base date")
        bd_info.set_color("#b85c00")
    else:
        state["end_date"] = parsed
        bd_info.set_color("#666")
    redraw()

tb_edate.on_submit(on_end_date_submit)

def make_freq_cb(fname):
    def cb(event):
        state["freq"] = fname
        code = FREQ_CODES[fname]
        state["resampled"] = resample_series(code)
        e, sk = common_start(state["resampled"])
        state["common_earliest"] = e
        state["shortest"] = sk
        bd_info.set_text(
            f"Common start: {e.strftime('%Y-%m')}  (limited by '{sk}')")
        bd_info.set_color("#666")
        # highlight active button
        for b, fn in zip(freq_btns, FREQ_OPTIONS):
            b.color = "#c8dff5" if fn == fname else "#eeeeee"
            b.hovercolor = "#aaccee"
            b.ax.set_facecolor(b.color)
        redraw()
    return cb

for btn, fname in zip(freq_btns, FREQ_OPTIONS):
    btn.on_clicked(make_freq_cb(fname))

def select_all(event):
    for i, lb in enumerate(labels_list):
        if not state["active"][lb]: chk.set_active(i)
def clear_all(event):
    for i, lb in enumerate(labels_list):
        if state["active"][lb]: chk.set_active(i)
def save_png(event):
    fname = f"g20_{state['freq'].lower()}_{pd.Timestamp(state['base_date']).strftime('%Y%m%d')}_to_{pd.Timestamp(state['end_date']).strftime('%Y%m%d')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {fname}")

btn_all.on_clicked(select_all)
btn_clear.on_clicked(clear_all)
btn_save.on_clicked(save_png)

# ══════════════════════════════════════════════
#  8. FOOTER + INITIAL DRAW
# ══════════════════════════════════════════════
fig.text(0.01, 0.01,
         f"Data: Yahoo Finance · Generated {date.today()} · All prices in JPY",
         fontsize=6.5, color="#aaa")

redraw()
plt.show()