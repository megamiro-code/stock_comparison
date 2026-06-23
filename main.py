"""
G20 Major Markets — Interactive Chart  v5
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
from datetime import date
import yfinance as yf

# ══════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════
BASE_VALUE    = 100
FETCH_START   = "2000-01-01"
FETCH_END     = date.today().strftime("%Y-%m-%d")
RESAMPLE_FREQ = "Weekly"

# ── Dark mode palette ──
BG_DARK   = "#1a1a2e"
BG_PANEL  = "#16213e"
BG_CHART  = "#0f3460"
BG_CTRL   = "#1a1a2e"
TXT_MAIN  = "#e0e0e0"
TXT_DIM   = "#8888aa"
GRID_MAJ  = "#2a2a4a"
GRID_MIN  = "#222240"
SPINE_CLR = "#444466"
BTN_DEF   = "#2a2a4a"
BTN_HOV   = "#3a3a6a"
BTN_ACT   = "#1f4e8c"
BTN_ALL   = "#1a3a5c"
BTN_CLR   = "#4a1a1a"
BTN_SAV   = "#1a3a2a"
WARN_CLR  = "#ffaa44"
ERR_CLR   = "#ff6666"
BASE_LINE = "#556688"

COLORS = [
    "#4db8ff","#ff6b6b","#6bff6b","#ffaa4d","#cc88ff",
    "#ff88cc","#4dffee","#ffee4d","#ff884d","#88ffcc",
    "#88aaff","#ffcc88","#cc4dff","#88ff44","#ff4488",
    "#44ccff","#ffbb44",
]

TICKERS = {
    "Nikkei 225":        ("^N225",       "JPY"),
    "TOPIX (1306.T)":    ("1306.T",      "JPY"),
    "S&P 500":           ("^GSPC",       "USD"),
    "DAX":               ("^GDAXI",      "EUR"),
    "FTSE 100":          ("^FTSE",       "GBP"),
    "CAC 40":            ("^FCHI",       "EUR"),
    "Shanghai Comp.":    ("000001.SS",   "CNY"),
    "SENSEX":            ("^BSESN",      "INR"),
    "KOSPI":             ("^KS11",       "KRW"),
    "IBOVESPA":          ("^BVSP",       "BRL"),
    "ASX 200":           ("^AXJO",       "AUD"),
    "TSX":               ("^GSPTSE",     "CAD"),
    "FTSE MIB":          ("FTSEMIB.MI",  "EUR"),
    "Gold (JPY/g)":      ("GC=F",        "USD"),
    "Silver (JPY/g)":    ("SI=F",        "USD"),
    "J-REIT":            ("1343.T",      "JPY"),
    "US REIT (VNQ)":     ("VNQ",         "USD"),
}

FX_MAP = {
    "USD": "JPY=X",    "EUR": "EURJPY=X", "GBP": "GBPJPY=X",
    "AUD": "AUDJPY=X", "CAD": "CADJPY=X", "CNY": "CNYJPY=X",
    "INR": "INRJPY=X", "KRW": "KRWJPY=X", "BRL": "BRLJPY=X",
    "JPY": None,
}

TROY_TO_G    = 31.1035
DEFAULT_ON   = {"Nikkei 225", "S&P 500", "Gold (JPY/g)"}
FREQ_OPTIONS = ["Daily", "Weekly", "Monthly"]
FREQ_CODES   = {"Daily": "D", "Weekly": "W-FRI", "Monthly": "ME"}

# ══════════════════════════════════════════════
#  1. DOWNLOAD
# ══════════════════════════════════════════════
print("Downloading price data …")

all_t  = list({t for t, _ in TICKERS.values()})
fx_t   = list({FX_MAP[c] for _, c in TICKERS.values() if c != "JPY"})

raw_p  = yf.download(all_t, start=FETCH_START, end=FETCH_END,
                     progress=True, auto_adjust=True)["Close"]
raw_fx = yf.download(fx_t,  start=FETCH_START, end=FETCH_END,
                     progress=False, auto_adjust=True)["Close"]

if isinstance(raw_p,  pd.Series): raw_p  = raw_p.to_frame(name=all_t[0])
if isinstance(raw_fx, pd.Series): raw_fx = raw_fx.to_frame(name=fx_t[0])

# ── Build daily JPY series ──
daily = {}
for label, (ticker, ccy) in TICKERS.items():
    if ticker not in raw_p.columns:
        print(f"  [SKIP] {label} — ticker not found"); continue
    s = raw_p[ticker].dropna()
    if s.empty:
        print(f"  [SKIP] {label} — empty"); continue
    if ccy != "JPY":
        fx_col = FX_MAP[ccy]
        if fx_col not in raw_fx.columns:
            print(f"  [SKIP] {label} — FX missing"); continue
        fx = raw_fx[fx_col].reindex(s.index, method="ffill").dropna()
        s  = s.reindex(fx.index).dropna() * fx
    if label in ("Gold (JPY/g)", "Silver (JPY/g)"):
        s = s / TROY_TO_G
    # Spike removal: drop points > 50% away from 30-day rolling median
    med   = s.rolling(30, min_periods=1, center=True).median()
    ratio = s / med
    s     = s[ratio.between(0.5, 2.0)]
    if len(s) < 5:
        print(f"  [SKIP] {label} — insufficient clean data"); continue
    daily[label] = s
    print(f"  [OK]  {label}: {s.index[0].date()} — {s.index[-1].date()} ({len(s)} days)")

if not daily:
    sys.exit("No data downloaded.")

labels_list = list(daily.keys())

# ══════════════════════════════════════════════
#  2. RESAMPLE — pre-compute all three frequencies once
# ══════════════════════════════════════════════
def _resample(s, code):
    if code == "D":    return s.copy()
    if code == "W-FRI": return s.resample("W-FRI").last().dropna()
    return s.resample("ME").last().dropna()

cache = {}
for fname, code in FREQ_CODES.items():
    cache[fname] = {lb: _resample(s, code) for lb, s in daily.items()
                    if len(_resample(s, code)) >= 2}

def global_earliest(freq):
    return min(v.index[0] for v in cache[freq].values())

# ══════════════════════════════════════════════
#  3. NORMALISE HELPER  (fast — no extra work)
# ══════════════════════════════════════════════
def get_normalised(label, freq, base_date, end_date, base_value):
    """Return (series|None, reason|None)."""
    rs = cache[freq]
    if label not in rs:
        return None, "no data"
    s  = rs[label]
    bd = pd.Timestamp(base_date)
    ed = pd.Timestamp(end_date)
    if s.index[0] > bd:
        return None, f"starts {s.index[0].strftime('%Y-%m')}"
    clipped = s[(s.index >= bd) & (s.index <= ed)]
    if clipped.empty:
        return None, "no data in range"
    base = clipped.iloc[0]
    if base == 0:
        return None, "base=0"
    return clipped / base * base_value, None

def parse_date(text):
    text = text.strip()
    for fmt in ("%Y-%m", "%Y-%m-%d", "%Y/%m", "%Y/%m/%d"):
        try: return pd.to_datetime(text, format=fmt)
        except ValueError: pass
    try: return pd.to_datetime(text)
    except: return None

# ══════════════════════════════════════════════
#  4. STATE
# ══════════════════════════════════════════════
_init_freq = RESAMPLE_FREQ
_ge        = global_earliest(_init_freq)

state = {
    "freq":       _init_freq,
    "base_date":  _ge,
    "end_date":   pd.Timestamp(FETCH_END),
    "base_value": BASE_VALUE,
    "active":     {lb: (lb in DEFAULT_ON) for lb in labels_list},
    "_dirty":     False,   # guard against double-redraws from CheckButtons
}

# ══════════════════════════════════════════════
#  5. FIGURE
# ══════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    BG_CHART,
    "text.color":        TXT_MAIN,
    "axes.labelcolor":   TXT_MAIN,
    "xtick.color":       TXT_DIM,
    "ytick.color":       TXT_DIM,
    "axes.edgecolor":    SPINE_CLR,
})

fig = plt.figure(figsize=(20, 11))

# axes
ax_chart  = fig.add_axes([0.25, 0.21, 0.73, 0.69])
ax_checks = fig.add_axes([0.005, 0.19, 0.235, 0.75])
ax_title  = fig.add_axes([0.25,  0.91, 0.73,  0.07]); ax_title.axis("off")

for ax in (ax_chart, ax_checks, ax_title):
    ax.set_facecolor(BG_PANEL)
ax_chart.set_facecolor(BG_CHART)

ax_checks.set_title("Select Indices", fontsize=12, pad=6,
                     color=TXT_MAIN, fontweight="bold")
ax_checks.tick_params(left=False, bottom=False,
                      labelleft=False, labelbottom=False)
for sp in ax_checks.spines.values():
    sp.set_color(SPINE_CLR)

# ── control row 1  (y≈0.125) ──
y1 = 0.128

def mk_lbl(rect, text, fs=8.5):
    a = fig.add_axes(rect); a.axis("off")
    a.set_facecolor(BG_CTRL)
    a.text(1.0, 0.5, text, ha="right", va="center",
           fontsize=fs, color=TXT_MAIN)
    return a

def mk_tb(rect, initial):
    a = fig.add_axes(rect)
    a.set_facecolor(BG_CTRL)
    tb = TextBox(a, "", initial=initial,
                 color=BTN_DEF, hovercolor=BTN_HOV)
    tb.label.set_color(TXT_MAIN)
    tb.text_disp.set_color(TXT_MAIN)
    return tb

mk_lbl([0.25, y1, 0.095, 0.038], "Base date (YYYY-MM):")
tb_date  = mk_tb([0.348, y1+0.004, 0.095, 0.030], _ge.strftime("%Y-%m"))

mk_lbl([0.455, y1, 0.065, 0.038], "End date:")
tb_edate = mk_tb([0.523, y1+0.004, 0.095, 0.030],
                 pd.Timestamp(FETCH_END).strftime("%Y-%m"))

mk_lbl([0.632, y1, 0.065, 0.038], "Base value:")
tb_bval  = mk_tb([0.700, y1+0.004, 0.075, 0.030], str(BASE_VALUE))

ax_info = fig.add_axes([0.782, y1, 0.21, 0.038]); ax_info.axis("off")
ax_info.set_facecolor(BG_CTRL)
info_txt = ax_info.text(0, 0.5, "", va="center",
                        fontsize=7.5, color=TXT_DIM)

# ── control row 2  (y≈0.073) ──
y2 = 0.073
freq_btns = []
for i, fname in enumerate(FREQ_OPTIONS):
    a = fig.add_axes([0.25 + i*0.082, y2, 0.075, 0.038])
    a.set_facecolor(BG_CTRL)
    c = BTN_ACT if fname == RESAMPLE_FREQ else BTN_DEF
    b = Button(a, fname, color=c, hovercolor=BTN_HOV)
    b.label.set_fontsize(9); b.label.set_color(TXT_MAIN)
    freq_btns.append(b)

def mk_btn(rect, label, color):
    a = fig.add_axes(rect); a.set_facecolor(BG_CTRL)
    b = Button(a, label, color=color, hovercolor=BTN_HOV)
    b.label.set_fontsize(9); b.label.set_color(TXT_MAIN)
    return b

btn_all   = mk_btn([0.503, y2, 0.082, 0.038], "Select All",  BTN_ALL)
btn_clear = mk_btn([0.590, y2, 0.082, 0.038], "Clear All",   BTN_CLR)
btn_save  = mk_btn([0.880, y2, 0.095, 0.038], "Save PNG",    BTN_SAV)

fig.text(0.01, 0.01,
         f"Data: Yahoo Finance · Generated {date.today()} · All prices in JPY",
         fontsize=6.5, color=TXT_DIM)

# ══════════════════════════════════════════════
#  6. CHECKBOXES  (large, dark-mode styled)
# ══════════════════════════════════════════════
check_init = [state["active"][lb] for lb in labels_list]
chk = CheckButtons(ax_checks, labels_list, check_init,
                   label_props={"fontsize": [11]*len(labels_list),
                                "color":    [TXT_MAIN]*len(labels_list)},
                   frame_props={"edgecolor": [SPINE_CLR]*len(labels_list),
                                "facecolor": [BG_PANEL]*len(labels_list)},
                   check_props={"color":     [COLORS[i % len(COLORS)]
                                              for i in range(len(labels_list))]})

# ══════════════════════════════════════════════
#  7. REDRAW  (optimised)
# ══════════════════════════════════════════════
LINE_WIDTH = 3.0   # 1.5× the previous ~2.0

def redraw():
    ax_chart.cla()
    ax_chart.set_facecolor(BG_CHART)
    for sp in ax_chart.spines.values():
        sp.set_color(SPINE_CLR)
    ax_chart.grid(axis="y", linestyle="--", lw=0.7, color=GRID_MAJ, alpha=0.9)
    ax_chart.grid(axis="x", linestyle=":",  lw=0.5, color=GRID_MIN, alpha=0.7)
    ax_chart.tick_params(colors=TXT_DIM)

    freq = state["freq"]
    bd   = state["base_date"]
    ed   = state["end_date"]
    bv   = state["base_value"]

    plotted, skipped = [], []

    for i, label in enumerate(labels_list):
        if not state["active"][label]: continue
        ns, reason = get_normalised(label, freq, bd, ed, bv)
        if ns is None:
            skipped.append(f"{label} [{reason}]"); continue
        color = COLORS[i % len(COLORS)]
        ax_chart.plot(ns.index, ns.values, label=label,
                      color=color, linewidth=LINE_WIDTH, alpha=0.92)
        ax_chart.annotate(f"{ns.iloc[-1]:.1f}",
                          xy=(ns.index[-1], ns.iloc[-1]),
                          xytext=(5, 0), textcoords="offset points",
                          fontsize=7.5, color=color, fontweight="bold", va="center")
        plotted.append(label)

    ax_chart.axhline(bv, color=BASE_LINE, lw=1.1, linestyle="--", alpha=0.7)
    ax_chart.set_ylabel(f"Index  ({pd.Timestamp(bd).strftime('%Y-%m')} = {bv:.4g})",
                        fontsize=9.5, color=TXT_MAIN)
    ax_chart.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # adaptive x-axis
    span = 365
    if plotted:
        ns0, _ = get_normalised(plotted[0], freq, bd, ed, bv)
        if ns0 is not None:
            span = (ns0.index[-1] - ns0.index[0]).days

    if span <= 180:
        ax_chart.xaxis.set_major_locator(mdates.MonthLocator())
        ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    elif span <= 730:
        ax_chart.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    elif span <= 1825:
        ax_chart.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]))
        ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    else:
        ax_chart.xaxis.set_major_locator(mdates.YearLocator())
        ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.setp(ax_chart.get_xticklabels(), rotation=35, fontsize=8.5,
             ha="right", color=TXT_DIM)
    plt.setp(ax_chart.get_yticklabels(), fontsize=8.5, color=TXT_DIM)

    if plotted:
        leg = ax_chart.legend(loc="upper left", fontsize=8.5,
                              framealpha=0.5, edgecolor=SPINE_CLR,
                              fancybox=True,
                              labelcolor=TXT_MAIN,
                              facecolor=BG_PANEL)

    # title
    ax_title.cla(); ax_title.axis("off"); ax_title.set_facecolor(BG_PANEL)
    ax_title.text(0, 0.78,
                  "G20 Major Markets: Equities · Gold · Silver · REITs",
                  fontsize=13, fontweight="bold", color=TXT_MAIN)
    ax_title.text(0, 0.18,
                  f"JPY-denominated · {freq} · "
                  f"Base {pd.Timestamp(bd).strftime('%Y-%m')} = {bv:.4g} · "
                  "Source: Yahoo Finance",
                  fontsize=8.5, color=TXT_DIM)

    # info strip
    if skipped:
        msg   = "No data at base date: " + "  ·  ".join(skipped)
        color = WARN_CLR
    else:
        msg   = f"{pd.Timestamp(bd).strftime('%Y-%m')} → {pd.Timestamp(ed).strftime('%Y-%m')}  |  All selected series plotted"
        color = TXT_DIM
    info_txt.set_text(msg); info_txt.set_color(color)

    fig.canvas.draw_idle()

# ══════════════════════════════════════════════
#  8. CALLBACKS
# ══════════════════════════════════════════════
def on_check(clicked):
    state["active"][clicked] = not state["active"][clicked]
    redraw()
chk.on_clicked(on_check)

def on_date_submit(text):
    p = parse_date(text)
    if p is None:
        info_txt.set_text("Invalid base date — use YYYY-MM")
        info_txt.set_color(ERR_CLR); fig.canvas.draw_idle(); return
    if p > state["end_date"]:
        p = state["end_date"]
        tb_date.set_val(p.strftime("%Y-%m"))
    state["base_date"] = p; redraw()
tb_date.on_submit(on_date_submit)

def on_edate_submit(text):
    p = parse_date(text)
    if p is None:
        info_txt.set_text("Invalid end date — use YYYY-MM")
        info_txt.set_color(ERR_CLR); fig.canvas.draw_idle(); return
    if p < state["base_date"]:
        p = state["base_date"]
        tb_edate.set_val(p.strftime("%Y-%m"))
    state["end_date"] = p; redraw()
tb_edate.on_submit(on_edate_submit)

def on_bval_submit(text):
    try:
        v = float(text.strip())
        if v > 0: state["base_value"] = v; redraw()
    except ValueError: pass
tb_bval.on_submit(on_bval_submit)

def make_freq_cb(fname):
    def cb(event):
        if state["freq"] == fname: return   # no-op if already selected
        state["freq"] = fname
        for b, fn in zip(freq_btns, FREQ_OPTIONS):
            c = BTN_ACT if fn == fname else BTN_DEF
            b.color = c; b.ax.set_facecolor(c)
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
    fn = (f"g20_{state['freq'].lower()}"
          f"_{pd.Timestamp(state['base_date']).strftime('%Y%m')}"
          f"_{pd.Timestamp(state['end_date']).strftime('%Y%m')}.png")
    fig.savefig(fn, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {fn}")

btn_all.on_clicked(select_all)
btn_clear.on_clicked(clear_all)
btn_save.on_clicked(save_png)

# ══════════════════════════════════════════════
#  9. INITIAL DRAW
# ══════════════════════════════════════════════
redraw()
plt.show()