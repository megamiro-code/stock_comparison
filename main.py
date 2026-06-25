"""
G20 Major Markets — Interactive Chart  v6
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
FETCH_START   = "1990-01-01"
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
BTN_JPY   = "#2a4a1a"   # active currency button colour (JPY = greenish)
BTN_USD   = "#1a2a4a"   # active currency button colour (USD = blueish)
WARN_CLR  = "#ffaa44"
ERR_CLR   = "#ff6666"
BASE_LINE = "#556688"

COLORS = [
    "#4db8ff",   #  1 sky blue        — Nikkei 225
    "#ff4444",   #  2 vivid red       — TOPIX
    "#44ff88",   #  3 mint green      — S&P 500
    "#ffaa00",   #  4 amber           — DAX
    "#cc77ff",   #  5 violet          — FTSE 100
    "#ff77aa",   #  6 rose pink       — CAC 40
    "#00e5cc",   #  7 cyan-teal       — Shanghai
    "#ffe066",   #  8 bright yellow   — SENSEX
    "#ff7733",   #  9 orange          — KOSPI
    "#66ffcc",   # 10 aquamarine      — IBOVESPA
    "#aabbff",   # 11 periwinkle      — ASX 200
    "#ffcc44",   # 12 golden yellow   — TSX
    "#ee44ff",   # 13 magenta         — FTSE MIB
    "#99ff44",   # 14 lime green      — Gold
    "#ff99cc",   # 15 soft pink       — Silver
    "#44ddff",   # 16 light cyan      — J-REIT
    "#ffbb77",   # 17 peach           — US REIT
    "#bb99ff",   # 18 lavender        — eMAXIS Slim
    "#ffffff",   # 19 white           — spare
]

TICKERS = {
    "Nikkei 225 (Japan)":     ("^N225",      "JPY"),
    "TOPIX (Japan)":          ("1306.T",     "JPY"),
    "S&P 500 (USA)":          ("^GSPC",      "USD"),
    "DAX (Germany)":          ("^GDAXI",     "EUR"),
    "FTSE 100 (UK)":          ("^FTSE",      "GBP"),
    "CAC 40 (France)":        ("^FCHI",      "EUR"),
    "Shanghai Comp. (China)": ("000001.SS",  "CNY"),
    "SENSEX (India)":         ("^BSESN",     "INR"),
    "KOSPI (South Korea)":    ("^KS11",      "KRW"),
    "IBOVESPA (Brasil)":      ("^BVSP",      "BRL"),
    "ASX 200 (Australia)":    ("^AXJO",      "AUD"),
    "TSX (Canada)":           ("^GSPTSE",    "CAD"),
    "FTSE MIB (Italy)":       ("FTSEMIB.MI", "EUR"),
    "Gold (per g)":           ("GC=F",       "USD"),
    "Silver (per g)":         ("SI=F",       "USD"),
    "J-REIT":                 ("1343.T",     "JPY"),
    "US REIT (VNQ)":          ("VNQ",        "USD"),
    "eMAXIS Slim All-World":  ("0331418A.T", "JPY"),
}

# All FX quoted as XXX/JPY  (1 foreign unit = N yen)
FX_MAP = {
    "USD": "JPY=X",    "EUR": "EURJPY=X", "GBP": "GBPJPY=X",
    "AUD": "AUDJPY=X", "CAD": "CADJPY=X", "CNY": "CNYJPY=X",
    "INR": "INRJPY=X", "KRW": "KRWJPY=X", "BRL": "BRLJPY=X",
    "JPY": None,
}

TROY_TO_G    = 31.1035
DEFAULT_ON   = {"Nikkei 225 (Japan)", "S&P 500 (USA)"}
FREQ_OPTIONS = ["Daily", "Weekly", "Monthly"]
FREQ_CODES   = {"Daily": "D", "Weekly": "W-FRI", "Monthly": "ME"}
CCY_OPTIONS  = ["JPY", "USD"]

# ══════════════════════════════════════════════
#  DATA CLEANING
# ══════════════════════════════════════════════
STITCH_THRESHOLD = 0.30

def _remove_spikes(s: pd.Series) -> pd.Series:
    if len(s) < 3:
        return s
    vals  = s.values.copy().astype(float)
    dates = s.index
    mask  = np.ones(len(vals), dtype=bool)
    for i in range(1, len(vals) - 1):
        prev, cur, nxt = vals[i-1], vals[i], vals[i+1]
        if prev <= 0 or cur <= 0 or nxt <= 0:
            continue
        r1 = (cur - prev) / prev
        r2 = (nxt - cur)  / cur
        if abs(r1) > 0.40 and abs(r2) > 0.40 and (r1 * r2 < 0):
            mask[i] = False
    if (~mask).sum():
        print(f"    [CLEAN] removed {(~mask).sum()} spike(s)")
    return pd.Series(vals[mask], index=dates[mask])

def _stitch_level_shifts(s: pd.Series, label: str) -> pd.Series:
    if len(s) < 10:
        return s
    s    = s.copy()
    vals = s.values.astype(float)
    n    = len(vals)
    W    = 30
    for _ in range(10):
        pct   = np.diff(vals) / np.abs(vals[:-1])
        pct   = np.where(np.isfinite(pct), pct, 0.0)
        jumps = np.where(np.abs(pct) > STITCH_THRESHOLD)[0]
        if not len(jumps):
            break
        fixed = False
        for idx in jumps:
            b_seg = vals[max(0, idx-W) : idx+1]
            a_seg = vals[idx+1 : min(n-1, idx+1+W)+1]
            if len(b_seg) < 3 or len(a_seg) < 3:
                continue
            def _vol(seg):
                lr = np.diff(np.log(np.abs(seg)+1e-9))
                return float(np.std(lr)) if len(lr) > 1 else 0.0
            vol_max  = max(_vol(b_seg), _vol(a_seg), 1e-9)
            jump_mag = abs(pct[idx])
            if jump_mag / vol_max > 8.0:
                factor = vals[idx+1] / vals[idx]
                vals[:idx+1] *= factor
                print(f"    [STITCH] {label}: {pct[idx]*100:.1f}% on "
                      f"{s.index[idx+1].date()} ×{factor:.5g}")
                fixed = True
                break
        if not fixed:
            break
    return pd.Series(vals, index=s.index)

# ══════════════════════════════════════════════
#  1. DOWNLOAD
# ══════════════════════════════════════════════
print("Downloading price data …")

# Fallback tickers tried when the primary ticker returns no data
TICKER_FALLBACKS = {
    "0331418A.T": ["2559.T", "7T29.T"],   # オルカン: MAXIS全世界 or SBI全世界
}

all_t  = list({t for t, _ in TICKERS.values()})
# also pre-fetch any fallback tickers
for fallbacks in TICKER_FALLBACKS.values():
    all_t.extend(fallbacks)
all_t  = list(set(all_t))

fx_t   = list({FX_MAP[c] for _, c in TICKERS.values() if c != "JPY"})

raw_p  = yf.download(all_t, start=FETCH_START, end=FETCH_END,
                     progress=True,  auto_adjust=True)["Close"]
raw_fx = yf.download(fx_t,  start=FETCH_START, end=FETCH_END,
                     progress=False, auto_adjust=True)["Close"]

if isinstance(raw_p,  pd.Series): raw_p  = raw_p.to_frame(name=all_t[0])
if isinstance(raw_fx, pd.Series): raw_fx = raw_fx.to_frame(name=fx_t[0])

def _resolve_ticker(primary: str) -> str:
    """Return the first ticker (primary or fallback) that has data."""
    candidates = [primary] + TICKER_FALLBACKS.get(primary, [])
    for t in candidates:
        if t in raw_p.columns and not raw_p[t].dropna().empty:
            if t != primary:
                print(f"    [FALLBACK] {primary} → using {t}")
            return t
    return primary   # will be reported as missing downstream

# ── build per-currency daily series ──
# daily_local : price in original currency (no FX conversion)
# daily_jpy   : price converted to JPY
# daily_usd   : price converted to USD
daily_local = {}   # {label: (series_in_native_ccy, native_ccy_str)}
daily_jpy   = {}
daily_usd   = {}

# We need USD/JPY to go JPY→USD
def _get_usdjpy():
    col = "JPY=X"
    if col in raw_fx.columns:
        return raw_fx[col].dropna()
    return None

usdjpy_series = _get_usdjpy()   # 1 USD = N JPY

for label, (ticker, ccy) in TICKERS.items():
    ticker = _resolve_ticker(ticker)   # use fallback if primary missing
    if ticker not in raw_p.columns:
        print(f"  [SKIP] {label} — ticker not found"); continue
    s_raw = raw_p[ticker].dropna()
    if s_raw.empty:
        print(f"  [SKIP] {label} — empty"); continue

    # ── JPY series ──
    if ccy == "JPY":
        s_jpy = s_raw.copy()
    else:
        fx_col = FX_MAP[ccy]
        if fx_col not in raw_fx.columns:
            print(f"  [SKIP] {label} — FX missing"); continue
        fx = raw_fx[fx_col].reindex(s_raw.index, method="ffill").dropna()
        s_jpy = s_raw.reindex(fx.index).dropna() * fx   # local → JPY

    # gold/silver: /oz → /g
    if label in ("Gold (per g)", "Silver (per g)"):
        s_jpy = s_jpy / TROY_TO_G

    s_jpy = _remove_spikes(s_jpy)
    s_jpy = _stitch_level_shifts(s_jpy, label)
    if len(s_jpy) < 5:
        print(f"  [SKIP] {label} — insufficient data"); continue

    # ── USD series  (JPY ÷ USDJPY) ──
    if usdjpy_series is not None:
        fx_usd = usdjpy_series.reindex(s_jpy.index, method="ffill").dropna()
        s_usd  = (s_jpy.reindex(fx_usd.index).dropna() / fx_usd)
    else:
        s_usd = s_jpy / 150.0   # fallback constant rate

    daily_jpy[label] = s_jpy
    daily_usd[label] = s_usd
    print(f"  [OK]  {label}: {s_jpy.index[0].date()} — {s_jpy.index[-1].date()}"
          f" ({len(s_jpy)} days)")

if not daily_jpy:
    sys.exit("No data downloaded.")

labels_list = list(daily_jpy.keys())

# ══════════════════════════════════════════════
#  2. RESAMPLE CACHE  (all freq × all ccy)
# ══════════════════════════════════════════════
def _resample(s, code):
    if code == "D":     return s.copy()
    if code == "W-FRI": return s.resample("W-FRI").last().dropna()
    return s.resample("ME").last().dropna()

# cache[ccy][freq][label] = resampled series
cache = {}
for ccy_key, daily_dict in [("JPY", daily_jpy), ("USD", daily_usd)]:
    cache[ccy_key] = {}
    for fname, code in FREQ_CODES.items():
        cache[ccy_key][fname] = {
            lb: _resample(s, code)
            for lb, s in daily_dict.items()
            if len(_resample(s, code)) >= 2
        }

def global_earliest(ccy, freq):
    return min(v.index[0] for v in cache[ccy][freq].values())

# ══════════════════════════════════════════════
#  3. NORMALISE
# ══════════════════════════════════════════════
def get_normalised(label, ccy, freq, base_date, end_date, base_value):
    rs = cache[ccy][freq]
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
_init_ccy  = "JPY"
_init_freq = RESAMPLE_FREQ
_ge = pd.Timestamp("2024-01-01")

state = {
    "ccy":        _init_ccy,
    "freq":       _init_freq,
    "base_date":  _ge,
    "end_date":   pd.Timestamp(FETCH_END),
    "base_value": BASE_VALUE,
    "active":     {lb: (lb in DEFAULT_ON) for lb in labels_list},
}

# ══════════════════════════════════════════════
#  5. FIGURE
# ══════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor":   BG_CHART,
    "text.color":       TXT_MAIN,
    "axes.labelcolor":  TXT_MAIN,
    "xtick.color":      TXT_DIM,
    "ytick.color":      TXT_DIM,
    "axes.edgecolor":   SPINE_CLR,
})

# Scale figure height so each checkbox row gets ~0.40 inch regardless of count
# Figure height scales with ticker count so every checkbox row fits
N_LABELS = len(labels_list)
FIG_H    = max(9.0, N_LABELS * 0.40 + 3.0)   # 0.40 inch per row, min 9 in

# All axes positions are fixed fractions — simple and reliable
y1 = 0.128   # control row 1: dates / base value
y2 = 0.073   # control row 2: freq / ccy / util buttons

fig = plt.figure(figsize=(18, FIG_H))

ax_chart  = fig.add_axes([0.24, 0.22, 0.73, 0.70])
ax_checks = fig.add_axes([0.005, 0.18, 0.20, 0.76])
ax_title  = fig.add_axes([0.25,  0.91, 0.73, 0.07])
ax_title.axis("off")

for ax in (ax_chart, ax_checks, ax_title):
    ax.set_facecolor(BG_PANEL)
ax_chart.set_facecolor(BG_CHART)

ax_checks.set_title("Select Indices", fontsize=12, pad=6,
                     color=TXT_MAIN, fontweight="bold")
ax_checks.tick_params(left=False, bottom=False,
                      labelleft=False, labelbottom=False)
for sp in ax_checks.spines.values():
    sp.set_color(SPINE_CLR)

# ── helpers ──
def mk_lbl(rect, text, fs=8.5):
    a = fig.add_axes(rect); a.axis("off"); a.set_facecolor(BG_CTRL)
    a.text(1.0, 0.5, text, ha="right", va="center", fontsize=fs, color=TXT_MAIN)
    return a

def mk_tb(rect, initial):
    a = fig.add_axes(rect); a.set_facecolor(BG_CTRL)
    tb = TextBox(a, "", initial=initial, color=BTN_DEF, hovercolor=BTN_HOV)
    tb.label.set_color(TXT_MAIN); tb.text_disp.set_color(TXT_MAIN)
    return tb

def mk_btn(rect, label, color):
    a = fig.add_axes(rect); a.set_facecolor(BG_CTRL)
    b = Button(a, label, color=color, hovercolor=BTN_HOV)
    b.label.set_fontsize(9); b.label.set_color(TXT_MAIN)
    return b

# ── control row 1: dates + base value ──
mk_lbl([0.25, y1, 0.095, 0.038], "Base date (YYYY-MM):")
tb_date  = mk_tb([0.348, y1+0.004, 0.095, 0.030], _ge.strftime("%Y-%m"))

mk_lbl([0.455, y1, 0.065, 0.038], "End date:")
tb_edate = mk_tb([0.523, y1+0.004, 0.095, 0.030],
                 pd.Timestamp(FETCH_END).strftime("%Y-%m"))

mk_lbl([0.632, y1, 0.065, 0.038], "Base value:")
tb_bval  = mk_tb([0.700, y1+0.004, 0.075, 0.030], str(BASE_VALUE))

ax_info = fig.add_axes([0.782, y1, 0.21, 0.038]); ax_info.axis("off")
ax_info.set_facecolor(BG_CTRL)
info_txt = ax_info.text(0, 0.5, "", va="center", fontsize=7.5, color=TXT_DIM)

# ── control row 2: freq + currency + util buttons ──

# Frequency buttons
freq_btns = []
for i, fname in enumerate(FREQ_OPTIONS):
    a = fig.add_axes([0.25 + i*0.075, y2, 0.068, 0.038])
    a.set_facecolor(BG_CTRL)
    c = BTN_ACT if fname == RESAMPLE_FREQ else BTN_DEF
    b = Button(a, fname, color=c, hovercolor=BTN_HOV)
    b.label.set_fontsize(9); b.label.set_color(TXT_MAIN)
    freq_btns.append(b)

# Currency toggle buttons  (JPY / USD)
ccy_btns = []
ccy_start_x = 0.25 + len(FREQ_OPTIONS)*0.075 + 0.012
for i, cname in enumerate(CCY_OPTIONS):
    c = BTN_JPY if cname == _init_ccy else BTN_DEF
    b = mk_btn([ccy_start_x + i*0.058, y2, 0.052, 0.038], cname, c)
    b.label.set_fontweight("bold")
    ccy_btns.append(b)

# Util buttons
btn_all   = mk_btn([0.680, y2, 0.075, 0.038], "Select All", BTN_ALL)
btn_clear = mk_btn([0.760, y2, 0.075, 0.038], "Clear All",  BTN_CLR)
btn_save  = mk_btn([0.880, y2, 0.095, 0.038], "Save PNG",   BTN_SAV)

fig.text(0.01, 0.005,
         f"Data: Yahoo Finance · Generated {date.today()}",
         fontsize=6.5, color=TXT_DIM)

# ══════════════════════════════════════════════
#  6. CHECKBOXES
# ══════════════════════════════════════════════
check_init = [state["active"][lb] for lb in labels_list]
chk = CheckButtons(ax_checks, labels_list, check_init,
                   label_props={"fontsize": [11]*len(labels_list),
                                "color":    [TXT_MAIN]*len(labels_list)},
                   frame_props={"edgecolor": [SPINE_CLR]*len(labels_list),
                                "facecolor": [BG_PANEL]*len(labels_list)},
                   check_props={"color": [COLORS[i % len(COLORS)]
                                          for i in range(len(labels_list))]})

# ══════════════════════════════════════════════
#  7. REDRAW
# ══════════════════════════════════════════════
LINE_WIDTH = 3.0

def redraw():
    ax_chart.cla()
    ax_chart.set_facecolor(BG_CHART)
    for sp in ax_chart.spines.values(): sp.set_color(SPINE_CLR)
    ax_chart.grid(axis="y", linestyle="--", lw=0.7, color=GRID_MAJ, alpha=0.9)
    ax_chart.grid(axis="x", linestyle=":",  lw=0.5, color=GRID_MIN, alpha=0.7)
    ax_chart.tick_params(colors=TXT_DIM)

    ccy  = state["ccy"]
    freq = state["freq"]
    bd   = state["base_date"]
    ed   = state["end_date"]
    bv   = state["base_value"]

    ccy_symbol = "¥" if ccy == "JPY" else "$"
    plotted, skipped = [], []

    for i, label in enumerate(labels_list):
        if not state["active"][label]: continue
        ns, reason = get_normalised(label, ccy, freq, bd, ed, bv)
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
    ax_chart.set_ylabel(
        f"Index  ({pd.Timestamp(bd).strftime('%Y-%m')} = {bv:.4g})  [{ccy_symbol}{ccy}]",
        fontsize=9.5, color=TXT_MAIN)
    ax_chart.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # adaptive x-axis
    span = 365
    if plotted:
        ns0, _ = get_normalised(plotted[0], ccy, freq, bd, ed, bv)
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
        ax_chart.legend(loc="upper left", fontsize=8.5, framealpha=0.5,
                        edgecolor=SPINE_CLR, fancybox=True,
                        labelcolor=TXT_MAIN, facecolor=BG_PANEL)

    # title
    ax_title.cla(); ax_title.axis("off"); ax_title.set_facecolor(BG_PANEL)
    ax_title.text(0, 0.78,
                  "G20 Major Markets: Equities · Gold · Silver · REITs",
                  fontsize=13, fontweight="bold", color=TXT_MAIN)
    ax_title.text(0, 0.18,
                  f"{ccy}-denominated · {freq} · "
                  f"Base {pd.Timestamp(bd).strftime('%Y-%m')} = {bv:.4g} · "
                  "Source: Yahoo Finance",
                  fontsize=8.5, color=TXT_DIM)

    # info strip
    if skipped:
        msg   = "No data at base date: " + "  ·  ".join(skipped)
        color = WARN_CLR
    else:
        msg   = (f"{pd.Timestamp(bd).strftime('%Y-%m')} → "
                 f"{pd.Timestamp(ed).strftime('%Y-%m')}  |  "
                 f"All selected series plotted  [{ccy}]")
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
    if p > state["end_date"]: p = state["end_date"]; tb_date.set_val(p.strftime("%Y-%m"))
    state["base_date"] = p; redraw()
tb_date.on_submit(on_date_submit)

def on_edate_submit(text):
    p = parse_date(text)
    if p is None:
        info_txt.set_text("Invalid end date — use YYYY-MM")
        info_txt.set_color(ERR_CLR); fig.canvas.draw_idle(); return
    if p < state["base_date"]: p = state["base_date"]; tb_edate.set_val(p.strftime("%Y-%m"))
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
        if state["freq"] == fname: return
        state["freq"] = fname
        for b, fn in zip(freq_btns, FREQ_OPTIONS):
            c = BTN_ACT if fn == fname else BTN_DEF
            b.color = c; b.ax.set_facecolor(c)
        redraw()
    return cb
for btn, fname in zip(freq_btns, FREQ_OPTIONS):
    btn.on_clicked(make_freq_cb(fname))

def make_ccy_cb(cname):
    def cb(event):
        if state["ccy"] == cname: return
        state["ccy"] = cname
        # update button colours
        for b, cn in zip(ccy_btns, CCY_OPTIONS):
            active_c = BTN_JPY if cn == "JPY" else BTN_USD
            c = active_c if cn == cname else BTN_DEF
            b.color = c; b.ax.set_facecolor(c)
        # recalculate earliest available base date for this ccy
        ge = global_earliest(cname, state["freq"])
        if state["base_date"] < ge:
            state["base_date"] = ge
            tb_date.set_val(ge.strftime("%Y-%m"))
        redraw()
    return cb
for btn, cname in zip(ccy_btns, CCY_OPTIONS):
    btn.on_clicked(make_ccy_cb(cname))

def select_all(event):
    for i, lb in enumerate(labels_list):
        if not state["active"][lb]: chk.set_active(i)
def clear_all(event):
    for i, lb in enumerate(labels_list):
        if state["active"][lb]: chk.set_active(i)
def save_png(event):
    fn = (f"g20_{state['ccy']}_{state['freq'].lower()}"
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