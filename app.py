"""
app.py - Community Mental-Wellbeing Signal Tracker (Streamlit dashboard)
=========================================================================
Regional Early-Warning and Stressor Monitoring System - Sri Lanka.

DEMONSTRATION MODE: every number shown here is SYNTHETIC.

How the pieces fit together
---------------------------
    generate_data.py  -> produces synthetic AGGREGATED data
    privacy.py        -> suppresses every province-week-theme cell with < 10 responses
    analysis.py       -> (optional) early-warning detection
    app.py (this)     -> loads data, applies privacy FIRST, then draws charts

Integration contract (see INTEGRATION CONTRACT below). If your generate_data.py or
analysis.py use different function names, change the two NAME constants only.
The app never displays raw text or individual records: record-level columns
are reduced to counts immediately after loading, and every chart and table
reads from the privacy-filtered data only.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import privacy  # k-anonymity-style small-cell suppression (k = 10)

# Optional modules. If a module is missing or has other function names, the app
# falls back to small built-in versions so the dashboard still runs.
try:
    import generate_data
except Exception:  # noqa: BLE001 - any import problem -> use fallback
    generate_data = None
try:
    import analysis
except Exception:  # noqa: BLE001
    analysis = None


# =========================================================================
# 1. SETTINGS
# =========================================================================

# ---- INTEGRATION CONTRACT (edit these names to match your own modules) ----
# generate_data.<GENERATE_FUNCTION_NAME>() must return a DataFrame that is EITHER
#   (a) aggregated: province, week, theme, response_count, percentage_share
#   (b) record-level with at least: province, week, theme
#       (it is counted into aggregates straight away; other columns are dropped).
GENERATE_FUNCTION_NAME = "generate_synthetic_data"
# analysis.<ALERT_FUNCTION_NAME>(safe_df, current_week) must return a DataFrame
# with the columns listed in ALERT_COLUMNS below.
ALERT_FUNCTION_NAME = "detect_alerts"
ALERT_COLUMNS = ["province", "theme", "baseline_pct", "current_pct",
                 "change_pp", "response_count", "status"]

K = privacy.K_THRESHOLD  # minimum group size (10)

PROVINCES = ["Western", "Central", "Southern", "Northern", "Eastern",
             "North Western", "North Central", "Uva", "Sabaragamuwa"]
THEMES = ["Job and Money", "Exams and Studies", "Loneliness and Isolation",
          "Family and Relationships", "Other"]
NUM_WEEKS = 12

# Early-warning rules used by the built-in detector
BASELINE_WEEKS = 4      # baseline = mean of up to 4 weeks before the current week
MIN_BASELINE_WEEKS = 2  # need at least 2 visible baseline weeks
WATCH_PP = 3.0          # rise of >= 3 percentage points -> WATCH
ALERT_PP = 5.0          # rise of >= 5 points AND statistically unusual -> ALERT
STD_FLOOR = 1.0         # minimum spread (pp) so very flat baselines don't over-react

# Colours (navy and blue scheme)
NAVY, BLUE, LIGHT_BLUE, BG = "#0F172A", "#1F5FA8", "#8DC3EE", "#F8FAFC"
INDIGO = "#6366F1"  # primary accent
THEME_COLOURS = {
    "Job and Money": "#0B2545",
    "Exams and Studies": "#1F5FA8",
    "Loneliness and Isolation": "#4FA3E0",
    "Family and Relationships": "#8DC3EE",
    "Other": "#A8B3C5",
}
LINE_COLOURS = ["#0B2545", "#1F78C8", "#4FA3E0", "#2A9D8F", "#E9A23B",
                "#C8553D", "#5B6B82", "#8DC3EE", "#7A4EAB"]

# Sidebar navigation: (page name, Material icon)
NAV_ITEMS = [("Overview", ":material/dashboard:"),
             ("Stressor Analysis", ":material/bar_chart:"),
             ("Weekly Trends", ":material/trending_up:"),
             ("Regional Comparison", ":material/grid_view:"),
             ("Early Warning", ":material/notifications:"),
             ("Privacy & Data Quality", ":material/shield:"),
             ("Methodology", ":material/menu_book:")]
PAGES = [name for name, _ in NAV_ITEMS]

# Small inline SVG icons (Lucide-style paths) used in cards and banners
ICON_PATHS = {
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "dollar": '<line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
    "warning": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
}


def svg(name: str, size: int = 18) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round">{ICON_PATHS[name]}</svg>')


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, button, input, select, textarea {{ font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif; }}
.stApp {{ background-color: {BG}; color: {NAVY}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppDeployButton"] {{ display: none; }}
.block-container {{ padding: 0 2rem 3rem 2rem; max-width: 100%; }}

/* ---------- Top bar and banner ---------- */
.topbar {{ background: #fff; border-bottom: 1px solid #E2E8F0; margin: 0 -2rem 1.5rem -2rem;
  padding: 1.1rem 4.5rem 1.1rem 2rem; display: flex; align-items: center;
  justify-content: space-between; gap: 1rem; flex-wrap: wrap; }}
.topbar-title {{ font-size: 1.3rem; font-weight: 600; color: {NAVY}; line-height: 1.3; }}
.topbar-sub {{ color: #64748B; font-size: .88rem; }}
.demo-pill {{ display: inline-flex; align-items: center; gap: .5rem; background: #FFFBEB; border: 1px solid #FDE68A;
  color: #92400E; border-radius: 999px; padding: .35rem .85rem; font-size: .78rem; }}
.demo-pill b {{ font-weight: 600; }}
.demo-pill .dot {{ width: 6px; height: 6px; border-radius: 50%; background: #D97706; }}
.demo-banner {{ display: flex; align-items: center; gap: .75rem; background: #FFFBEB; border: 1px solid #FDE68A;
  color: #92400E; border-radius: 10px; padding: .8rem 1rem; font-size: .92rem; line-height: 1.4; margin-bottom: 1.4rem; }}
.demo-banner svg {{ flex-shrink: 0; }}
.page-head {{ display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap;
  gap: .5rem; margin: .2rem 0 1rem 0; }}
.page-title {{ font-size: 1.1rem; font-weight: 600; color: {NAVY}; }}
.page-meta {{ color: #64748B; font-size: .82rem; }}
.section-note {{ color: #64748B; font-size: .88rem; }}

/* ---------- KPI cards ---------- */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.1rem; margin: 0 0 1.4rem 0; }}
.kpi-card {{ background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.1rem 1.25rem;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); display: flex; flex-direction: column;
  justify-content: space-between; min-height: 140px; }}
.kpi-card.primary {{ border-top: 3px solid {INDIGO}; box-shadow: 0 4px 14px rgba(99,102,241,.14); }}
.kpi-top {{ display: flex; justify-content: space-between; align-items: center; gap: .5rem; }}
.kpi-label {{ color: #334155; font-size: .88rem; font-weight: 500; }}
.kpi-icon {{ width: 32px; height: 32px; border-radius: 8px; background: #F1F5F9; color: #64748B;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
.kpi-icon.indigo {{ background: #EEF2FF; color: {INDIGO}; }}
.kpi-icon.coral {{ background: #FFF1F2; color: #E11D48; }}
.kpi-value {{ color: {NAVY}; font-size: 2rem; font-weight: 700; line-height: 1.2; margin-top: .7rem; }}
.kpi-value.text {{ font-size: 1.4rem; font-weight: 600; margin-top: .9rem; }}
.kpi-value .of {{ color: #94A3B8; font-size: 1rem; font-weight: 500; margin-left: .25rem; }}
.kpi-note {{ color: #64748B; font-size: .8rem; margin-top: .55rem; }}
.progress {{ height: 3px; background: #E2E8F0; border-radius: 3px; margin-top: .75rem; overflow: hidden; }}
.progress > div {{ height: 100%; background: {BLUE}; border-radius: 3px; }}
.pill {{ display: inline-flex; align-items: center; gap: .4rem; font-size: .72rem; font-weight: 600;
  padding: .15rem .6rem; border-radius: 999px; margin-top: .55rem; }}
.pill.coral {{ background: #FFF1F2; color: #BE123C; border: 1px solid #FECDD3; }}
.pill.grey {{ background: #F1F5F9; color: #64748B; border: 1px solid #E2E8F0; }}
.pill .dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}

/* ---------- Information callout ---------- */
.callout {{ display: flex; gap: 1rem; background: #fff; border: 1px solid #E2E8F0; border-left: 4px solid {INDIGO};
  border-radius: 12px; padding: 1.2rem 1.4rem; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
.callout-icon {{ width: 32px; height: 32px; border-radius: 8px; background: #EEF2FF; color: {INDIGO};
  display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
.callout-title {{ font-weight: 600; color: {NAVY}; margin-bottom: .5rem; }}
.callout-text {{ color: #334155; font-size: .95rem; line-height: 1.5; }}
.disclaimer {{ display: inline-flex; align-items: center; gap: .5rem; background: #F1F5F9; border-radius: 8px;
  padding: .4rem .8rem; color: #64748B; font-style: italic; font-size: .82rem; margin-top: .8rem; }}

/* ---------- Alert cards ---------- */
.alert-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 1.1rem; margin-top: .5rem; }}
.alert-card {{ background: #fff; border-radius: 12px; padding: 1rem 1.2rem; border: 1px solid #E2E8F0;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
.alert-card.alert {{ border-left: 5px solid #E11D48; }}
.alert-card.watch {{ border-left: 5px solid #F59E0B; }}
.badge {{ display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .05em;
  padding: .18rem .55rem; border-radius: 999px; color: #fff; }}
.badge.alert {{ background: #E11D48; }}
.badge.watch {{ background: #D97706; }}
.alert-title {{ color: {NAVY}; font-weight: 600; font-size: 1.02rem; margin: .5rem 0 .6rem 0; }}
.metric-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: .45rem .8rem; }}
.metric-label {{ color: #64748B; font-size: .72rem; text-transform: uppercase; letter-spacing: .03em; }}
.metric-value {{ color: {NAVY}; font-weight: 600; font-size: .98rem; }}
.sim-tag {{ color: #64748B; font-size: .72rem; margin-top: .7rem; font-style: italic; }}

/* ---------- Dark sidebar ---------- */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{ background: {NAVY}; }}
section[data-testid="stSidebar"][aria-expanded="true"] {{ min-width: 264px; max-width: 264px; width: 264px; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid #1E293B; }}
[data-testid="stSidebarHeader"] {{ height: 0; min-height: 0; padding: 0; margin: 0; }}
[data-testid="stSidebarCollapseButton"] {{ position: absolute; top: 1.05rem; right: .8rem; z-index: 10; }}
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"] svg {{ color: #94A3B8; }}
[data-testid="stSidebarUserContent"] {{ padding: 1.1rem .9rem 1.5rem .9rem; }}
.brand {{ display: flex; align-items: center; gap: .7rem; padding: .1rem .4rem 1.1rem .4rem; }}
.brand-icon {{ width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, {INDIGO}, #8B5CF6);
  color: #fff; display: flex; align-items: center; justify-content: center; }}
.brand-name {{ color: #fff; font-weight: 600; font-size: 1rem; }}
section[data-testid="stSidebar"] .stButton > button {{ width: 100%; justify-content: flex-start; background: transparent;
  border: none; border-left: 3px solid transparent; border-radius: 8px; color: #CBD5E1; font-weight: 500;
  font-size: .9rem; padding: .5rem .75rem; box-shadow: none; }}
section[data-testid="stSidebar"] .stButton > button > div {{ justify-content: flex-start; width: 100%; }}
section[data-testid="stSidebar"] .stButton > button:hover {{ background: #1E293B; color: #fff; }}
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {{
  background: #1E293B; color: #fff; border-left: 3px solid {INDIGO}; }}
.sidebar-spacer {{ height: 2.5rem; }}
.filter-head {{ color: #94A3B8; font-size: .75rem; font-weight: 500; padding-top: .35rem; }}
.st-key-reset_filters button {{ width: auto !important; margin-left: auto; color: #94A3B8 !important;
  font-size: .75rem !important; padding: .2rem 0 !important; border-left: none !important; background: transparent !important; }}
.filter-label {{ display: flex; justify-content: space-between; color: #E2E8F0; font-size: .82rem;
  font-weight: 500; margin: .9rem 0 .4rem 0; }}
.filter-label span {{ color: #64748B; font-size: .75rem; }}
/* chip toggles (st.pills) */
section[data-testid="stSidebar"] [data-testid="stBaseButton-pills"],
section[data-testid="stSidebar"] button[kind="pills"] {{ background: transparent; border: 1px solid #334155;
  color: #CBD5E1; border-radius: 999px; font-size: .72rem; padding: .05rem .65rem; min-height: 1.75rem; }}
section[data-testid="stSidebar"] [data-testid="stBaseButton-pillsActive"],
section[data-testid="stSidebar"] button[kind="pillsActive"] {{ background: rgba(99,102,241,.18);
  border: 1px solid {INDIGO}; color: #E0E7FF; border-radius: 999px; font-size: .72rem;
  padding: .05rem .65rem; min-height: 1.75rem; }}
section[data-testid="stSidebar"] [data-testid="stThumbValue"],
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"] {{ color: #CBD5E1; }}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: #64748B; }}
</style>
"""


# =========================================================================
# 2. DATA LOADING (generate_data.py -> aggregate -> privacy.py)
# =========================================================================

def generate_fallback_data(seed: int = 42) -> pd.DataFrame:
    """
    Built-in synthetic AGGREGATED data, used only if generate_data.py cannot be used.
    9 provinces x 12 weeks x 5 themes. Smaller provinces get small counts so that
    privacy suppression is visible, a few weeks are made to rise so the early-warning
    system has something to detect, and a few cells are left missing on purpose.
    """
    rng = np.random.default_rng(seed)
    base_mix = np.array([0.30, 0.22, 0.16, 0.22, 0.10])  # order of THEMES
    weekly_size = {"Western": 300, "Central": 170, "Southern": 150, "Northern": 80,
                   "Eastern": 110, "North Western": 150, "North Central": 90,
                   "Uva": 75, "Sabaragamuwa": 120}
    # (province, theme): size of simulated rise over the last 3 weeks
    rising = {("Eastern", "Job and Money"): 0.07,
              ("Western", "Exams and Studies"): 0.06,
              ("Northern", "Loneliness and Isolation"): 0.08,
              ("Southern", "Family and Relationships"): 0.05}
    rows = []
    for prov in PROVINCES:
        prov_mix = base_mix * rng.uniform(0.85, 1.15, len(THEMES))
        prov_mix /= prov_mix.sum()
        for week in range(1, NUM_WEEKS + 1):
            mix = prov_mix.copy()
            for (p, t), delta in rising.items():
                if p == prov:
                    mix[THEMES.index(t)] += delta * max(0, week - 9) / 3
            mix = np.clip(mix * rng.uniform(0.95, 1.05, len(THEMES)), 0.01, None)
            mix /= mix.sum()
            total = max(int(rng.poisson(weekly_size[prov])), 1)
            counts = rng.multinomial(total, mix)
            for theme, c in zip(THEMES, counts):
                rows.append({"province": prov, "week": week, "theme": theme,
                             "response_count": int(c),
                             "percentage_share": round(100 * c / total, 1)})
    df = pd.DataFrame(rows)
    df[["response_count", "percentage_share"]] = df[["response_count", "percentage_share"]].astype(float)
    gaps = rng.choice(len(df), size=4, replace=False)  # simulate missing data
    df.loc[gaps, ["response_count", "percentage_share"]] = np.nan
    return df


def to_aggregated(raw) -> pd.DataFrame:
    """
    Make sure we only keep aggregate columns. If the data is record-level,
    count it per province-week-theme and drop everything else (e.g. any text).
    """
    df = pd.DataFrame(raw)
    needed = privacy.REQUIRED_COLUMNS
    if all(c in df.columns for c in needed):
        return df[needed].copy()
    if all(c in df.columns for c in ["province", "week", "theme"]):
        agg = (df.groupby(["province", "week", "theme"]).size()
                 .rename("response_count").reset_index())
        totals = agg.groupby(["province", "week"])["response_count"].transform("sum")
        agg["percentage_share"] = (100 * agg["response_count"] / totals).round(1)
        return agg[needed]
    raise ValueError("Data must contain province, week and theme columns "
                     "(plus response_count and percentage_share if aggregated).")


def _natural_key(value):
    """Sort 'Week 2' before 'Week 10' (and plain numbers numerically)."""
    parts = re.findall(r"\d+|\D+", str(value))
    return [(0, int(p)) if p.isdigit() else (1, p) for p in parts]


@st.cache_data(show_spinner="Applying privacy protection...")
def load_data():
    """
    Load -> aggregate -> privacy filter. Returns only privacy-safe data.
    Raw / pre-filter values never leave this function.
    """
    raw, source = None, "built-in fallback generator"
    if generate_data is not None and hasattr(generate_data, GENERATE_FUNCTION_NAME):
        try:
            raw = getattr(generate_data, GENERATE_FUNCTION_NAME)()
            source = "generate_data.py"
        except Exception:  # noqa: BLE001
            raw = None
    if raw is None:
        raw = generate_fallback_data()

    agg = to_aggregated(raw)
    del raw

    # Data quality: how many province-week-theme cells are missing or incomplete?
    expected = agg["province"].nunique() * agg["week"].nunique() * agg["theme"].nunique()
    valid = agg.dropna(subset=["response_count", "percentage_share"])
    quality = {"expected_cells": int(expected),
               "missing_cells": int(max(expected - len(valid), 0)),
               "source": source}

    # PRIVACY STEP: suppress every cell with fewer than K = 10 responses
    safe, summary = privacy.apply_privacy_filter(agg)
    safe["response_count"] = safe["response_count"].astype(float)  # NA -> NaN for easy maths
    return safe, summary, quality


# =========================================================================
# 3. ANALYSIS HELPERS
# =========================================================================

def visible_only(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that survived privacy suppression (the only rows used for results)."""
    return df[~df["suppressed"]]


def week_label(week) -> str:
    return f"Week {week}" if isinstance(week, (int, np.integer)) else str(week)


def builtin_alerts(df: pd.DataFrame, end_week, week_order: list) -> pd.DataFrame:
    """
    Simple early-warning detector (used if analysis.py does not provide one).

    For each province-theme: compare the current week with the mean of up to
    BASELINE_WEEKS earlier weeks. Only VISIBLE (non-suppressed) cells are used.
    """
    empty = pd.DataFrame(columns=ALERT_COLUMNS)
    visible = visible_only(df)
    pos = week_order.index(end_week)
    base_weeks = week_order[max(0, pos - BASELINE_WEEKS):pos]
    current = visible[visible["week"] == end_week].set_index(["province", "theme"])
    base = (visible[visible["week"].isin(base_weeks)]
            .groupby(["province", "theme"])["percentage_share"]
            .agg(["mean", "std", "count"]))
    if current.empty or base.empty:
        return empty

    rows = []
    for key, cur in current.iterrows():
        if key not in base.index:
            continue
        b = base.loc[key]
        if b["count"] < MIN_BASELINE_WEEKS:
            continue
        change = cur["percentage_share"] - b["mean"]
        spread = max(0.0 if pd.isna(b["std"]) else float(b["std"]), STD_FLOOR)
        z = change / spread
        p_value = stats.norm.sf(z)  # one-sided: chance of a rise this big by luck
        if change >= ALERT_PP and p_value < 0.05:
            status = "ALERT"
        elif change >= WATCH_PP:
            status = "WATCH"
        else:
            continue
        rows.append({"province": key[0], "theme": key[1],
                     "baseline_pct": float(b["mean"]),
                     "current_pct": float(cur["percentage_share"]),
                     "change_pp": float(change),
                     "response_count": int(cur["response_count"]),
                     "status": status})
    if not rows:
        return empty
    out = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    out["_rank"] = (out["status"] != "ALERT").astype(int)  # ALERT first
    return (out.sort_values(["_rank", "change_pp"], ascending=[True, False])
               .drop(columns="_rank").reset_index(drop=True))


def compute_alerts(df: pd.DataFrame, end_week, week_order: list):
    """Use analysis.py if it provides a valid detector, otherwise the built-in one."""
    if analysis is not None and hasattr(analysis, ALERT_FUNCTION_NAME):
        try:
            out = getattr(analysis, ALERT_FUNCTION_NAME)(df.copy(), end_week)
            if isinstance(out, pd.DataFrame) and set(ALERT_COLUMNS) <= set(out.columns):
                return out[ALERT_COLUMNS].reset_index(drop=True), "analysis.py"
        except Exception:  # noqa: BLE001
            pass
    return builtin_alerts(df, end_week, week_order), "built-in detector"


# =========================================================================
# 4. CHART BUILDERS (each returns a Plotly figure)
# =========================================================================

def style_figure(fig: go.Figure, height: int = 460) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=NAVY),
                      margin=dict(l=20, r=20, t=50, b=30))
    return fig


def build_donut(filtered: pd.DataFrame, themes: list):
    totals = (visible_only(filtered).groupby("theme")["response_count"].sum()
              .reindex(themes).dropna())
    totals = totals[totals > 0]
    if totals.empty:
        return None
    fig = go.Figure(go.Pie(
        labels=list(totals.index), values=list(totals.values), hole=0.58, sort=False,
        marker=dict(colors=[THEME_COLOURS.get(t, LIGHT_BLUE) for t in totals.index],
                    line=dict(color="white", width=2)),
        textinfo="percent",
        hovertemplate="%{label}<br>%{value:,.0f} responses (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{totals.sum():,.0f}</b><br>responses", showarrow=False,
                       font=dict(size=16, color=NAVY))
    fig.update_layout(legend=dict(orientation="h", y=-0.08))
    return style_figure(fig, 480)


def build_trend(filtered: pd.DataFrame, theme: str, provinces: list, weeks: list):
    sub = filtered[filtered["theme"] == theme]
    pct = sub.pivot(index="week", columns="province", values="percentage_share").reindex(weeks)
    cnt = sub.pivot(index="week", columns="province", values="response_count").reindex(weeks)
    labels = [week_label(w) for w in weeks]
    fig = go.Figure()
    for i, prov in enumerate(p for p in provinces if p in pct.columns):
        fig.add_trace(go.Scatter(
            x=labels, y=pct[prov].astype(float), name=prov, mode="lines+markers",
            connectgaps=False,  # suppressed weeks stay as gaps
            line=dict(color=LINE_COLOURS[i % len(LINE_COLOURS)], width=2.5),
            customdata=cnt[prov].astype(float),
            hovertemplate="%{x}<br>" + prov + ": %{y:.1f}% (n=%{customdata:,.0f})<extra></extra>"))
    fig.update_layout(yaxis=dict(title="Percentage share (%)", ticksuffix="%", rangemode="tozero",
                                 gridcolor="#DCE6F2"),
                      xaxis=dict(type="category", categoryorder="array", categoryarray=labels),
                      legend=dict(orientation="h", y=-0.2), hovermode="x unified")
    return style_figure(fig, 480)


def build_heatmap(filtered: pd.DataFrame, provinces: list, themes: list):
    vis = visible_only(filtered)
    grid = (vis.groupby(["province", "theme"])["percentage_share"].mean()
               .unstack().reindex(index=provinces, columns=themes))
    values = grid.to_numpy(dtype=float)
    text = [["-" if np.isnan(v) else f"{v:.1f}%" for v in row] for row in values]
    fig = go.Figure(go.Heatmap(
        z=values, x=themes, y=provinces, text=text, texttemplate="%{text}", zmin=0,
        colorscale=[[0, "#EAF2FB"], [0.5, "#4FA3E0"], [1, NAVY]], hoverongaps=False,
        colorbar=dict(title="Share (%)"),
        hovertemplate="%{y}<br>%{x}<br>Average share: %{z:.1f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, 520)


# =========================================================================
# 5. SMALL HTML HELPERS
# =========================================================================

def kpi_cards(cards: list) -> None:
    """cards = list of (label, value, note). Rendered as a responsive grid."""
    items = "".join(
        f'<div class="kpi-card"><div><div class="kpi-top"><div class="kpi-label">{html.escape(str(l))}</div></div>'
        f'<div class="kpi-value">{html.escape(str(v))}</div></div>'
        f'<div class="kpi-note">{html.escape(str(n))}</div></div>'
        for l, v, n in cards)
    st.markdown(f'<div class="kpi-grid">{items}</div>', unsafe_allow_html=True)


def kpi_card_html(label: str, icon: str, value: str, extra: str = "",
                  card_cls: str = "", icon_cls: str = "", value_cls: str = "") -> str:
    """One Overview KPI card: label + icon chip, big value, and an extra line."""
    return (f'<div class="kpi-card {card_cls}"><div><div class="kpi-top"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-icon {icon_cls}">{svg(icon, 16)}</div></div>'
            f'<div class="kpi-value {value_cls}">{value}</div></div><div>{extra}</div></div>')


def page_header(title: str, ctx: dict) -> None:
    """Page title on the left, week context on the right."""
    meta = ""
    if ctx.get("weeks"):
        weeks = ctx["weeks"]
        if title == "Overview":
            meta = week_label(ctx["end_week"])
        elif len(weeks) > 1:
            meta = f"{week_label(weeks[0])} \u2013 {week_label(weeks[-1])}"
        else:
            meta = week_label(weeks[0])
        meta += " &middot; Released cells only"
    st.markdown(f'<div class="page-head"><div class="page-title">{html.escape(title)}</div>'
                f'<div class="page-meta">{meta}</div></div>', unsafe_allow_html=True)


def render_topbar_and_banner() -> None:
    st.markdown(
        '<div class="topbar"><div><div class="topbar-title">Community Mental-Wellbeing Signal Tracker</div>'
        '<div class="topbar-sub">Regional Early-Warning and Stressor Monitoring System &ndash; Sri Lanka</div></div>'
        '<div class="demo-pill"><span class="dot"></span><b>Demonstration mode</b>'
        '<span>Synthetic data</span></div></div>'
        f'<div class="demo-banner">{svg("warning", 18)}<div><b>Demonstration mode &ndash; synthetic data.</b> '
        'All results are simulated and do not represent actual population conditions.</div></div>',
        unsafe_allow_html=True)


def alert_cards(alerts: pd.DataFrame) -> None:
    out = []
    for r in alerts.itertuples(index=False):
        status = str(r.status).upper()
        css = "alert" if status == "ALERT" else "watch"

        def metric(label, value):
            return (f'<div><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{html.escape(str(value))}</div></div>')

        out.append(
            f'<div class="alert-card {css}"><span class="badge {css}">SIMULATED {html.escape(status)}</span>'
            f'<div class="alert-title">{html.escape(str(r.province))} &middot; {html.escape(str(r.theme))}</div>'
            f'<div class="metric-row">'
            f'{metric("Previous baseline", f"{r.baseline_pct:.1f}%")}'
            f'{metric("Current", f"{r.current_pct:.1f}%")}'
            f'{metric("Change", f"{r.change_pp:+.1f} pp")}'
            f'{metric("Responses", f"{int(r.response_count):,}")}'
            f'</div><div class="sim-tag">Simulated signal from synthetic data.</div></div>')
    st.markdown(f'<div class="alert-grid">{"".join(out)}</div>', unsafe_allow_html=True)


def show_chart(fig) -> None:
    """Display a Plotly chart (works on older and newer Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:  # noqa: BLE001 - older Streamlit
        st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# 6. PAGES
# =========================================================================

def page_overview(ctx: dict) -> None:
    page_header("Overview", ctx)
    vis = visible_only(ctx["filtered"])
    totals = vis.groupby("theme")["response_count"].sum().sort_values(ascending=False)
    if totals.empty or totals.sum() == 0:
        dominant, note = "Not available", "All selected cells are suppressed"
    else:
        dominant = totals.index[0]
        note = f"{100 * totals.iloc[0] / totals.sum():.1f}% of responses"
    n_alerts = int((ctx["alerts"]["status"].astype(str).str.upper() == "ALERT").sum())
    n_sel, n_all = len(ctx["provinces"]), len(ctx["all_provinces"])
    coverage = 100 * n_sel / n_all if n_all else 0

    alert_badge = (f'<span class="pill coral"><span class="dot"></span>{html.escape(week_label(ctx["end_week"]))} active</span>'
                   if n_alerts else '<span class="pill grey"><span class="dot"></span>None this week</span>')
    cards = [
        kpi_card_html("Total synthetic responses", "message", f"{totals.sum():,.0f}",
                      '<div class="kpi-note">Released cells only</div>'),
        kpi_card_html("Provinces included", "pin", f'{n_sel} <span class="of">/ {n_all}</span>',
                      f'<div class="progress"><div style="width:{coverage:.0f}%"></div></div>'
                      f'<div class="kpi-note">{"Complete coverage" if n_sel == n_all else f"{coverage:.0f}% of provinces selected"}</div>'),
        kpi_card_html("Dominant stressor", "dollar", html.escape(str(dominant)),
                      f'<div class="kpi-note">{html.escape(note)}</div>',
                      card_cls="primary", icon_cls="indigo", value_cls="text"),
        kpi_card_html("Simulated alerts", "bell", str(n_alerts), alert_badge, icon_cls="coral"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="callout"><div class="callout-icon">{svg("info", 18)}</div><div>'
        '<div class="callout-title">What this tool shows</div>'
        '<div class="callout-text">Monitors changes in community stressors using aggregated, anonymized, '
        'and synthetic data.</div>'
        f'<div class="disclaimer">{svg("shield", 14)}Reports stressor themes only. '
        'Does not provide clinical diagnoses or treatment advice.</div></div></div>',
        unsafe_allow_html=True)


def page_stressors(ctx: dict) -> None:
    page_header("Stressor Analysis", ctx)
    st.markdown('<p class="section-note">Share of released responses by stressor theme '
                'for the current filters.</p>', unsafe_allow_html=True)
    fig = build_donut(ctx["filtered"], ctx["themes"])
    if fig is None:
        st.info("No releasable data for this selection (all cells are below the minimum group size).")
    else:
        show_chart(fig)


def page_trends(ctx: dict) -> None:
    page_header("Weekly Trend Analysis", ctx)
    theme = st.selectbox("Select a theme", ctx["themes"])
    if len(ctx["weeks"]) < 2:
        st.info("Select a range of at least two weeks in the sidebar to see a trend.")
        return
    if visible_only(ctx["filtered"][ctx["filtered"]["theme"] == theme]).empty:
        st.info("No releasable data for this theme and selection.")
        return
    show_chart(build_trend(ctx["filtered"], theme, ctx["provinces"], ctx["weeks"]))
    st.caption("Lines show each province's weekly percentage share. Gaps mean the value was "
               f"suppressed (fewer than {K} responses) or missing.")


def page_regional(ctx: dict) -> None:
    page_header("Regional Comparison", ctx)
    st.markdown('<p class="section-note">Average weekly percentage share of each theme by '
                'province, over the selected weeks. Privacy suppression is applied first.</p>',
                unsafe_allow_html=True)
    if visible_only(ctx["filtered"]).empty:
        st.info("No releasable data for this selection.")
        return
    show_chart(build_heatmap(ctx["filtered"], ctx["provinces"], ctx["themes"]))
    st.caption('A "-" means no releasable data (suppressed or missing) for that province and theme.')


def page_alerts(ctx: dict) -> None:
    page_header("Early Warning System", ctx)
    st.warning("All alerts below are SIMULATED from synthetic data. They do not describe real communities.")
    st.markdown(
        f"Compared with the previous baseline (mean of up to {BASELINE_WEEKS} weeks before "
        f"**{week_label(ctx['end_week'])}**). WATCH = rise of {WATCH_PP:.0f}+ percentage points (pp). "
        f"ALERT = rise of {ALERT_PP:.0f}+ pp that is also statistically unusual. "
        f"Detector: {ctx['alert_source']}.")
    if ctx["alerts"].empty:
        st.info("No simulated alerts for the current selection and week.")
    else:
        alert_cards(ctx["alerts"])


def page_privacy(ctx: dict) -> None:
    page_header("Privacy & Data Quality", ctx)
    s, q = ctx["summary"], ctx["quality"]
    kpi_cards([
        ("Minimum group size (k)", K, "Cells below this are hidden"),
        ("Suppressed groups", f"{s['suppressed_cells']:,}", f"{s['percent_suppressed']}% of all cells"),
        ("Visible groups", f"{s['visible_cells']:,}", "Released to charts"),
        ("Missing data", f"{q['missing_cells']:,}",
         f"of {q['expected_cells']:,} expected cells"),
    ])
    st.markdown(
        f"- **Rule:** every province-week-theme group with fewer than **{K}** responses has its "
        "count and percentage hidden. Hidden values are never charted, tabulated or downloaded.\n"
        "- **Complementary suppression:** if only one group in a province-week is hidden, the "
        "next-smallest group is also hidden so the hidden value cannot be worked out by subtraction.\n"
        "- **No individual data:** the dashboard holds aggregates only. No raw text or "
        "individual records are shown.\n"
        "- **Synthetic data:** all values are simulated and do not represent real people.")
    insufficient = s["provinces_with_insufficient_data"]
    partial = s["provinces_with_some_suppression"]
    st.markdown("**Provinces with insufficient data (all groups suppressed):** "
                + (", ".join(insufficient) if insufficient else "None"))
    st.markdown("**Provinces with some suppressed groups:** "
                + (", ".join(partial) if partial else "None"))

    by_prov = (ctx["safe"].groupby("province")["suppressed"].agg(["sum", "count"])
               .rename(columns={"sum": "Suppressed groups", "count": "Total groups"}))
    by_prov["Visible groups"] = by_prov["Total groups"] - by_prov["Suppressed groups"]
    by_prov = by_prov[["Visible groups", "Suppressed groups", "Total groups"]]
    st.dataframe(by_prov.reset_index().rename(columns={"province": "Province"}), hide_index=True)
    st.caption(f"Data source in use: {q['source']}.")


def page_methodology(ctx: dict) -> None:
    page_header("Methodology", ctx)
    st.markdown(f"""
**Data generation.** All data is synthetic. It is created by `generate_data.py` (or a small
built-in simulator if that module is unavailable) for 9 provinces, 12 weeks and 5 themes.
Some provinces are small, a few cells are left missing, and a few themes are made to rise
gently so the early-warning logic can be demonstrated. Nothing here reflects real conditions.

**Theme classification assumptions.** Each simulated response is assigned to one primary theme:
Job and Money, Exams and Studies, Loneliness and Isolation, Family and Relationships, or Other.
Themes describe *stressors*, not people. The dashboard makes no diagnosis and gives no treatment advice.

**Weekly aggregation.** Responses are counted per province, week and theme. The percentage
share is the theme's count divided by all responses in that province-week.

**Baseline comparison.** For the current week (the end of the selected range), each
province-theme share is compared with the mean of up to {BASELINE_WEEKS} previous weeks
(at least {MIN_BASELINE_WEEKS} visible weeks needed). A rise of {WATCH_PP:.0f}+ percentage points
is a WATCH; a rise of {ALERT_PP:.0f}+ points that a one-sided z-test also finds unusual (p < 0.05,
spread floored at {STD_FLOOR:.0f} pp) is an ALERT. All of these are simulated.

**Privacy suppression.** Groups with fewer than {K} responses are suppressed (count and percentage
hidden) before any chart or table is built, using `privacy.py`. A neighbouring group is also hidden
when a single hidden group could otherwise be recovered by subtraction.

**Limitations.** Synthetic data, small-number suppression and simple thresholds mean
this is a demonstration of method, not a source of findings about real communities.
""")


# =========================================================================
# 7. MAIN APP
# =========================================================================

def go_to(page: str) -> None:
    """Button callback: remember which page is open."""
    st.session_state["page"] = page


def reset_filters(provinces: list, themes: list, weeks: list) -> None:
    """Button callback: put every filter back to its default."""
    st.session_state["f_prov"] = list(provinces)
    st.session_state["f_theme"] = list(themes)
    if len(weeks) > 1:
        st.session_state["f_week"] = (weeks[0], weeks[-1])


def nav_button(name: str, icon: str, active: bool) -> None:
    kwargs = dict(key=f"nav_{name}", on_click=go_to, args=(name,),
                  type="primary" if active else "secondary")
    try:
        st.sidebar.button(name, icon=icon, **kwargs)
    except TypeError:  # older Streamlit without button icons
        st.sidebar.button(name, **kwargs)


def chip_filter(title: str, options: list, key: str, all_label: bool = False) -> list:
    """Chip toggles (st.pills) for a multi-select filter, with a '9 of 9' style counter."""
    st.session_state.setdefault(key, list(options))
    n = len(st.session_state[key])
    count = "All" if (all_label and n == len(options)) else f"{n} of {len(options)}"
    st.sidebar.markdown(f'<div class="filter-label">{html.escape(title)}<span>{count}</span></div>',
                        unsafe_allow_html=True)
    if hasattr(st, "pills"):
        chosen = st.sidebar.pills(title, options, selection_mode="multi", key=key,
                                  label_visibility="collapsed") or []
    else:  # older Streamlit: plain multiselect instead of chips
        chosen = st.sidebar.multiselect(title, options, key=key, label_visibility="collapsed")
    return [o for o in options if o in chosen]


def main() -> None:
    st.set_page_config(page_title="Community Mental-Wellbeing Signal Tracker",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    render_topbar_and_banner()

    # ---- Load privacy-safe data ----
    try:
        safe, summary, quality = load_data()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load data: {exc}")
        st.stop()
    if safe.empty:
        st.error("The dataset is empty.")
        st.stop()

    all_provinces = ([p for p in PROVINCES if p in set(safe["province"])]
                     + sorted(set(safe["province"]) - set(PROVINCES)))
    all_themes = ([t for t in THEMES if t in set(safe["theme"])]
                  + sorted(set(safe["theme"]) - set(THEMES)))
    week_order = sorted(safe["week"].unique().tolist(), key=_natural_key)

    # ---- Sidebar: brand, navigation, filters ----
    st.session_state.setdefault("page", "Overview")
    page = st.session_state["page"]
    st.sidebar.markdown(f'<div class="brand"><div class="brand-icon">{svg("activity", 18)}</div>'
                        '<div class="brand-name">Signal Tracker</div></div>', unsafe_allow_html=True)
    for name, icon in NAV_ITEMS:
        nav_button(name, icon, active=(name == page))

    st.sidebar.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
    head_left, head_right = st.sidebar.columns([1, 1])
    head_left.markdown('<div class="filter-head">Filters</div>', unsafe_allow_html=True)
    head_right.button("Reset", key="reset_filters", on_click=reset_filters,
                      args=(all_provinces, all_themes, week_order))

    provinces = chip_filter("Province", all_provinces, "f_prov")
    themes = chip_filter("Theme", all_themes, "f_theme", all_label=True)

    if len(week_order) > 1:
        st.sidebar.markdown('<div class="filter-label">Week range<span></span></div>', unsafe_allow_html=True)
        start, end = st.sidebar.select_slider("Week range", options=week_order,
                                              value=(week_order[0], week_order[-1]), key="f_week",
                                              format_func=week_label, label_visibility="collapsed")
    else:
        start = end = week_order[0]
    st.sidebar.caption(f"Data source: {quality['source']}")
    st.sidebar.caption(f"Privacy: groups under {K} responses are hidden.")

    if page == "Methodology":
        page_methodology({})
        return
    if not provinces or not themes:
        st.warning("Please select at least one province and one theme in the sidebar.")
        return

    # ---- Apply filters (to privacy-safe data only) ----
    weeks = week_order[week_order.index(start): week_order.index(end) + 1]
    by_selection = safe[safe["province"].isin(provinces) & safe["theme"].isin(themes)]
    filtered = by_selection[by_selection["week"].isin(weeks)]
    if filtered.empty:
        st.info("No data matches the current filters.")
        return

    alerts, alert_source = compute_alerts(by_selection, end, week_order)
    ctx = {"safe": safe, "filtered": filtered, "summary": summary, "quality": quality,
           "provinces": provinces, "themes": themes, "weeks": weeks, "end_week": end,
           "all_provinces": all_provinces, "alerts": alerts, "alert_source": alert_source}

    {"Overview": page_overview, "Stressor Analysis": page_stressors,
     "Weekly Trends": page_trends, "Regional Comparison": page_regional,
     "Early Warning": page_alerts, "Privacy & Data Quality": page_privacy}[page](ctx)

    st.caption("DEMONSTRATION MODE - SYNTHETIC DATA. No raw responses or individual records are shown.")


if __name__ == "__main__":
    main()