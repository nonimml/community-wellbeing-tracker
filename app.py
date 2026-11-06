"""
app.py - Community Mental-Wellbeing Signal Tracker (Streamlit dashboard)
=========================================================================
Regional Early-Warning and Stressor Monitoring System — Sri Lanka.

DEMONSTRATION MODE: All data is retrieved from Supabase and is SYNTHETIC.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import database
import geography
import privacy  # k-anonymity-style small-cell suppression (k = 10)

try:
    import analysis
except Exception:  # noqa: BLE001
    analysis = None


# =========================================================================
# 1. SETTINGS & CONSTANTS
# =========================================================================

ALERT_FUNCTION_NAME = "detect_alerts"
ALERT_COLUMNS = ["region", "theme", "baseline_pct", "current_pct",
                 "change_pp", "response_count", "status"]

K = privacy.K_THRESHOLD  # minimum group size (10)

PROVINCES = list(geography.PROVINCES)
DISTRICTS = list(geography.DISTRICTS)
THEMES = ["Job and Money", "Exams and Studies", "Loneliness and Isolation",
          "Family and Relationships", "Other"]
NUM_WEEKS = 12

BASELINE_WEEKS = 4      
MIN_BASELINE_WEEKS = 2  
WATCH_PP = 3.0          
ALERT_PP = 5.0          
STD_FLOOR = 1.0         

NAVY, BLUE, LIGHT_BLUE, BG = "#0F172A", "#1F5FA8", "#8DC3EE", "#F8FAFC"
INDIGO = "#6366F1"  
THEME_COLOURS = {
    "Job and Money": "#0B2545",
    "Exams and Studies": "#1F5FA8",
    "Loneliness and Isolation": "#4FA3E0",
    "Family and Relationships": "#8DC3EE",
    "Other": "#A8B3C5",
}
LINE_COLOURS = ["#0B2545", "#1F78C8", "#4FA3E0", "#2A9D8F", "#E9A23B",
                "#C8553D", "#5B6B82", "#8DC3EE", "#7A4EAB"]

NAV_ITEMS = [("Overview", ":material/dashboard:"),
             ("Stressor Analysis", ":material/bar_chart:"),
             ("Weekly Trends", ":material/trending_up:"),
             ("Regional Comparison", ":material/grid_view:"),
             ("Early Warning", ":material/notifications:"),
             ("Privacy & Data Quality", ":material/shield:"),
             ("Methodology", ":material/menu_book:")]
PAGES = [name for name, _ in NAV_ITEMS]

ICON_PATHS = {
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "dollar": '<line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
    "warning": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1 z"/>',
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
.kpi-card {{ 
    background: #fff; 
    border: 1px solid #E2E8F0; 
    border-top: 3px solid {INDIGO};
    border-radius: 12px; 
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(15,23,42,.04); 
    display: flex; 
    flex-direction: column;
    justify-content: space-between; 
    min-height: 140px;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}}
.kpi-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
    border-color: #CBD5E1;
    border-top-color: {INDIGO};
}}
.kpi-card.primary {{ box-shadow: 0 4px 14px rgba(99,102,241,.14); }}
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
.callout {{ 
    display: flex; 
    gap: 1rem; 
    background: #fff; 
    border: 1px solid #E2E8F0; 
    border-left: 4px solid {INDIGO};
    border-radius: 12px; 
    padding: 1.2rem 1.4rem; 
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}
.callout:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(15, 23, 42, 0.06);
}}
.callout-icon {{ width: 32px; height: 32px; border-radius: 8px; background: #EEF2FF; color: {INDIGO};
  display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
.callout-title {{ font-weight: 600; color: {NAVY}; margin-bottom: .5rem; }}
.callout-text {{ color: #334155; font-size: .95rem; line-height: 1.5; }}
.disclaimer {{ display: inline-flex; align-items: center; gap: .5rem; background: #F1F5F9; border-radius: 8px;
  padding: .4rem .8rem; color: #64748B; font-style: italic; font-size: .82rem; margin-top: .8rem; }}

/* ---------- Alert cards ---------- */
.alert-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 1.1rem; margin-top: .5rem; }}
.alert-card {{ 
    background: #fff; 
    border-radius: 12px; 
    padding: 1rem 1.2rem; 
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}
.alert-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 16px rgba(15, 23, 42, 0.06);
}}
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
.st-key-reset_filters,
.st-key-reset_filters .stButton {{ display: flex; justify-content: center; width: 100%; }}
.st-key-reset_filters button {{ width: auto !important; margin: 0 !important; color: #E2E8F0 !important;
  font-size: .75rem !important; padding: .2rem .6rem !important; border-left: none !important;
  background: transparent !important; min-height: 0 !important; }}
.st-key-reset_filters button > div {{ justify-content: center !important; }}
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
# 2. SUPABASE DATA LOADING & TRANSFORMATION LAYER
# =========================================================================

@st.cache_data(
    show_spinner="Connecting to Supabase and applying privacy suppression...",
    ttl=300,
    max_entries=2,
)
def load_supabase_data(geography_level: str = "province") -> tuple[pd.DataFrame, dict, dict]:
    """
    Fetch records from Supabase wellbeing_responses table, apply transformation layer
    into province- and district-level aggregates. Both levels are privacy
    filtered together so their totals cannot expose one suppressed district.
    """
    raw_records = database.fetch_responses()
    if not raw_records:
        raise ValueError("No records returned from Supabase table public.wellbeing_responses.")

    df = pd.DataFrame(raw_records)
    safe_provinces, safe_districts, province_summary, district_summary = (
        privacy.aggregate_response_levels(df)
    )
    if geography_level == "province":
        safe, summary = safe_provinces, province_summary
    elif geography_level == "district":
        safe, summary = safe_districts, district_summary
    else:
        raise ValueError("geography_level must be 'province' or 'district'.")

    safe["response_count"] = safe["response_count"].astype(float)
    quality = {
        "source": "Supabase database (public.wellbeing_responses)",
        "total_records": len(df),
    }
    return safe, summary, quality


def to_aggregated(raw) -> pd.DataFrame:
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
    raise ValueError("Data must contain province, week and theme columns.")


def _natural_key(value):
    parts = re.findall(r"\d+|\D+", str(value))
    return [(0, int(p)) if p.isdigit() else (1, p) for p in parts]


# =========================================================================
# 3. ANALYSIS HELPERS
# =========================================================================

def visible_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["suppressed"]]


def week_label(week) -> str:
    return f"Week {week}" if isinstance(week, (int, np.integer)) else str(week)


def builtin_alerts(df: pd.DataFrame, end_week, week_order: list) -> pd.DataFrame:
    empty = pd.DataFrame(columns=ALERT_COLUMNS)
    visible = visible_only(df)
    pos = week_order.index(end_week)
    base_weeks = week_order[max(0, pos - BASELINE_WEEKS):pos]
    current = visible[visible["week"] == end_week].set_index(["region", "theme"])
    base = (visible[visible["week"].isin(base_weeks)]
            .groupby(["region", "theme"])["percentage_share"]
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
        p_value = stats.norm.sf(z)
        if change >= ALERT_PP and p_value < 0.05:
            status = "ALERT"
        elif change >= WATCH_PP:
            status = "WATCH"
        else:
            continue
        rows.append({"region": key[0], "theme": key[1],
                     "baseline_pct": float(b["mean"]),
                     "current_pct": float(cur["percentage_share"]),
                     "change_pp": float(change),
                     "response_count": int(cur["response_count"]),
                     "status": status})
    if not rows:
        return empty
    out = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    out["_rank"] = (out["status"] != "ALERT").astype(int)
    return (out.sort_values(["_rank", "change_pp"], ascending=[True, False])
               .drop(columns="_rank").reset_index(drop=True))


def compute_alerts(df: pd.DataFrame, end_week, week_order: list):
    if analysis is not None and hasattr(analysis, ALERT_FUNCTION_NAME):
        try:
            analysis_input = df.rename(columns={"region": "province"})
            out = getattr(analysis, ALERT_FUNCTION_NAME)(analysis_input.copy(), end_week)
            if isinstance(out, pd.DataFrame):
                out = out.rename(columns={"province": "region"})
                if set(ALERT_COLUMNS) <= set(out.columns):
                    return out[ALERT_COLUMNS].reset_index(drop=True), "analysis.py"
        except Exception:  # noqa: BLE001
            pass
    return builtin_alerts(df, end_week, week_order), "built-in detector"


# =========================================================================
# 4. CHART BUILDERS
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


def build_trend(filtered: pd.DataFrame, theme: str, regions: list, weeks: list):
    sub = filtered[filtered["theme"] == theme]
    pct = sub.pivot(index="week", columns="region", values="percentage_share").reindex(weeks)
    cnt = sub.pivot(index="week", columns="region", values="response_count").reindex(weeks)
    labels = [week_label(w) for w in weeks]
    fig = go.Figure()
    for i, region in enumerate(p for p in regions if p in pct.columns):
        fig.add_trace(go.Scatter(
            x=labels, y=pct[region].astype(float), name=region, mode="lines+markers",
            connectgaps=False,
            line=dict(color=LINE_COLOURS[i % len(LINE_COLOURS)], width=2.5),
            customdata=cnt[region].astype(float),
            hovertemplate="%{x}<br>" + region + ": %{y:.1f}% (n=%{customdata:,.0f})<extra></extra>"))
    fig.update_layout(yaxis=dict(title="Percentage share (%)", ticksuffix="%", rangemode="tozero",
                                 gridcolor="#DCE6F2"),
                      xaxis=dict(type="category", categoryorder="array", categoryarray=labels),
                      legend=dict(orientation="h", y=-0.2), hovermode="x unified")
    return style_figure(fig, 480)


def build_heatmap(filtered: pd.DataFrame, regions: list, themes: list):
    vis = visible_only(filtered)
    grid = (vis.groupby(["region", "theme"])["percentage_share"].mean()
               .unstack().reindex(index=regions, columns=themes))
    values = grid.to_numpy(dtype=float)
    text = [["-" if np.isnan(v) else f"{v:.1f}%" for v in row] for row in values]
    fig = go.Figure(go.Heatmap(
        z=values, x=themes, y=regions, text=text, texttemplate="%{text}", zmin=0,
        colorscale=[[0, "#EAF2FB"], [0.5, "#4FA3E0"], [1, NAVY]], hoverongaps=False,
        colorbar=dict(title="Share (%)"),
        hovertemplate="%{y}<br>%{x}<br>Average share: %{z:.1f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, 520)


# =========================================================================
# 5. UI COMPONENTS & PAGES
# =========================================================================

def kpi_cards(cards: list) -> None:
    items = "".join(
        f'<div class="kpi-card"><div><div class="kpi-top"><div class="kpi-label">{html.escape(str(l))}</div></div>'
        f'<div class="kpi-value">{html.escape(str(v))}</div></div>'
        f'<div class="kpi-note">{html.escape(str(n))}</div></div>'
        for l, v, n in cards)
    st.markdown(f'<div class="kpi-grid">{items}</div>', unsafe_allow_html=True)


def kpi_card_html(label: str, icon: str, value: str, extra: str = "",
                  card_cls: str = "", icon_cls: str = "", value_cls: str = "") -> str:
    return (f'<div class="kpi-card {card_cls}"><div><div class="kpi-top"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-icon {icon_cls}">{svg(icon, 16)}</div></div>'
            f'<div class="kpi-value {value_cls}">{value}</div></div><div>{extra}</div></div>')


def page_header(title: str, ctx: dict) -> None:
    meta = ""
    if ctx.get("weeks"):
        weeks = ctx["weeks"]
        if title == "Overview":
            meta = week_label(ctx["end_week"])
        elif len(weeks) > 1:
            meta = f"{week_label(weeks[0])} \u2013 {week_label(weeks[-1])}"
        else:
            meta = week_label(weeks[0])
        meta += " · Released cells only"
    st.markdown(f'<div class="page-head"><div class="page-title">{html.escape(title)}</div>'
                f'<div class="page-meta">{meta}</div></div>', unsafe_allow_html=True)


def render_topbar_and_banner() -> None:
    st.markdown(
        '<div class="topbar"><div><div class="topbar-title">Community Mental-Wellbeing Signal Tracker</div>'
        '<div class="topbar-sub">Regional Early-Warning and Stressor Monitoring System – Sri Lanka</div></div></div>'
        f'<div class="demo-banner">{svg("warning", 18)}<div><b>DEMONSTRATION MODE – ALL DATA IS SYNTHETIC.</b> ',
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
            f'<div class="alert-title">{html.escape(str(r.region))} · {html.escape(str(r.theme))}</div>'
            f'<div class="metric-row">'
            f'{metric("Previous baseline", f"{r.baseline_pct:.1f}%")}'
            f'{metric("Current", f"{r.current_pct:.1f}%")}'
            f'{metric("Change", f"{r.change_pp:+.1f} pp")}'
            f'{metric("Responses", f"{int(r.response_count):,}")}'
            f'</div><div class="sim-tag">Simulated signal from Supabase synthetic records.</div></div>')
    st.markdown(f'<div class="alert-grid">{"".join(out)}</div>', unsafe_allow_html=True)


def show_chart(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


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
    n_sel, n_all = len(ctx["regions"]), len(ctx["all_regions"])
    coverage = 100 * n_sel / n_all if n_all else 0
    region_title = ctx["geography"].title()

    alert_badge = (f'<span class="pill coral"><span class="dot"></span>{html.escape(week_label(ctx["end_week"]))} active</span>'
                   if n_alerts else '<span class="pill grey"><span class="dot"></span>None this week</span>')
    cards = [
        kpi_card_html("Total records", "message", f"{ctx['quality']['total_records']:,.0f}"),
        kpi_card_html(f"{region_title}s included", "pin", f'{n_sel} <span class="of">/ {n_all}</span>',
                      f'<div class="progress"><div style="width:{coverage:.0f}%"></div></div>'
                      f'<div class="kpi-note">Sri Lanka demonstration coverage</div>'),
        kpi_card_html("Dominant stressor", "dollar", html.escape(str(dominant)),
                      f'<div class="kpi-note">{html.escape(note)}</div>',
                      card_cls="primary", icon_cls="indigo", value_cls="text"),
        kpi_card_html("Simulated alerts", "bell", str(n_alerts), alert_badge, icon_cls="coral"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="callout"><div class="callout-icon">{svg("info", 18)}</div><div>'
        '<div class="callout-title">What this tool shows</div>'
        '<div class="callout-text">Shows synthetic demonstration patterns for all nine provinces '
        'and 25 districts. The generated values do not describe actual Sri Lankan communities.</div>'
        f'<div class="disclaimer">{svg("shield", 14)}Reports stressor themes only. '
        'Does not provide clinical diagnoses or treatment advice.</div></div></div>',
        unsafe_allow_html=True)


def page_stressors(ctx: dict) -> None:
    page_header("Stressor Analysis", ctx)
    st.markdown('<p class="section-note">Share of released responses by stressor theme '
                'for the current region filters.</p>', unsafe_allow_html=True)
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
    show_chart(build_trend(ctx["filtered"], theme, ctx["regions"], ctx["weeks"]))
    st.caption(f"Lines show each selected {ctx['geography']}'s weekly percentage share. Gaps mean the value was "
               f"suppressed (fewer than {K} responses) or missing.")


def page_regional(ctx: dict) -> None:
    page_header(f"{ctx['geography'].title()} Comparison", ctx)
    st.markdown('<p class="section-note">Average weekly percentage share of each theme by '
                'selected region, over the selected weeks.</p>',
                unsafe_allow_html=True)
    if visible_only(ctx["filtered"]).empty:
        st.info("No releasable data for this selection.")
        return
    show_chart(build_heatmap(ctx["filtered"], ctx["regions"], ctx["themes"]))
    st.caption('A "-" means no releasable data (suppressed or missing) for that region and theme.')


def page_alerts(ctx: dict) -> None:
    page_header("Early Warning System", ctx)
    st.warning("All alerts below are SIMULATED from Supabase synthetic data.")
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
        ("Supabase records", f"{q['total_records']:,}", "Total rows fetched"),
    ])
    st.markdown(
        f"- **Rule:** every {ctx['geography']}-week-theme group with fewer than **{K}** responses has its "
        "count and percentage hidden.\n"
        "- **Scope:** all nine provinces and 25 districts of Sri Lanka.\n"
        "- **Synthetic data:** All records in Supabase are simulated.\n"
        "- **Limit:** small-cell suppression is a risk reduction measure, not a guarantee of anonymity.")
    insufficient = s["regions_with_insufficient_data"]
    partial = s["regions_with_some_suppression"]
    st.markdown(f"**{ctx['geography'].title()}s with insufficient data (all groups suppressed):** "
                + (", ".join(insufficient) if insufficient else "None"))
    st.markdown(f"**{ctx['geography'].title()}s with some suppressed groups:** "
                + (", ".join(partial) if partial else "None"))

    by_prov = (ctx["safe"].groupby("region")["suppressed"].agg(["sum", "count"])
               .rename(columns={"sum": "Suppressed groups", "count": "Total groups"}))
    by_prov["Visible groups"] = by_prov["Total groups"] - by_prov["Suppressed groups"]
    by_prov = by_prov[["Visible groups", "Suppressed groups", "Total groups"]]
    st.dataframe(
        by_prov.reset_index().rename(columns={"region": ctx["geography"].title()}),
        hide_index=True,
    )
    st.caption(f"Data source: {q['source']}.")


def page_methodology(ctx: dict) -> None:
    page_header("Methodology", ctx)
    st.markdown(f"""
**Supabase Integration.** Synthetic demonstration records are stored in Supabase (`public.wellbeing_responses`) for all 9 provinces and 25 districts across 12 weeks. Province/district combinations are validated against the Sri Lankan geographic mapping.

**Theme classification & Aggregation.** Responses are categorized into 5 themes: Job and Money, Exams and Studies, Loneliness and Isolation, Family and Relationships, and Other.

**Privacy suppression.** District- and province-level groups with fewer than {K} responses are suppressed using `privacy.py`. Complementary suppression reduces simple subtraction risks; k-based suppression alone does not guarantee complete anonymity.

**Limitations.** All generated values are synthetic, are not estimates of actual community conditions, and are not clinical findings. Small-number suppression means this is a demonstration of method, not a source of real-world community findings.
""")


# =========================================================================
# 6. MAIN APP
# =========================================================================

def go_to(page: str) -> None:
    st.session_state["page"] = page


def reset_filters(provinces: list, themes: list, weeks: list) -> None:
    """Button callback: clear filter keys from session state cleanly."""
    st.session_state.pop("f_region", None)
    st.session_state.pop("f_theme", None)
    st.session_state.pop("f_week", None)


def chip_filter(title: str, options: list, key: str, all_label: bool = False) -> list:
    st.session_state.setdefault(key, list(options))
    n = len(st.session_state[key])
    count = "All" if (all_label and n == len(options)) else f"{n} of {len(options)}"
    st.sidebar.markdown(f'<div class="filter-label">{html.escape(title)}<span>{count}</span></div>',
                        unsafe_allow_html=True)
    if hasattr(st, "pills"):
        chosen = st.sidebar.pills(title, options, selection_mode="multi", key=key,
                                  label_visibility="collapsed") or []
    else:
        chosen = st.sidebar.multiselect(title, options, key=key, label_visibility="collapsed")
    return [o for o in options if o in chosen]

def _swap_geography_if_cleared() -> None:
    """Clicking the already-selected option deselects it (None); swap to the other one instead."""
    if st.session_state.get("f_geography") is None:
        previous = st.session_state.get("_previous_geography", "province")
        st.session_state["f_geography"] = "District" if previous == "province" else "Province"


def main() -> None:
    st.set_page_config(page_title="Community Wellbeing Signal Tracker - Sri Lanka",
                       page_icon=None,
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    render_topbar_and_banner()

    # ---- Manual Refresh Button & Data Loading from Supabase ----
    col_ref1, col_ref2 = st.columns([6, 1])
    with col_ref2:
        if st.button("🔄 Refresh Data", help="Clear cache and re-fetch from Supabase"):
            st.cache_data.clear()
            st.rerun()

    st.session_state.setdefault("f_geography", "Province")

    if hasattr(st.sidebar, "segmented_control"):
        geography_label = st.sidebar.segmented_control(
            "Geographic level",
            ("Province", "District"),
            key="f_geography",
            on_change=_swap_geography_if_cleared,
        )
    else:
        geography_label = st.sidebar.radio(
            "Geographic level",
            ("Province", "District"),
            horizontal=True,
            key="f_geography",
        )
    if geography_label not in {"Province", "District"}:
        raise ValueError("Select either Province or District.")
    geography_level = geography_label.lower()
    if st.session_state.get("_previous_geography") != geography_level:
        st.session_state.pop("f_region", None)
        st.session_state["_previous_geography"] = geography_level

    try:
        safe, summary, quality = load_supabase_data(geography_level)
    except Exception as exc:
        st.error(f"Failed to load data from Supabase: {exc}")
        st.error("Please verify your `.env` configuration (SUPABASE_URL and keys) and ensure Supabase is reachable.")
        st.stop()

    if safe.empty:
        st.error("The Supabase dataset returned 0 records.")
        st.stop()

    all_regions = PROVINCES if geography_level == "province" else DISTRICTS
    all_themes = THEMES
    week_order = sorted(safe["week"].unique().tolist(), key=_natural_key)

    st.session_state.setdefault("page", "Overview")
    page = st.session_state["page"]
    st.sidebar.markdown(f'<div class="brand"><div class="brand-icon">{svg("activity", 18)}</div>'
                        '<div class="brand-name">Signal Tracker</div></div>', unsafe_allow_html=True)
    for name, icon in NAV_ITEMS:
        nav_button = st.sidebar.button(name, key=f"nav_{name}", on_click=go_to, args=(name,),
                                       type="primary" if name == page else "secondary")

    st.sidebar.button("Reset", key="reset_filters", on_click=reset_filters,
                  args=(all_regions, all_themes, week_order))

    regions = chip_filter(geography_label, all_regions, "f_region")
    themes = chip_filter("Theme", all_themes, "f_theme", all_label=True)

    if len(week_order) > 1:
        st.sidebar.markdown('<div class="filter-label">Week range<span></span></div>', unsafe_allow_html=True)
        start, end = st.sidebar.select_slider("Week range", options=week_order,
                                              value=(week_order[0], week_order[-1]), key="f_week",
                                              format_func=week_label, label_visibility="collapsed")
    else:
        start = end = week_order[0]
    st.sidebar.caption(f"Source: {quality['source']}")
    st.sidebar.caption(f"Privacy: groups under {K} responses are hidden.")

    if page == "Methodology":
        page_methodology({})
        return
    if not regions or not themes:
        st.warning(f"Please select at least one {geography_level} and one theme in the sidebar.")
        return

    weeks = week_order[week_order.index(start): week_order.index(end) + 1]
    by_selection = safe[safe["region"].isin(regions) & safe["theme"].isin(themes)]
    filtered = by_selection[by_selection["week"].isin(weeks)]
    if filtered.empty:
        st.info("No data matches the current filters.")
        return

    alerts, alert_source = compute_alerts(by_selection, end, week_order)
    ctx = {"safe": safe, "filtered": filtered, "summary": summary, "quality": quality,
           "regions": regions, "all_regions": all_regions, "geography": geography_level,
           "themes": themes, "weeks": weeks, "end_week": end,
           "alerts": alerts, "alert_source": alert_source}

    {"Overview": page_overview, "Stressor Analysis": page_stressors,
     "Weekly Trends": page_trends, "Regional Comparison": page_regional,
     "Early Warning": page_alerts, "Privacy & Data Quality": page_privacy}[page](ctx)

    st.caption(
        "DEMONSTRATION MODE — ALL RECORDS ARE SYNTHETIC. Values do not represent "
        "actual wellbeing conditions in Sri Lanka."
    )


if __name__ == "__main__":
    main()