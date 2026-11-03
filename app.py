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
NAVY, BLUE, LIGHT_BLUE, BG = "#0B2545", "#1F5FA8", "#8DC3EE", "#F4F7FB"
THEME_COLOURS = {
    "Job and Money": "#0B2545",
    "Exams and Studies": "#1F5FA8",
    "Loneliness and Isolation": "#4FA3E0",
    "Family and Relationships": "#8DC3EE",
    "Other": "#A8B3C5",
}
LINE_COLOURS = ["#0B2545", "#1F78C8", "#4FA3E0", "#2A9D8F", "#E9A23B",
                "#C8553D", "#5B6B82", "#8DC3EE", "#7A4EAB"]

PAGES = ["Overview", "Stressor Analysis", "Weekly Trends", "Regional Comparison",
         "Early Warning", "Privacy and Data Quality", "Methodology"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp {{ font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif; }}
.stApp {{ background-color: {BG}; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1250px; }}
.app-header {{ background: linear-gradient(120deg, {NAVY} 0%, {BLUE} 100%);
  padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: .9rem; }}
.app-header h1 {{ color: #fff; font-size: 1.9rem; font-weight: 700; margin: 0 0 .25rem 0; padding: 0; }}
.app-header p {{ color: #D6E6F7; margin: 0; font-size: 1rem; }}
.demo-banner {{ background: #B42318; color: #fff; padding: .85rem 1.2rem; border-radius: 10px;
  font-weight: 700; font-size: .98rem; border-left: 8px solid #FDB022;
  margin-bottom: 1.2rem; line-height: 1.4; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem; margin: .5rem 0 1.2rem 0; }}
.kpi-card {{ background: #fff; border-radius: 12px; padding: 1rem 1.2rem; border: 1px solid #D9E3F0;
  border-top: 4px solid {BLUE}; box-shadow: 0 1px 3px rgba(11,37,69,.08); }}
.kpi-label {{ color: #5B6B82; font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
.kpi-value {{ color: {NAVY}; font-size: 1.65rem; font-weight: 700; margin-top: .25rem; line-height: 1.2; }}
.kpi-note {{ color: #5B6B82; font-size: .8rem; margin-top: .2rem; }}
.alert-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 1rem; margin-top: .5rem; }}
.alert-card {{ background: #fff; border-radius: 12px; padding: 1rem 1.2rem; border: 1px solid #D9E3F0;
  box-shadow: 0 1px 3px rgba(11,37,69,.08); }}
.alert-card.alert {{ border-left: 6px solid #D92D20; }}
.alert-card.watch {{ border-left: 6px solid #F79009; }}
.badge {{ display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .05em;
  padding: .18rem .55rem; border-radius: 999px; color: #fff; }}
.badge.alert {{ background: #D92D20; }}
.badge.watch {{ background: #F79009; }}
.alert-title {{ color: {NAVY}; font-weight: 700; font-size: 1.02rem; margin: .5rem 0 .6rem 0; }}
.metric-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: .45rem .8rem; }}
.metric-label {{ color: #5B6B82; font-size: .72rem; text-transform: uppercase; letter-spacing: .03em; }}
.metric-value {{ color: {NAVY}; font-weight: 600; font-size: .98rem; }}
.sim-tag {{ color: #5B6B82; font-size: .72rem; margin-top: .7rem; font-style: italic; }}
.section-note {{ color: #5B6B82; font-size: .88rem; }}
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
        f'<div class="kpi-card"><div class="kpi-label">{html.escape(str(l))}</div>'
        f'<div class="kpi-value">{html.escape(str(v))}</div>'
        f'<div class="kpi-note">{html.escape(str(n))}</div></div>'
        for l, v, n in cards)
    st.markdown(f'<div class="kpi-grid">{items}</div>', unsafe_allow_html=True)


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
    st.subheader("Overview")
    vis = visible_only(ctx["filtered"])
    totals = vis.groupby("theme")["response_count"].sum().sort_values(ascending=False)
    if totals.empty or totals.sum() == 0:
        dominant, note = "Not available", "All selected cells are suppressed"
    else:
        dominant = totals.index[0]
        note = f"{100 * totals.iloc[0] / totals.sum():.1f}% of released responses"
    n_alerts = int((ctx["alerts"]["status"].astype(str).str.upper() == "ALERT").sum())
    kpi_cards([
        ("Total synthetic responses", f"{totals.sum():,.0f}", "Released cells only (groups of 10 or more)"),
        ("Provinces", len(ctx["provinces"]), f"of {len(ctx['all_provinces'])} in the dataset"),
        ("Dominant stressor", dominant, note),
        ("Simulated alerts", n_alerts, f"Current: {week_label(ctx['end_week'])}"),
    ])
    st.markdown(
        "This dashboard helps regional wellbeing planners monitor **changes in community "
        "stressors** using aggregated, anonymised and **synthetic** data. Use the sidebar "
        "to choose provinces, themes and a week range, then explore the pages above."
    )
    st.caption("It reports stressor themes only. It gives no diagnoses and no treatment advice.")


def page_stressors(ctx: dict) -> None:
    st.subheader("Stressor Analysis")
    st.markdown('<p class="section-note">Share of released responses by stressor theme '
                'for the current filters.</p>', unsafe_allow_html=True)
    fig = build_donut(ctx["filtered"], ctx["themes"])
    if fig is None:
        st.info("No releasable data for this selection (all cells are below the minimum group size).")
    else:
        show_chart(fig)


def page_trends(ctx: dict) -> None:
    st.subheader("Weekly Trend Analysis")
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
    st.subheader("Regional Comparison")
    st.markdown('<p class="section-note">Average weekly percentage share of each theme by '
                'province, over the selected weeks. Privacy suppression is applied first.</p>',
                unsafe_allow_html=True)
    if visible_only(ctx["filtered"]).empty:
        st.info("No releasable data for this selection.")
        return
    show_chart(build_heatmap(ctx["filtered"], ctx["provinces"], ctx["themes"]))
    st.caption('A "-" means no releasable data (suppressed or missing) for that province and theme.')


def page_alerts(ctx: dict) -> None:
    st.subheader("Early Warning System")
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
    st.subheader("Privacy and Data Quality")
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
    st.subheader("Methodology")
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

def main() -> None:
    st.set_page_config(page_title="Community Mental-Wellbeing Signal Tracker",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    # ---- Header and permanent demonstration warning ----
    st.markdown(
        '<div class="app-header"><h1>Community Mental-Wellbeing Signal Tracker</h1>'
        '<p>Regional Early-Warning and Stressor Monitoring System &ndash; Sri Lanka</p></div>'
        '<div class="demo-banner">DEMONSTRATION MODE &ndash; SYNTHETIC DATA. All results are '
        'simulated and do not represent actual population conditions.</div>',
        unsafe_allow_html=True)

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

    # ---- Sidebar: navigation + filters ----
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("### Filters")
    provinces = st.sidebar.multiselect("Province", all_provinces, default=all_provinces)
    themes = st.sidebar.multiselect("Theme", all_themes, default=all_themes)
    if len(week_order) > 1:
        start, end = st.sidebar.select_slider("Week range", options=week_order,
                                              value=(week_order[0], week_order[-1]),
                                              format_func=week_label)
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
     "Early Warning": page_alerts, "Privacy and Data Quality": page_privacy}[page](ctx)

    st.caption("DEMONSTRATION MODE - SYNTHETIC DATA. No raw responses or individual records are shown.")


if __name__ == "__main__":
    main()