"""
analysis.py
Community Mental-Wellbeing Signal Tracker - analysis layer

!! IMPORTANT !!
This module works ONLY with SYNTHETIC data (see generate_data.py). Its output is
a demonstration of how change-detection could work. Nothing it produces is a
validated real-world mental-health alert, a finding about real communities, a
diagnosis, or a risk score. It works with aggregate counts and percentages only:
it never reads, stores or returns free-text responses, and it never scores
individuals.

Design notes (for Streamlit use)
  * Every function takes a DataFrame (from load_data) and returns plain
    DataFrames / dicts. No printing, no global state, no file access except
    load_data(), so each function can be wrapped in @st.cache_data.
  * Small cells (fewer than MIN_SAMPLE_SIZE responses) are suppressed: their
    counts and percentages are replaced with NaN so they cannot be displayed.

Statistical method (used by detect_changes)
  For one province and theme we compare:
      current week   : x1 theme responses out of n1 responses that week
      baseline       : x2 theme responses out of n2 responses in the previous
                       four weeks (pooled, so busier weeks count for more)
  We run a ONE-SIDED two-proportion z-test of
      H0: current share <= baseline share    vs    H1: current share > baseline share
  and report
      p_value        : probability of seeing an increase at least this large by
                       random sampling variation alone, if H0 were true.
      confidence_pct : (1 - p_value) * 100. A rough "strength of evidence"
                       score. It is NOT the probability that the increase is
                       real, and it ignores that many province/theme pairs are
                       tested at once (some false signals are expected).
  The z-test uses a normal approximation, which is why a minimum sample size
  is required before any result is reported.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

# --------------------------------------------------------------------------
# Configuration (all can be overridden through function arguments)
# --------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "synthetic_responses.csv")
REQUIRED_COLUMNS = [
    "response_id", "province", "week", "response_date",
    "theme", "language", "data_type",
]
BASELINE_WEEKS = 4        # number of previous weeks that form the baseline
MIN_SAMPLE_SIZE = 10      # minimum responses needed to report a share / alert
MIN_CHANGE_PP = 10.0      # minimum increase (percentage points) for an alert
ALPHA = 0.05              # one-sided p-value must be below this for an alert

DISCLAIMER = (
    "All data shown is SYNTHETIC and for demonstration only. Alerts are "
    "simulated statistical signals, not validated real-world mental-health "
    "alerts, diagnoses or risk scores."
)

PathOrBuffer = Union[str, "os.PathLike[str]", object]

_DETECT_COLUMNS = [
    "province", "theme", "week", "baseline_share_pct", "current_share_pct",
    "change_pp", "sample_size", "baseline_total", "z_score", "p_value",
    "confidence_pct",
]
_ALERT_COLUMNS = [
    "province", "theme", "baseline_share_pct", "current_share_pct",
    "change_pp", "sample_size", "alert_status", "confidence_pct",
    "p_value", "is_alert",
]


# --------------------------------------------------------------------------
# Small internal helpers
# --------------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, columns: list) -> None:
    """Raise a clear error if the DataFrame lacks columns we need."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {missing}")


def _pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Percentage = numerator / denominator * 100, NaN when denominator is 0."""
    return numerator.div(denominator.where(denominator > 0)) * 100


def _distribution_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unsuppressed counts for EVERY province x week x theme combination.

    Combinations with no responses get count 0 (rather than being absent), and
    weeks with no data at all are kept so that "the previous four weeks" always
    means four calendar weeks. share_pct is NaN when the province-week has no
    responses. Internal use only: public functions apply small-cell suppression.
    """
    columns = ["province", "week", "theme", "count", "total", "share_pct"]
    _require_columns(df, ["province", "week", "theme"])

    # Ignore rows missing any key field instead of failing.
    clean = df.dropna(subset=["province", "week", "theme"])
    if clean.empty:
        return pd.DataFrame(columns=columns)

    provinces = sorted(clean["province"].unique())
    themes = sorted(clean["theme"].unique())
    weeks = range(int(clean["week"].min()), int(clean["week"].max()) + 1)
    grid = pd.MultiIndex.from_product(
        [provinces, weeks, themes], names=["province", "week", "theme"]
    )

    counts = (
        clean.groupby(["province", "week", "theme"]).size()
        .reindex(grid, fill_value=0)
        .rename("count")
        .reset_index()
    )
    counts["total"] = counts.groupby(["province", "week"])["count"].transform("sum")
    counts["share_pct"] = _pct(counts["count"], counts["total"])
    return counts[columns]


def _resolve_latest_week(raw: pd.DataFrame, latest_week: Optional[int]) -> int:
    """Use the requested week, or default to the latest week in the data."""
    return int(latest_week) if latest_week is not None else int(raw["week"].max())


def _baseline_raw(raw: pd.DataFrame, latest_week: int, baseline_weeks: int) -> pd.DataFrame:
    """
    Pooled counts over the `baseline_weeks` weeks BEFORE latest_week.

    For each province and theme:
      baseline_count          = theme responses in the window
      baseline_total          = all responses in the province in the window
      baseline_weeks_with_data= weeks in the window with at least one response
    """
    window = raw[(raw["week"] >= latest_week - baseline_weeks) & (raw["week"] < latest_week)]
    window = window.assign(has_data=window["total"] > 0)
    # For a fixed (province, theme) each week appears once, so summing "total"
    # gives the province's total responses across the window.
    return (
        window.groupby(["province", "theme"])
        .agg(
            baseline_count=("count", "sum"),
            baseline_total=("total", "sum"),
            baseline_weeks_with_data=("has_data", "sum"),
        )
        .reset_index()
    )


# --------------------------------------------------------------------------
# 1. load_data
# --------------------------------------------------------------------------
def load_data(source: PathOrBuffer = DATA_PATH) -> pd.DataFrame:
    """
    Load the synthetic CSV and validate it.

    `source` may be a file path or a file-like object (e.g. the object returned
    by Streamlit's st.file_uploader). Handling of problems:
      * missing file / missing columns / non-synthetic rows -> clear error
      * rows missing province, week, theme or response_id    -> dropped
      * missing language                                      -> "Unknown"
      * unparseable dates                                     -> NaT (row kept)
      * duplicate response_id                                 -> first one kept
    Only the seven expected columns are kept.
    """
    if isinstance(source, (str, os.PathLike)) and not os.path.exists(source):
        raise FileNotFoundError(
            f"Data file not found: {source}. Run generate_data.py first."
        )

    df = pd.read_csv(source, dtype=str)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}")
    df = df[REQUIRED_COLUMNS].copy()

    # Safety check: this tool must only ever analyse synthetic data.
    data_type = df["data_type"].str.strip().str.lower()
    if not (data_type == "synthetic").all():
        raise ValueError(
            "All rows must have data_type = 'synthetic'. Refusing to analyse "
            "data that is missing this label or is labelled otherwise."
        )

    # Tidy text columns; blank strings count as missing.
    for col in ["response_id", "province", "theme", "language", "data_type"]:
        df[col] = df[col].str.strip().replace("", np.nan)
    df["data_type"] = "synthetic"
    df["language"] = df["language"].fillna("Unknown")

    # Week must be a whole number; anything else is treated as missing.
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df.loc[df["week"] % 1 != 0, "week"] = np.nan
    df["response_date"] = pd.to_datetime(df["response_date"], errors="coerce")

    df = df.dropna(subset=["response_id", "province", "week", "theme"])
    df = df.drop_duplicates(subset="response_id", keep="first")
    df["week"] = df["week"].astype(int)

    if df.empty:
        raise ValueError("No valid rows remain after cleaning the data.")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. calculate_theme_distribution
# --------------------------------------------------------------------------
def calculate_theme_distribution(
    df: pd.DataFrame,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    suppress_small_cells: bool = True,
) -> pd.DataFrame:
    """
    Response count and percentage share of each theme, by province and week.

    Columns: province, week, theme, count, total, share_pct, suppressed.
    `total` is the number of responses in that province-week. When `total` is
    below `min_sample_size`, `suppressed` is True and (if suppress_small_cells)
    count and share_pct are set to missing so small groups are not exposed.
    """
    dist = _distribution_raw(df)
    dist["suppressed"] = dist["total"] < min_sample_size
    if suppress_small_cells and not dist.empty:
        dist["count"] = dist["count"].astype("Int64")
        dist.loc[dist["suppressed"], ["count", "share_pct"]] = np.nan
    return dist


# --------------------------------------------------------------------------
# 3. calculate_weekly_trends
# --------------------------------------------------------------------------
def calculate_weekly_trends(
    df: pd.DataFrame,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    suppress_small_cells: bool = True,
) -> pd.DataFrame:
    """
    Weekly theme percentages per province in wide format (one column per theme).

    Columns: province, week, sample_size, <one column per theme>.
    Ready for line charts, e.g. filter one province and plot week vs themes.
    Suppressed province-weeks have NaN percentages.
    """
    dist = calculate_theme_distribution(df, min_sample_size, suppress_small_cells)
    if dist.empty:
        return pd.DataFrame(columns=["province", "week", "sample_size"])

    wide = dist.pivot(index=["province", "week"], columns="theme", values="share_pct")
    wide.columns.name = None
    wide = wide.reset_index()

    sizes = (
        dist.drop_duplicates(["province", "week"])[["province", "week", "total"]]
        .rename(columns={"total": "sample_size"})
    )
    wide = wide.merge(sizes, on=["province", "week"])
    theme_cols = [c for c in wide.columns if c not in ("province", "week", "sample_size")]
    return wide[["province", "week", "sample_size"] + theme_cols]


# --------------------------------------------------------------------------
# 4. calculate_baseline
# --------------------------------------------------------------------------
def calculate_baseline(
    df: pd.DataFrame,
    latest_week: Optional[int] = None,
    baseline_weeks: int = BASELINE_WEEKS,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Previous four-week average share of each theme, per province.

    The window is the `baseline_weeks` weeks before `latest_week` (default: the
    latest week in the data, which is excluded from its own baseline). The
    average is pooled: baseline share = theme responses / all responses over
    the whole window, so weeks with more responses carry more weight.

    Columns: province, theme, baseline_count, baseline_total,
    baseline_weeks_with_data, baseline_share_pct. If baseline_total is below
    `min_sample_size`, count and share are suppressed (NaN).
    """
    columns = ["province", "theme", "baseline_count", "baseline_total",
               "baseline_weeks_with_data", "baseline_share_pct"]
    raw = _distribution_raw(df)
    if raw.empty:
        return pd.DataFrame(columns=columns)

    latest = _resolve_latest_week(raw, latest_week)
    base = _baseline_raw(raw, latest, baseline_weeks)
    base["baseline_share_pct"] = _pct(base["baseline_count"], base["baseline_total"])

    small = base["baseline_total"] < min_sample_size
    base["baseline_count"] = base["baseline_count"].astype("Int64")
    base.loc[small, ["baseline_count", "baseline_share_pct"]] = np.nan
    return base[columns]


# --------------------------------------------------------------------------
# 5. detect_changes
# --------------------------------------------------------------------------
def detect_changes(
    df: pd.DataFrame,
    latest_week: Optional[int] = None,
    baseline_weeks: int = BASELINE_WEEKS,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Compare the latest week's theme share with its previous four-week baseline.

    One row per province x theme. Columns:
      baseline_share_pct, current_share_pct, change_pp (percentage points),
      sample_size (responses in the province in the latest week),
      baseline_total (responses in the baseline window),
      z_score, p_value, confidence_pct (see the module docstring: one-sided
      two-proportion z-test; confidence_pct = (1 - p_value) * 100).

    Privacy: if the latest week has fewer than `min_sample_size` responses the
    current share is hidden; if either the latest week or the baseline has
    fewer than `min_sample_size` responses, the baseline, change and test
    results are hidden (NaN).
    """
    raw = _distribution_raw(df)
    if raw.empty:
        return pd.DataFrame(columns=_DETECT_COLUMNS)

    latest = _resolve_latest_week(raw, latest_week)

    current = (
        raw.loc[raw["week"] == latest, ["province", "theme", "count", "total", "share_pct"]]
        .rename(columns={"count": "current_count", "total": "sample_size",
                         "share_pct": "current_share_pct"})
    )
    base = _baseline_raw(raw, latest, baseline_weeks)
    out = current.merge(base, on=["province", "theme"], how="left")
    out[["baseline_count", "baseline_total"]] = out[["baseline_count", "baseline_total"]].fillna(0)
    out["baseline_share_pct"] = _pct(out["baseline_count"], out["baseline_total"])
    out["week"] = latest
    out["change_pp"] = out["current_share_pct"] - out["baseline_share_pct"]

    # One-sided two-proportion z-test (H1: current share > baseline share).
    n1 = out["sample_size"].astype(float).where(out["sample_size"] > 0)
    n2 = out["baseline_total"].astype(float).where(out["baseline_total"] > 0)
    x1 = out["current_count"].astype(float)
    x2 = out["baseline_count"].astype(float)
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se = se.where(se > 0)                       # SE of 0 -> test not computable
    out["z_score"] = (x1 / n1 - x2 / n2) / se
    out["p_value"] = pd.Series(norm.sf(out["z_score"]), index=out.index)
    out["confidence_pct"] = (1 - out["p_value"]) * 100

    # Small-cell suppression.
    small_current = out["sample_size"] < min_sample_size
    small_baseline = out["baseline_total"] < min_sample_size
    out.loc[small_current, "current_share_pct"] = np.nan
    out.loc[small_current | small_baseline,
            ["baseline_share_pct", "change_pp", "z_score", "p_value", "confidence_pct"]] = np.nan

    return out[_DETECT_COLUMNS].sort_values(["province", "theme"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 6. generate_alerts
# --------------------------------------------------------------------------
def generate_alerts(
    df: pd.DataFrame,
    latest_week: Optional[int] = None,
    baseline_weeks: int = BASELINE_WEEKS,
    min_change_pp: float = MIN_CHANGE_PP,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    alpha: float = ALPHA,
    alerts_only: bool = False,
) -> pd.DataFrame:
    """
    Flag simulated signals: a theme whose latest share is at least
    `min_change_pp` percentage points above its baseline, with sample size
    >= `min_sample_size`, and a one-sided p-value below `alpha`.

    Returns province, theme, baseline_share_pct, current_share_pct, change_pp,
    sample_size, alert_status, confidence_pct, p_value, is_alert.

    alert_status is one of:
      "ALERT"                   all three criteria met
      "No alert"                reportable, but criteria not met
      "Suppressed (n < N)"      too few responses in the latest week
      "Insufficient baseline"   too few responses in the baseline window

    These are simulated signals on synthetic data, not real-world alerts.
    """
    changes = detect_changes(df, latest_week, baseline_weeks, min_sample_size)
    if changes.empty:
        return pd.DataFrame(columns=_ALERT_COLUMNS)

    small_current = changes["sample_size"] < min_sample_size
    small_baseline = changes["baseline_total"].fillna(0) < min_sample_size
    strong = (changes["change_pp"] >= min_change_pp) & (changes["p_value"] < alpha)

    changes["alert_status"] = np.select(
        [small_current, small_baseline, strong],
        [f"Suppressed (n < {min_sample_size})", "Insufficient baseline", "ALERT"],
        default="No alert",
    )
    changes["is_alert"] = changes["alert_status"] == "ALERT"

    out = changes[_ALERT_COLUMNS].copy()
    out = out.sort_values(["is_alert", "change_pp"], ascending=[False, False],
                          na_position="last").reset_index(drop=True)
    out = out.round({"baseline_share_pct": 2, "current_share_pct": 2, "change_pp": 2,
                     "confidence_pct": 1, "p_value": 4})
    if alerts_only:
        out = out[out["is_alert"]].reset_index(drop=True)
    out.attrs["disclaimer"] = DISCLAIMER
    return out


# --------------------------------------------------------------------------
# 7. get_summary
# --------------------------------------------------------------------------
def get_summary(
    df: pd.DataFrame,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> dict:
    """
    Overall dashboard statistics as a plain dict (easy to show with st.metric).

    Contains only aggregates: totals, counts by province / theme / language,
    the latest week, provinces suppressed in the latest week, the number of
    simulated alerts and the disclaimer text.
    """
    _require_columns(df, ["province", "week", "theme", "language", "response_date"])

    def counts(col: str) -> dict:
        return {str(k): int(v) for k, v in df[col].value_counts().items()}

    summary = {
        "total_responses": int(len(df)),
        "n_provinces": int(df["province"].nunique()),
        "n_themes": int(df["theme"].nunique()),
        "n_weeks": int(df["week"].nunique()),
        "first_week": None, "latest_week": None,
        "first_date": None, "last_date": None,
        "latest_week_responses": 0,
        "suppressed_provinces_latest_week": [],
        "alerts_count": 0,
        "responses_by_province": counts("province"),
        "responses_by_theme": counts("theme"),
        "responses_by_language": counts("language"),
        "disclaimer": DISCLAIMER,
    }
    if df.empty:
        return summary

    latest = int(df["week"].max())
    summary["first_week"] = int(df["week"].min())
    summary["latest_week"] = latest
    if df["response_date"].notna().any():
        summary["first_date"] = df["response_date"].min().strftime("%Y-%m-%d")
        summary["last_date"] = df["response_date"].max().strftime("%Y-%m-%d")

    # Provinces with too few latest-week responses (includes provinces with none).
    totals = (
        _distribution_raw(df).loc[lambda d: d["week"] == latest]
        .drop_duplicates("province").set_index("province")["total"]
    )
    summary["latest_week_responses"] = int(totals.sum())
    summary["suppressed_provinces_latest_week"] = sorted(
        totals[totals < min_sample_size].index.tolist()
    )
    summary["alerts_count"] = int(
        generate_alerts(df, alerts_only=True, min_sample_size=min_sample_size).shape[0]
    )
    return summary


# --------------------------------------------------------------------------
# Command-line demo (not used by Streamlit)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print(DISCLAIMER, "\n")

    print("Summary:")
    for key, value in get_summary(data).items():
        if key != "disclaimer":
            print(f"  {key}: {value}")

    print("\nAlert table (latest week vs previous 4 weeks):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(generate_alerts(data).to_string(index=False))