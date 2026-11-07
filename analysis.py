"""
analysis.py
Community Mental-Wellbeing Signal Tracker - statistical & evaluation layer

!! IMPORTANT !!
This module works ONLY with SYNTHETIC data (see generate_data.py). Its output is
a demonstration of how change-detection, Beta-Binomial credible intervals,
synthetic surge stress-testing, and Laplace differential-privacy noise evaluation
work. Nothing it produces is a validated real-world mental-health alert, a
finding about real communities, a diagnosis, or a risk score.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
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

# Public event calendar annotations (Proposal Section 5.1 Source B)
EVENT_CALENDAR = [
    {
        "start_week": 7,
        "end_week": 10,
        "label": "Cost-of-Living & Tariff Shift (W7–W10)",
        "color": "rgba(245, 158, 11, 0.12)",
        "border": "#D97706",
    },
    {
        "start_week": 11,
        "end_week": 12,
        "label": "University & National Exam Window (W11–W12)",
        "color": "rgba(99, 102, 241, 0.12)",
        "border": "#4F46E5",
    },
]

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


def compute_credible_interval(
    counts: Union[pd.Series, np.ndarray, float],
    totals: Union[pd.Series, np.ndarray, float],
    confidence: float = 0.95,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Beta-Binomial posterior credible intervals (in percentage points 0-100).

    Uses a Beta(prior_alpha, prior_beta) conjugate prior (default Beta(1, 1)
    uniform prior) updated with `counts` successes out of `totals` trials:
        posterior ~ Beta(prior_alpha + counts, prior_beta + totals - counts)
    Returns (lower_pct, upper_pct) arrays, with NaN where counts or totals are NaN/<=0.
    """
    c_arr = np.asarray(counts, dtype=float)
    n_arr = np.asarray(totals, dtype=float)
    valid = np.isfinite(c_arr) & np.isfinite(n_arr) & (n_arr > 0) & (c_arr >= 0) & (c_arr <= n_arr)

    lower = np.full_like(c_arr, np.nan, dtype=float)
    upper = np.full_like(c_arr, np.nan, dtype=float)
    if not np.any(valid):
        return lower, upper

    tail = (1.0 - confidence) / 2.0
    a_post = prior_alpha + c_arr[valid]
    b_post = prior_beta + (n_arr[valid] - c_arr[valid])
    lower[valid] = 100.0 * beta_dist.ppf(tail, a_post, b_post)
    upper[valid] = 100.0 * beta_dist.ppf(1.0 - tail, a_post, b_post)
    return lower, upper


def _distribution_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unsuppressed counts for EVERY province x week x theme combination.
    """
    columns = ["province", "week", "theme", "count", "total", "share_pct"]
    _require_columns(df, ["province", "week", "theme"])

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
    """
    window = raw[(raw["week"] >= latest_week - baseline_weeks) & (raw["week"] < latest_week)]
    window = window.assign(has_data=window["total"] > 0)
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

    data_type = df["data_type"].str.strip().str.lower()
    if not (data_type == "synthetic").all():
        raise ValueError(
            "All rows must have data_type = 'synthetic'. Refusing to analyse "
            "data that is missing this label or is labelled otherwise."
        )

    for col in ["response_id", "province", "theme", "language", "data_type"]:
        df[col] = df[col].str.strip().replace("", np.nan)
    df["data_type"] = "synthetic"
    df["language"] = df["language"].fillna("Unknown")

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
    Response count, percentage share, and 95% credible intervals by province and week.
    """
    dist = _distribution_raw(df)
    dist["suppressed"] = dist["total"] < min_sample_size
    ci_low, ci_high = compute_credible_interval(dist["count"], dist["total"])
    dist["ci_lower_pct"] = ci_low
    dist["ci_upper_pct"] = ci_high
    if suppress_small_cells and not dist.empty:
        dist["count"] = dist["count"].astype("Int64")
        dist.loc[dist["suppressed"], ["count", "share_pct", "ci_lower_pct", "ci_upper_pct"]] = np.nan
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

    n1 = out["sample_size"].astype(float).where(out["sample_size"] > 0)
    n2 = out["baseline_total"].astype(float).where(out["baseline_total"] > 0)
    x1 = out["current_count"].astype(float)
    x2 = out["baseline_count"].astype(float)
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se = se.where(se > 0)
    out["z_score"] = (x1 / n1 - x2 / n2) / se
    out["p_value"] = pd.Series(norm.sf(out["z_score"]), index=out.index)
    out["confidence_pct"] = (1 - out["p_value"]) * 100

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
    Flag simulated signals on synthetic data.
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
# 7. Synthetic Stress-Test & Laplace DP Evaluation (Proposal Section 9)
# --------------------------------------------------------------------------
def evaluate_surge_detection(
    sample_sizes: tuple[int, ...] = (50, 95, 150, 250),
    surge_sizes_pp: tuple[float, ...] = (0.0, 3.0, 5.0, 8.0, 10.0, 15.0),
    n_trials: int = 250,
    baseline_share: float = 0.22,
    watch_pp: float = 3.0,
    alert_pp: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Controlled synthetic surge stress-test (Proposal Section 9).

    Simulates 4 baseline weeks with true theme probability `baseline_share`
    and a test week with `baseline_share + surge_pp / 100`, then measures:
      - alert_rate_pct : % of trials triggering an ALERT (change >= alert_pp & p < 0.05)
      - watch_or_alert_rate_pct : % of trials triggering WATCH or ALERT (change >= watch_pp)
      - mean_detected_change_pp : average estimated percentage-point shift
    When surge_pp == 0.0, alert_rate_pct is the False-Alarm Rate (target < 20%).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for n in sample_sizes:
        for surge_pp in surge_sizes_pp:
            p_base = baseline_share
            p_curr = min(0.95, max(0.01, baseline_share + surge_pp / 100.0))

            # Shape: (n_trials, 4) for 4 baseline weeks, and (n_trials,) for current week
            base_counts = rng.binomial(n, p_base, size=(n_trials, 4))
            curr_counts = rng.binomial(n, p_curr, size=n_trials)

            # Mask cells below k = 10 just like the privacy gate
            base_pcts = np.where(base_counts >= MIN_SAMPLE_SIZE, 100.0 * base_counts / n, np.nan)
            curr_pcts = np.where(curr_counts >= MIN_SAMPLE_SIZE, 100.0 * curr_counts / n, np.nan)

            valid_base_n = np.sum(~np.isnan(base_pcts), axis=1)
            with np.errstate(invalid="ignore"):
                mean_base = np.nanmean(base_pcts, axis=1)
                std_base = np.nanstd(base_pcts, axis=1, ddof=1)

            spread = np.maximum(np.where(np.isnan(std_base), 0.0, std_base), 1.0)
            change = curr_pcts - mean_base
            z = change / spread
            p_val = norm.sf(z)

            valid_trial = (valid_base_n >= 2) & (~np.isnan(curr_pcts))
            is_alert = valid_trial & (change >= alert_pp) & (p_val < 0.05)
            is_watch_or_alert = valid_trial & (change >= watch_pp)

            rows.append(
                {
                    "sample_size_n": int(n),
                    "injected_surge_pp": float(surge_pp),
                    "scenario": "No surge (False-alarm check)" if surge_pp == 0 else f"+{surge_pp:.0f} pp surge",
                    "alert_rate_pct": round(100.0 * float(np.mean(is_alert)), 1),
                    "watch_or_alert_rate_pct": round(100.0 * float(np.mean(is_watch_or_alert)), 1),
                    "mean_estimated_change_pp": round(
                        float(np.nanmean(change)) if np.any(valid_trial) else 0.0, 2
                    ),
                }
            )

    return pd.DataFrame(rows)


def evaluate_laplace_privacy_cost(
    safe_df: pd.DataFrame,
    epsilon_values: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0),
    k: int = MIN_SAMPLE_SIZE,
    n_trials: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Measure the accuracy cost of adding calibrated Laplace(1/epsilon) noise
    to released aggregate counts on top of k=10 suppression (Proposal Section 6.5 & 9).

    Reports:
      - epsilon : privacy budget parameter (smaller = stronger privacy, more noise)
      - scale_b : Laplace scale 1 / epsilon (sensitivity = 1 response per cell)
      - mae_pp  : Mean Absolute Error in percentage share (percentage points)
      - rmse_pp : Root Mean Squared Error in percentage share (percentage points)
      - additional_suppressed_pct : % of previously visible cells pushed below k by noise
    """
    vis = safe_df.loc[~safe_df["suppressed"]].copy()
    if vis.empty:
        return pd.DataFrame(
            columns=["epsilon", "scale_b", "mae_pp", "rmse_pp", "additional_suppressed_pct"]
        )

    counts = vis["response_count"].astype(float).to_numpy()
    true_pct = vis["percentage_share"].astype(float).to_numpy()
    group_keys = (
        vis["region"].astype(str) + "__" + vis["week"].astype(str)
    ).to_numpy()

    # Map each row to its (region, week) group for fast vectorized normalization
    unique_groups, group_idx = np.unique(group_keys, return_inverse=True)
    rng = np.random.default_rng(seed)
    results = []

    for eps in epsilon_values:
        scale = 1.0 / float(eps)
        mae_list = []
        rmse_list = []
        newly_suppressed_list = []

        for _ in range(n_trials):
            noise = rng.laplace(loc=0.0, scale=scale, size=len(counts))
            noisy_counts = np.maximum(0.0, np.round(counts + noise))
            newly_suppressed = noisy_counts < k
            newly_suppressed_list.append(100.0 * float(np.mean(newly_suppressed)))

            group_totals = np.bincount(
                group_idx, weights=noisy_counts, minlength=len(unique_groups)
            )
            row_totals = group_totals[group_idx]
            noisy_pct = np.where(row_totals > 0, 100.0 * noisy_counts / row_totals, np.nan)
            diff = np.abs(noisy_pct - true_pct)
            mae_list.append(float(np.nanmean(diff)))
            rmse_list.append(float(np.sqrt(np.nanmean((noisy_pct - true_pct) ** 2))))

        results.append(
            {
                "epsilon": float(eps),
                "scale_b": round(scale, 2),
                "mae_pp": round(float(np.mean(mae_list)), 2),
                "rmse_pp": round(float(np.mean(rmse_list)), 2),
                "additional_suppressed_pct": round(float(np.mean(newly_suppressed_list)), 2),
            }
        )

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# 8. get_summary
# --------------------------------------------------------------------------
def get_summary(
    df: pd.DataFrame,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> dict:
    """
    Overall dashboard statistics as a plain dict.
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



