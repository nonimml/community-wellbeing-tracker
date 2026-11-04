"""
privacy.py - Privacy protection for the Community Mental-Wellbeing Signal Tracker.

HOW THE PRIVACY THRESHOLD WORKS
-------------------------------
The tracker only ever handles AGGREGATED data: one row per
(province, week, theme) cell, holding a response count and a percentage share.

A cell built from very few people can point back to those individuals
(e.g. "2 people in a small province mentioned 'loneliness' this week").
To prevent this we use a k-anonymity-style minimum group size rule:

    * K_THRESHOLD = 10
    * Any cell with response_count < 10 is SUPPRESSED:
      its count AND its percentage are replaced with missing values (NA),
      and the row is flagged `suppressed = True`.
    * Suppression happens BEFORE data reaches charts, tables or downloads,
      and the original values are not kept in the output, so they cannot
      leak through any downstream channel.

Extra safeguards:
    * Complementary suppression: if exactly ONE cell in a province-week is
      suppressed, its value could be back-calculated from totals/percentages
      of its neighbours. In that case the next-smallest visible cell in that
      province-week is also suppressed (see `complementary_suppression`).
    * Column whitelist: only the five aggregate columns are ever returned.
      Raw text, user IDs or any other individual-level column is dropped.
    * The data-quality summary contains only counts of cells and province
      names - never the suppressed numbers themselves.

Requires: pandas
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
K_THRESHOLD = 10  # Minimum number of responses a cell needs to be released.

REQUIRED_COLUMNS = ["province", "week", "theme", "response_count", "percentage_share"]
GROUP_KEYS = ["province", "week"]  # Cells are compared within these groups.


# --------------------------------------------------------------------------
# Core function
# --------------------------------------------------------------------------
def apply_privacy_filter(
    data: pd.DataFrame | list[dict[str, Any]],
    k: int = K_THRESHOLD,
    complementary_suppression: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Suppress small cells and return (safe_data, data_quality_summary).

    Parameters
    ----------
    data : DataFrame or list of dicts with columns:
           province, week, theme, response_count, percentage_share
    k    : minimum responses required to release a cell (default 10).
    complementary_suppression : also hide the smallest visible cell when only
           one cell in a province-week was suppressed, to stop the hidden
           value being recovered by subtraction.

    Returns
    -------
    safe_data : DataFrame with columns
                province, week, theme, response_count, percentage_share, suppressed
                Suppressed rows have response_count and percentage_share = NA.
    summary   : dict with suppressed/visible cell counts, percentage suppressed
                and provinces with insufficient data.
    """
    if k < 1:
        raise ValueError("k must be a positive integer.")

    df = pd.DataFrame(data)

    # 1. Validate input and enforce the column whitelist ------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        # Never pass through raw text or individual-level fields.
        warnings.warn(
            f"Dropping non-aggregate columns (not released): {extra}", stacklevel=2
        )
    df = df[REQUIRED_COLUMNS].copy()

    # 2. Primary suppression: count < k (missing/invalid counts are treated
    #    as unsafe and suppressed too) -------------------------------------
    counts = pd.to_numeric(df["response_count"], errors="coerce")
    suppressed = counts.isna() | (counts < k)

    # 3. Complementary (secondary) suppression ----------------------------
    if complementary_suppression:
        for _, idx in df.groupby(GROUP_KEYS).groups.items():
            idx = list(idx)
            if suppressed.loc[idx].sum() == 1:
                visible_idx = [i for i in idx if not suppressed.loc[i]]
                if visible_idx:
                    # Hide the smallest remaining cell (it is closest to k).
                    suppressed.loc[counts.loc[visible_idx].idxmin()] = True

    # 4. Build the release copy: overwrite sensitive values with NA --------
    safe = df.copy()
    safe["response_count"] = counts.astype("Int64")
    safe["percentage_share"] = pd.to_numeric(safe["percentage_share"], errors="coerce")
    safe.loc[suppressed, ["response_count", "percentage_share"]] = pd.NA
    safe["suppressed"] = suppressed
    safe = safe.reset_index(drop=True)

    # 5. Data-quality summary (no raw values) -------------------------------
    summary = _build_summary(safe)
    return safe, summary


def _build_summary(safe: pd.DataFrame) -> dict[str, Any]:
    total = len(safe)
    n_suppressed = int(safe["suppressed"].sum())
    n_visible = total - n_suppressed

    by_province = safe.groupby("province")["suppressed"].agg(["sum", "count"])
    # "Insufficient data" = every cell for that province was suppressed.
    insufficient = sorted(by_province[by_province["sum"] == by_province["count"]].index)
    partial = sorted(by_province[by_province["sum"] > 0].index)

    return {
        "k_threshold": K_THRESHOLD,
        "suppressed_cells": n_suppressed,
        "visible_cells": n_visible,
        "percent_suppressed": round(100 * n_suppressed / total, 1) if total else 0.0,
        "provinces_with_insufficient_data": insufficient,  # all cells suppressed
        "provinces_with_some_suppression": partial,
    }


# --------------------------------------------------------------------------
# Helpers for dashboards and downloads (always use filtered data)
# --------------------------------------------------------------------------
def chart_data(safe: pd.DataFrame) -> pd.DataFrame:
    """Rows safe to plot. Suppressed cells are excluded entirely from charts."""
    return safe.loc[~safe["suppressed"]].drop(columns="suppressed").reset_index(drop=True)


def table_data(safe: pd.DataFrame, placeholder: str = "Suppressed (n<10)") -> pd.DataFrame:
    """Display table: suppressed cells show a label, never a number."""
    out = safe.astype({"response_count": "object", "percentage_share": "object"})
    out.loc[out["suppressed"], ["response_count", "percentage_share"]] = placeholder
    return out.drop(columns="suppressed")


def export_aggregate_report(safe: pd.DataFrame, path: str) -> None:
    """Write a downloadable CSV. Suppressed values are blank, never real numbers."""
    safe.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------
def test_no_group_below_k_is_released() -> None:
    """Confirms no cell with fewer than k = 10 responses is ever released."""
    sample = [
        {"province": "Western", "week": "2026-W40", "theme": "stress", "response_count": 120, "percentage_share": 40.0},
        {"province": "Western", "week": "2026-W40", "theme": "sleep", "response_count": 10, "percentage_share": 3.3},
        {"province": "Northern", "week": "2026-W40", "theme": "stress", "response_count": 9, "percentage_share": 45.0},
        {"province": "Northern", "week": "2026-W40", "theme": "sleep", "response_count": 3, "percentage_share": 15.0},
        # Only one small cell in this group -> complementary suppression applies.
        {"province": "Central", "week": "2026-W40", "theme": "stress", "response_count": 50, "percentage_share": 60.0},
        {"province": "Central", "week": "2026-W40", "theme": "sleep", "response_count": 25, "percentage_share": 30.0},
        {"province": "Central", "week": "2026-W40", "theme": "loneliness", "response_count": 4, "percentage_share": 5.0},
        # Extra column simulating raw text: must never be released.
        # (added below)
    ]
    df = pd.DataFrame(sample)
    df["free_text"] = "private comment"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        safe, summary = apply_privacy_filter(df)

    # No released count is below k.
    released = safe.loc[~safe["suppressed"], "response_count"]
    assert (released >= K_THRESHOLD).all(), "A cell below k was released!"

    # Every suppressed cell has no count or percentage.
    hidden = safe[safe["suppressed"]]
    assert hidden["response_count"].isna().all()
    assert hidden["percentage_share"].isna().all()

    # Original small values are absent from every output channel.
    assert "free_text" not in safe.columns
    assert chart_data(safe)["response_count"].min() >= K_THRESHOLD
    assert not any(
        v in (9, 3, 4) for v in table_data(safe)["response_count"] if isinstance(v, int)
    )

    # Summary checks.
    assert summary["suppressed_cells"] + summary["visible_cells"] == len(df)
    assert "Northern" in summary["provinces_with_insufficient_data"]
    assert "Central" in summary["provinces_with_some_suppression"]

    print("PASS: no group below k =", K_THRESHOLD, "was released.")
    print(summary)


if __name__ == "__main__":
    test_no_group_below_k_is_released()