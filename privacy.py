"""Small-cell privacy protections for synthetic aggregate outputs.

Counts below k=10 are hidden before data reaches the dashboard. The combined
province/district helper also applies complementary suppression so a released
province total cannot reveal one hidden district cell by subtraction. This is
a useful safeguard, not a guarantee of complete anonymity.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from geography import DISTRICT_TO_PROVINCE, DISTRICTS, PROVINCES

K_THRESHOLD = 10
WEEK_NUMBERS = tuple(range(1, 13))
THEMES = (
    "Job and Money",
    "Exams and Studies",
    "Loneliness and Isolation",
    "Family and Relationships",
    "Other",
)

# Kept for compatibility with callers that provide province-level aggregates.
REQUIRED_COLUMNS = [
    "province",
    "week",
    "theme",
    "response_count",
    "percentage_share",
]
OUTPUT_COLUMNS = [
    "region",
    "province",
    "district",
    "week",
    "theme",
    "response_count",
    "percentage_share",
]
_ALLOWED_COLUMNS = set(OUTPUT_COLUMNS)


def _is_synthetic(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _build_raw_aggregates(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create complete province- and district-week-theme count grids."""
    df = pd.DataFrame(records).copy()
    if "week_number" in df.columns and "week" not in df.columns:
        df = df.rename(columns={"week_number": "week"})
    if df.empty and not len(df.columns):
        df = pd.DataFrame(columns=["province", "district", "week", "theme"])
    required = {"province", "district", "week", "theme"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required response columns: {missing}")

    if "is_synthetic" in df.columns:
        df = df.loc[df["is_synthetic"].map(_is_synthetic)].copy()

    df = df.dropna(subset=["province", "district", "week", "theme"])
    if not df.empty:
        valid_pairs = df.apply(
            lambda row: DISTRICT_TO_PROVINCE.get(row["district"]) == row["province"],
            axis=1,
        )
        if not valid_pairs.all():
            invalid = df.loc[~valid_pairs, ["province", "district"]].drop_duplicates()
            raise ValueError(
                "Records contain invalid province/district pairs: "
                f"{invalid.to_dict(orient='records')}"
            )

        numeric_weeks = pd.to_numeric(df["week"], errors="coerce")
        if numeric_weeks.isna().any() or not numeric_weeks.isin(WEEK_NUMBERS).all():
            raise ValueError("week/week_number values must be integers from 1 to 12.")
        df["week"] = numeric_weeks.astype(int)
        if not df["theme"].isin(THEMES).all():
            raise ValueError("Records contain an unsupported wellbeing theme.")

    if df.empty:
        counts = pd.DataFrame(columns=["province", "district", "week", "theme"])
    else:
        counts = (
            df.groupby(["province", "district", "week", "theme"], observed=True)
            .size()
            .rename("response_count")
            .reset_index()
        )

    province_grid = pd.MultiIndex.from_product(
        [PROVINCES, WEEK_NUMBERS, THEMES],
        names=["province", "week", "theme"],
    ).to_frame(index=False)
    province_counts = (
        counts.groupby(["province", "week", "theme"], observed=True)
        ["response_count"].sum().reset_index()
        if not counts.empty
        else pd.DataFrame(columns=["province", "week", "theme", "response_count"])
    )
    province_grid = province_grid.merge(
        province_counts,
        on=["province", "week", "theme"],
        how="left",
    )
    province_grid["response_count"] = province_grid["response_count"].fillna(0).astype(int)
    province_grid["region"] = province_grid["province"]
    province_grid["district"] = pd.NA

    district_grid = pd.MultiIndex.from_product(
        [DISTRICTS, WEEK_NUMBERS, THEMES],
        names=["district", "week", "theme"],
    ).to_frame(index=False)
    district_grid["province"] = district_grid["district"].map(DISTRICT_TO_PROVINCE)
    district_grid = district_grid.merge(
        counts[["district", "week", "theme", "response_count"]]
        if not counts.empty
        else pd.DataFrame(columns=["district", "week", "theme", "response_count"]),
        on=["district", "week", "theme"],
        how="left",
    )
    district_grid["response_count"] = district_grid["response_count"].fillna(0).astype(int)
    district_grid["region"] = district_grid["district"]

    for aggregate in (province_grid, district_grid):
        totals = aggregate.groupby(["region", "week"])["response_count"].transform("sum")
        aggregate["percentage_share"] = (
            100 * aggregate["response_count"].div(totals.where(totals > 0))
        )

    columns = [
        "region",
        "province",
        "district",
        "week",
        "theme",
        "response_count",
        "percentage_share",
    ]
    return province_grid[columns], district_grid[columns]


def apply_privacy_filter(
    data: pd.DataFrame | list[dict[str, Any]],
    k: int = K_THRESHOLD,
    complementary_suppression: bool = True,
    geography: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Suppress counts and percentages below k in supplied aggregate rows.

    The historical province schema remains supported. Newer callers should
    pass aggregate rows with a `region` column and set `geography` to the
    grouping key used for complementary suppression.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    df = pd.DataFrame(data).copy()
    if geography is None:
        geography = "region" if "region" in df.columns else "province"
    if geography not in {"region", "province", "district"}:
        raise ValueError("geography must be 'province', 'district', or 'region'.")

    required = {
        geography,
        "week",
        "theme",
        "response_count",
        "percentage_share",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required aggregate columns: {missing}")

    extras = sorted(set(df.columns) - _ALLOWED_COLUMNS)
    if extras:
        warnings.warn(
            f"Dropping non-aggregate columns (not released): {extras}",
            stacklevel=2,
        )
    columns = [column for column in OUTPUT_COLUMNS if column in df.columns]
    df = df[columns].dropna(subset=[geography, "week", "theme"]).copy()

    counts = pd.to_numeric(df["response_count"], errors="coerce")
    valid_counts = counts.notna() & counts.ge(0) & counts.mod(1).eq(0)
    suppressed = ~valid_counts | counts.lt(k)

    if complementary_suppression and not df.empty:
        for _, indices in df.groupby([geography, "week"], dropna=False).groups.items():
            indices = list(indices)
            suppressed_indices = [index for index in indices if suppressed.loc[index]]
            visible_indices = [index for index in indices if not suppressed.loc[index]]
            if len(suppressed_indices) == 1 and visible_indices:
                smallest_visible = counts.loc[visible_indices].idxmin()
                suppressed.loc[smallest_visible] = True

    safe = df.copy()
    safe["response_count"] = counts.astype("Int64")
    safe["percentage_share"] = pd.to_numeric(
        safe["percentage_share"], errors="coerce"
    )
    invalid_percentages = (
        safe["percentage_share"].notna()
        & ~safe["percentage_share"].between(0, 100, inclusive="both")
    )
    suppressed |= invalid_percentages
    safe.loc[suppressed, ["response_count", "percentage_share"]] = pd.NA
    safe["suppressed"] = suppressed.astype(bool)
    safe = safe.reset_index(drop=True)
    return safe, _build_summary(safe, geography, k)


def _build_summary(
    safe: pd.DataFrame,
    geography: str,
    k: int = K_THRESHOLD,
) -> dict[str, Any]:
    total_cells = len(safe)
    suppressed_cells = int(safe["suppressed"].sum()) if total_cells else 0
    grouped = (
        safe.groupby(geography)["suppressed"].agg(["sum", "count"])
        if total_cells
        else pd.DataFrame(columns=["sum", "count"])
    )
    insufficient = sorted(
        str(region)
        for region in grouped.index[grouped["sum"] == grouped["count"]]
    )
    partial = sorted(
        str(region) for region in grouped.index[grouped["sum"] > 0]
    )
    summary = {
        "k_threshold": k,
        "geography": geography,
        "suppressed_cells": suppressed_cells,
        "visible_cells": total_cells - suppressed_cells,
        "percent_suppressed": (
            round(100 * suppressed_cells / total_cells, 1) if total_cells else 0.0
        ),
        "regions_with_insufficient_data": insufficient,
        "regions_with_some_suppression": partial,
    }
    if geography == "province":
        summary["provinces_with_insufficient_data"] = insufficient
        summary["provinces_with_some_suppression"] = partial
    elif geography == "district":
        summary["districts_with_insufficient_data"] = insufficient
        summary["districts_with_some_suppression"] = partial
    return summary


def _apply_cross_level_suppression(
    province_safe: pd.DataFrame,
    district_safe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hide a second district cell when its visible province total has one gap."""
    districts = district_safe.copy()
    province_visible = province_safe.loc[~province_safe["suppressed"]]

    for parent in province_visible.itertuples(index=False):
        children = districts.loc[
            (districts["province"] == parent.region)
            & (districts["week"] == parent.week)
            & (districts["theme"] == parent.theme)
        ]
        hidden_indices = children.index[children["suppressed"]].tolist()
        visible_indices = children.index[~children["suppressed"]].tolist()
        if len(hidden_indices) == 1 and visible_indices:
            smallest_visible = districts.loc[
                visible_indices, "response_count"
            ].idxmin()
            districts.loc[
                smallest_visible, ["response_count", "percentage_share"]
            ] = pd.NA
            districts.loc[smallest_visible, "suppressed"] = True

    return province_safe, districts


def aggregate_response_levels(
    records: pd.DataFrame | list[dict[str, Any]],
    k: int = K_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Return privacy-filtered province and district aggregate grids.

    Use this combined helper when both geographic levels may be displayed. It
    coordinates complementary suppression across levels to reduce subtraction
    attacks using province totals and district cells.
    """
    province_raw, district_raw = _build_raw_aggregates(pd.DataFrame(records))
    province_safe, province_summary = apply_privacy_filter(
        province_raw, k=k, geography="region"
    )
    district_safe, district_summary = apply_privacy_filter(
        district_raw, k=k, geography="region"
    )
    province_safe, district_safe = _apply_cross_level_suppression(
        province_safe, district_safe
    )
    province_summary = _build_summary(province_safe, "region", k)
    district_summary = _build_summary(district_safe, "region", k)
    province_summary["geography"] = "province"
    district_summary["geography"] = "district"
    return province_safe, district_safe, province_summary, district_summary


def aggregate_response_records(
    records: pd.DataFrame | list[dict[str, Any]],
    geography: str = "province",
    k: int = K_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convenience wrapper returning one level from the coordinated outputs."""
    if geography not in {"province", "district"}:
        raise ValueError("geography must be 'province' or 'district'.")
    province_safe, district_safe, province_summary, district_summary = (
        aggregate_response_levels(records, k=k)
    )
    if geography == "province":
        return province_safe, province_summary
    return district_safe, district_summary


def chart_data(safe: pd.DataFrame) -> pd.DataFrame:
    """Return only unsuppressed aggregate cells for charts and tooltips."""
    if "suppressed" not in safe.columns:
        raise ValueError("safe data must include a suppressed column.")
    return safe.loc[~safe["suppressed"]].drop(columns="suppressed").reset_index(drop=True)


def table_data(
    safe: pd.DataFrame,
    placeholder: str | None = None,
) -> pd.DataFrame:
    """Render hidden cells with a label; never restore their underlying values."""
    if "suppressed" not in safe.columns:
        raise ValueError("safe data must include a suppressed column.")
    if placeholder is None:
        placeholder = f"Suppressed (n<{K_THRESHOLD})"
    out = safe.copy()
    for column in ("response_count", "percentage_share"):
        out[column] = out[column].astype("object")
    out.loc[out["suppressed"], ["response_count", "percentage_share"]] = placeholder
    return out.drop(columns="suppressed")


def export_aggregate_report(safe: pd.DataFrame, path: str) -> None:
    """Write privacy-filtered aggregates; suppressed numeric values remain blank."""
    if "suppressed" not in safe.columns:
        raise ValueError("safe data must include a suppressed column.")
    safe.to_csv(path, index=False)


def test_small_cells_are_suppressed() -> None:
    """Exercise primary, complementary, and output-channel suppression."""
    sample = pd.DataFrame(
        [
            {
                "province": "Western",
                "week": 1,
                "theme": "Job and Money",
                "response_count": 20,
                "percentage_share": 66.7,
            },
            {
                "province": "Western",
                "week": 1,
                "theme": "Other",
                "response_count": 9,
                "percentage_share": 30.0,
            },
        ]
    )
    safe, summary = apply_privacy_filter(sample)
    assert safe["suppressed"].all()
    assert safe["response_count"].isna().all()
    assert safe["percentage_share"].isna().all()
    assert summary["suppressed_cells"] == 2
    assert chart_data(safe).empty
    assert table_data(safe)["response_count"].eq(
        f"Suppressed (n<{K_THRESHOLD})"
    ).all()


if __name__ == "__main__":
    test_small_cells_are_suppressed()
    print(f"PASS: small-cell suppression uses k={K_THRESHOLD}.")
