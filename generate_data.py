"""Generate clearly synthetic demonstration rows for every Sri Lankan district.

The target is the desired total number of synthetic rows in Supabase, not the
number to insert on each run. Existing records are preserved and counted
toward that target.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from database import get_admin_client
from geography import DISTRICT_TO_PROVINCE, DISTRICTS, PROVINCES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
TARGET_TOTAL_RECORDS = 25_000
N_WEEKS = 12
MINIMUM_PER_DISTRICT_WEEK = 50
BATCH_SIZE = 500
START_DATE = np.datetime64("2026-01-05")

THEMES = (
    "Job and Money",
    "Exams and Studies",
    "Loneliness and Isolation",
    "Family and Relationships",
    "Other",
)
LANGUAGES = ("Sinhala", "Tamil", "English")

# Illustrative generation weights only; these are not estimates of real
# community conditions or language use.
BASE_THEME_WEIGHTS = np.array([0.26, 0.18, 0.22, 0.22, 0.12], dtype=float)
PROVINCE_THEME_MULTIPLIERS = {
    "Western": (1.08, 1.08, 0.96, 0.98, 0.95),
    "Central": (0.97, 1.12, 1.04, 1.00, 0.93),
    "Southern": (1.04, 0.97, 1.02, 1.08, 0.95),
    "Northern": (0.96, 1.10, 1.10, 1.00, 0.91),
    "Eastern": (1.02, 1.00, 1.08, 1.04, 0.94),
    "North Western": (1.06, 1.03, 0.97, 1.02, 0.94),
    "North Central": (1.02, 1.10, 0.98, 1.00, 0.94),
    "Uva": (1.00, 1.06, 1.05, 1.02, 0.94),
    "Sabaragamuwa": (1.04, 1.00, 1.02, 1.04, 0.96),
}
PROVINCE_LANGUAGE_WEIGHTS = {
    "Western": (0.54, 0.15, 0.31),
    "Central": (0.61, 0.12, 0.27),
    "Southern": (0.63, 0.10, 0.27),
    "Northern": (0.20, 0.66, 0.14),
    "Eastern": (0.42, 0.39, 0.19),
    "North Western": (0.64, 0.10, 0.26),
    "North Central": (0.63, 0.09, 0.28),
    "Uva": (0.61, 0.11, 0.28),
    "Sabaragamuwa": (0.62, 0.10, 0.28),
}


def _normalized(weights: np.ndarray) -> np.ndarray:
    return weights / weights.sum()


def get_theme_probabilities(week: int, district: str) -> np.ndarray:
    """Return illustrative theme weights with province/district variation."""
    province = DISTRICT_TO_PROVINCE[district]
    district_index = DISTRICTS.index(district)
    multipliers = np.asarray(PROVINCE_THEME_MULTIPLIERS[province], dtype=float)
    district_shift = np.asarray(
        [
            1 + 0.08 * np.sin(district_index + 0.7),
            1 + 0.07 * np.cos(district_index + 1.3),
            1 + 0.06 * np.sin(district_index + 2.1),
            1 + 0.06 * np.cos(district_index + 0.4),
            1 + 0.05 * np.sin(district_index + 2.8),
        ],
        dtype=float,
    )
    weights = BASE_THEME_WEIGHTS * multipliers * district_shift

    # A fictional, time-varying scenario keeps weeks from being identical.
    if week in (7, 8, 9, 10):
        weights[0] *= {7: 1.08, 8: 1.16, 9: 1.16, 10: 1.08}[week]
    return _normalized(weights)


def get_language_probabilities(district: str) -> np.ndarray:
    """Return illustrative language weights with small district variation."""
    province = DISTRICT_TO_PROVINCE[district]
    district_index = DISTRICTS.index(district)
    weights = np.asarray(PROVINCE_LANGUAGE_WEIGHTS[province], dtype=float)
    weights *= np.asarray(
        [
            1 + 0.04 * np.sin(district_index + 0.3),
            1 + 0.05 * np.cos(district_index + 1.1),
            1 + 0.04 * np.sin(district_index + 2.4),
        ],
        dtype=float,
    )
    return _normalized(weights)


def _is_synthetic(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _canonical_date(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()[:10]


def _record_key(record: dict[str, Any]) -> tuple[str, ...]:
    """Identity based on every non-generated database field."""
    return (
        str(record.get("province", "")).strip(),
        str(record.get("district", "")).strip(),
        str(record.get("week_number", "")).strip(),
        _canonical_date(record.get("response_date")),
        str(record.get("theme", "")).strip(),
        str(record.get("language", "")).strip(),
        str(_is_synthetic(record.get("is_synthetic", False))),
    )


def validate_records(records: list[dict[str, Any]]) -> None:
    """Fail before upload if generated rows do not follow the data contract."""
    for index, record in enumerate(records):
        province = record.get("province")
        district = record.get("district")
        if province not in PROVINCES or DISTRICT_TO_PROVINCE.get(district) != province:
            raise ValueError(f"Record {index} has an invalid province/district pair.")
        if record.get("week_number") not in range(1, N_WEEKS + 1):
            raise ValueError(f"Record {index} has an invalid week_number.")
        if record.get("theme") not in THEMES:
            raise ValueError(f"Record {index} has an unsupported theme.")
        if record.get("language") not in LANGUAGES:
            raise ValueError(f"Record {index} has an unsupported language.")
        if record.get("is_synthetic") is not True:
            raise ValueError(f"Record {index} is not marked as synthetic.")
        try:
            date = np.datetime64(record.get("response_date"), "D")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Record {index} has an invalid response_date.") from exc
        if np.isnat(date):
            raise ValueError(f"Record {index} has an invalid response_date.")


def _candidate_records(district: str, week: int) -> list[dict[str, Any]]:
    """Build every distinct date/theme/language combination for one week."""
    province = DISTRICT_TO_PROVINCE[district]
    candidates = []
    candidate_weights = []
    theme_probabilities = get_theme_probabilities(week, district)
    language_probabilities = get_language_probabilities(district)

    for day_offset in range(7):
        response_date = str(
            START_DATE + np.timedelta64((week - 1) * 7 + day_offset, "D")
        )
        for theme_index, theme in enumerate(THEMES):
            for language_index, language in enumerate(LANGUAGES):
                candidates.append(
                    {
                        "province": province,
                        "district": district,
                        "week_number": week,
                        "response_date": response_date,
                        "theme": theme,
                        "language": language,
                        "is_synthetic": True,
                    }
                )
                candidate_weights.append(
                    theme_probabilities[theme_index]
                    * language_probabilities[language_index]
                    / 7
                )

    order_rng = np.random.default_rng(
        SEED + DISTRICTS.index(district) * N_WEEKS + week
    )
    order = order_rng.choice(
        len(candidates),
        size=len(candidates),
        replace=False,
        p=_normalized(np.asarray(candidate_weights, dtype=float)),
    )
    return [candidates[int(index)] for index in order]


def plan_additional_records(
    existing_records: list[dict[str, Any]],
    target_total: int = TARGET_TOTAL_RECORDS,
    minimum_per_district_week: int = MINIMUM_PER_DISTRICT_WEEK,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Plan unique additions without changing or re-inserting existing rows."""
    if target_total < 0:
        raise ValueError("target_total cannot be negative.")
    if minimum_per_district_week < 1:
        raise ValueError("minimum_per_district_week must be positive.")

    existing_keys = {_record_key(record) for record in existing_records}
    existing_synthetic = [
        record for record in existing_records
        if _is_synthetic(record.get("is_synthetic", False))
    ]

    counts_by_pair: Counter[tuple[str, int]] = Counter()
    counts_by_district: Counter[str] = Counter()
    for record in existing_synthetic:
        province = record.get("province")
        district = record.get("district")
        week = record.get("week_number")
        if province is None or district is None or week is None:
            continue
        if DISTRICT_TO_PROVINCE.get(district) != province:
            raise ValueError(
                f"Existing synthetic record has invalid geography: "
                f"{province!r}/{district!r}. Apply the geography migration "
                "and correct invalid rows before generating more data."
            )
        try:
            week_number = int(week)
        except (TypeError, ValueError):
            continue
        if week_number not in range(1, N_WEEKS + 1):
            continue
        counts_by_pair[(district, week_number)] += 1
        counts_by_district[district] += 1

    candidate_pools = {
        (district, week): _candidate_records(district, week)
        for district in DISTRICTS
        for week in range(1, N_WEEKS + 1)
    }
    available_by_pair = {
        pair: sum(_record_key(candidate) not in existing_keys for candidate in pool)
        for pair, pool in candidate_pools.items()
    }

    quotas: Counter[tuple[str, int]] = Counter()
    for district in DISTRICTS:
        for week in range(1, N_WEEKS + 1):
            pair = (district, week)
            deficit = max(
                0, minimum_per_district_week - counts_by_pair[pair]
            )
            if deficit > available_by_pair[pair]:
                raise RuntimeError(
                    f"Insufficient unique date/theme/language combinations for "
                    f"{district}, week {week}; reduce the requested minimum."
                )
            quotas[pair] = deficit

    required_for_coverage = sum(quotas.values())
    additional_needed = max(
        max(0, target_total - len(existing_synthetic)),
        required_for_coverage,
    )
    remaining = additional_needed - required_for_coverage

    allocation_rng = np.random.default_rng(SEED)
    district_weights = np.asarray(
        [0.85 + (index % 7) * 0.05 for index in range(len(DISTRICTS))],
        dtype=float,
    )

    while remaining:
        eligible_districts = []
        projected_totals = {}
        for district in DISTRICTS:
            available_weeks = [
                week for week in range(1, N_WEEKS + 1)
                if quotas[(district, week)] < available_by_pair[(district, week)]
            ]
            if available_weeks:
                eligible_districts.append(district)
                projected_totals[district] = (
                    counts_by_district[district]
                    + sum(quotas[(district, week)] for week in range(1, N_WEEKS + 1))
                )

        if not eligible_districts:
            raise RuntimeError(
                "The requested total exceeds the available unique synthetic "
                "record combinations."
            )

        minimum_total = min(projected_totals.values())
        least_covered = [
            district for district in eligible_districts
            if projected_totals[district] == minimum_total
        ]
        weights = np.asarray(
            [district_weights[DISTRICTS.index(district)] for district in least_covered],
            dtype=float,
        )
        district = str(
            allocation_rng.choice(least_covered, p=_normalized(weights))
        )
        eligible_weeks = [
            week for week in range(1, N_WEEKS + 1)
            if quotas[(district, week)] < available_by_pair[(district, week)]
        ]
        week_counts = [
            counts_by_pair[(district, week)] + quotas[(district, week)]
            for week in eligible_weeks
        ]
        minimum_week_count = min(week_counts)
        least_covered_weeks = [
            week for week, count in zip(eligible_weeks, week_counts)
            if count == minimum_week_count
        ]
        week = int(allocation_rng.choice(least_covered_weeks))
        quotas[(district, week)] += 1
        remaining -= 1

    records: list[dict[str, Any]] = []
    skipped_duplicates = 0
    for district in DISTRICTS:
        for week in range(1, N_WEEKS + 1):
            needed = quotas[(district, week)]
            if needed == 0:
                continue
            added_for_pair = 0
            for candidate in candidate_pools[(district, week)]:
                key = _record_key(candidate)
                if key in existing_keys:
                    skipped_duplicates += 1
                    continue
                records.append(candidate)
                existing_keys.add(key)
                added_for_pair += 1
                if added_for_pair == needed:
                    break

    if len(records) != additional_needed:
        raise RuntimeError(
            f"Planned {additional_needed} additions but generated {len(records)}."
        )
    validate_records(records)
    return records, {
        "already_present": len(existing_records),
        "existing_synthetic": len(existing_synthetic),
        "generated_candidates": len(records) + skipped_duplicates,
        "skipped_duplicates": skipped_duplicates,
        "to_upload": len(records),
    }


def fetch_existing_records(client: Any) -> list[dict[str, Any]]:
    """Read the whole table in bounded pages before planning any upload."""
    columns = (
        "id,province,district,week_number,response_date,"
        "theme,language,is_synthetic"
    )
    page_size = 1000
    start = 0
    records = []
    while True:
        response = (
            client.table("wellbeing_responses")
            .select(columns)
            .range(start, start + page_size - 1)
            .execute()
        )
        page = response.data or []
        records.extend(page)
        if len(page) < page_size:
            return records
        start += page_size


def main() -> None:
    print("=" * 68)
    print("SRI LANKA SYNTHETIC DEMONSTRATION DATA EXPANSION")
    print("Illustrative data only; not a representation of actual conditions.")
    print("=" * 68)

    client = get_admin_client()
    print("\nChecking existing Supabase rows before planning additions...")
    existing_records = fetch_existing_records(client)
    records, summary = plan_additional_records(existing_records)

    print(f"Existing rows: {summary['already_present']:,}")
    print(f"Existing synthetic rows: {summary['existing_synthetic']:,}")
    print(f"Target total synthetic rows: {TARGET_TOTAL_RECORDS:,}")
    print(f"Candidate rows generated: {summary['generated_candidates']:,}")
    print(f"Existing duplicate candidates skipped: {summary['skipped_duplicates']:,}")
    print(f"Rows to upload: {summary['to_upload']:,}")
    print(f"Geographic coverage: {len(PROVINCES)} provinces, {len(DISTRICTS)} districts")
    print(f"Weeks covered: 1-{N_WEEKS}")

    if not records:
        print("\nNo upload needed; the target and minimum district/week coverage are met.")
        print(f"Uploaded: 0 | Existing records preserved: {summary['already_present']:,}")
        return

    uploaded = 0
    try:
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start:start + BATCH_SIZE]
            response = (
                client.table("wellbeing_responses")
                .insert(batch)
                .select("id")
                .execute()
            )
            uploaded += len(response.data or [])
            print(f"Uploaded {uploaded:,}/{len(records):,} new rows")
    except Exception as exc:
        raise RuntimeError(
            f"Upload stopped after {uploaded:,} confirmed rows. Existing data was "
            "not deleted; rerunning safely checks the database before continuing."
        ) from exc

    print("\n" + "=" * 68)
    print("UPLOAD SUMMARY")
    print("=" * 68)
    print(f"Already present before run: {summary['already_present']:,}")
    print(f"Generated candidates: {summary['generated_candidates']:,}")
    print(f"Skipped as duplicates: {summary['skipped_duplicates']:,}")
    print(f"Uploaded: {uploaded:,}")
    print("Existing rows were not updated or deleted.")


if __name__ == "__main__":
    main()
