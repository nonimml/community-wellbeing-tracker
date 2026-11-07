"""Generate clearly synthetic demonstration rows for every Sri Lankan district.

The target is the desired total number of synthetic rows in Supabase, not the
number to insert on each run. Existing records are preserved and counted
toward that target.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from database import get_admin_client
from geography import DISTRICT_TO_PROVINCE, DISTRICTS, PROVINCES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
TARGET_TOTAL_RECORDS = 30_000
N_WEEKS = 12
MINIMUM_PER_DISTRICT_WEEK = 95
CANDIDATE_POOL_PER_DISTRICT_WEEK = 260
BATCH_SIZE = 500
START_DATE = np.datetime64("2026-01-05")
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "data" / "synthetic_responses.csv"

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
BASE_THEME_WEIGHTS = np.array([0.25, 0.18, 0.21, 0.21, 0.15], dtype=float)
PROVINCE_THEME_MULTIPLIERS = {
    "Western": (1.08, 1.08, 0.96, 0.98, 0.96),
    "Central": (0.97, 1.12, 1.04, 1.00, 0.95),
    "Southern": (1.04, 0.97, 1.02, 1.08, 0.96),
    "Northern": (0.96, 1.10, 1.10, 1.00, 0.94),
    "Eastern": (1.02, 1.00, 1.08, 1.04, 0.95),
    "North Western": (1.06, 1.03, 0.97, 1.02, 0.95),
    "North Central": (1.02, 1.10, 0.98, 1.00, 0.95),
    "Uva": (1.00, 1.06, 1.05, 1.02, 0.95),
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

# Intentional small-cell pairs in Week 1 only so the Privacy & Data Quality
# page demonstrates active k < 10 and complementary suppression while all 25
# districts remain visible across the 12-week window.
SPARSE_DEMO_PAIRS = {("Kilinochchi", 1), ("Mullaitivu", 1)}


def _normalized(weights: np.ndarray) -> np.ndarray:
    return weights / weights.sum()


def get_theme_probabilities(week: int, district: str) -> np.ndarray:
    """Return illustrative theme weights with province, district, and wave variation."""
    province = DISTRICT_TO_PROVINCE[district]
    district_index = DISTRICTS.index(district)
    multipliers = np.asarray(PROVINCE_THEME_MULTIPLIERS[province], dtype=float)
    weekly_jitter = np.asarray(
        [
            1 + 0.018 * np.sin(week * 1.3 + district_index * 0.5 + index * 0.9)
            for index in range(len(THEMES))
        ],
        dtype=float,
    )
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
    weights = BASE_THEME_WEIGHTS * multipliers * district_shift * weekly_jitter

    # Mid-period scenario: financial stressor surge in Weeks 7-10.
    if week in (7, 8, 9, 10):
        weights[0] *= {7: 1.22, 8: 1.42, 9: 1.42, 10: 1.22}[week]

    # Late-period scenario: regional stressor surges in Weeks 11-12 so
    # default Week 12 displays both ALERT and WATCH signals.
    if week in (11, 12):
        if province in ("Central", "Northern"):
            weights[1] *= {11: 1.28, 12: 1.85}[week]
        elif province == "Sabaragamuwa":
            weights[1] *= {11: 1.12, 12: 1.32}[week]
        elif province in ("Southern", "North Western"):
            weights[0] *= {11: 1.10, 12: 1.28}[week]
        elif province == "Western":
            weights[0] *= {11: 1.15, 12: 1.42}[week]
        elif province in ("Eastern", "Uva"):
            weights[2] *= {11: 1.15, 12: 1.56}[week]

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
    """Identity based on non-generated database fields."""
    return (
        str(record.get("province", "")).strip(),
        str(record.get("district", "")).strip(),
        str(record.get("week_number", record.get("week", ""))).strip(),
        _canonical_date(record.get("response_date")),
        str(record.get("theme", "")).strip(),
        str(record.get("language", "")).strip(),
        str(_is_synthetic(record.get("is_synthetic", True))),
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


def _apportion_counts(
    total_n: int,
    probabilities: np.ndarray,
    min_per_item: int = 0,
) -> list[int]:
    """Apportion total_n integers proportionally to probabilities with a minimum floor."""
    raw = [float(total_n * prob) for prob in probabilities]
    counts = [max(min_per_item, int(np.floor(val))) for val in raw]
    diff = total_n - sum(counts)
    if diff > 0:
        remainders = sorted(
            range(len(probabilities)),
            key=lambda idx: (raw[idx] - counts[idx], float(probabilities[idx])),
            reverse=True,
        )
        for idx in remainders[:diff]:
            counts[idx] += 1
    elif diff < 0:
        while diff < 0:
            eligible = [idx for idx in range(len(probabilities)) if counts[idx] > min_per_item]
            if not eligible:
                break
            largest = max(eligible, key=lambda idx: counts[idx])
            counts[largest] -= 1
            diff += 1
    return counts


def _candidate_records(
    district: str,
    week: int,
    pool_size: int = CANDIDATE_POOL_PER_DISTRICT_WEEK,
) -> list[dict[str, Any]]:
    """Build a stratified stream of synthetic candidate records for one district-week."""
    province = DISTRICT_TO_PROVINCE[district]
    theme_probabilities = get_theme_probabilities(week, district)
    language_probabilities = get_language_probabilities(district)
    rng = np.random.default_rng(SEED + DISTRICTS.index(district) * 100 + week)

    is_sparse_demo = (district, week) in SPARSE_DEMO_PAIRS
    block_size = 25
    n_blocks = (pool_size + block_size - 1) // block_size
    candidates: list[dict[str, Any]] = []
    other_emitted = 0

    for block_idx in range(n_blocks):
        min_theme_floor = 0 if is_sparse_demo else 3
        theme_counts = _apportion_counts(
            block_size, theme_probabilities, min_per_item=min_theme_floor
        )
        block_themes: list[str] = []
        for theme_idx, count in enumerate(theme_counts):
            for _ in range(count):
                if is_sparse_demo and theme_idx == 4 and other_emitted >= 7:
                    block_themes.append(THEMES[0])
                else:
                    if theme_idx == 4:
                        other_emitted += 1
                    block_themes.append(THEMES[theme_idx])
        rng.shuffle(block_themes)

        lang_counts = _apportion_counts(
            block_size, language_probabilities, min_per_item=1
        )
        block_languages: list[str] = []
        for lang_idx, count in enumerate(lang_counts):
            block_languages.extend([LANGUAGES[lang_idx]] * count)
        rng.shuffle(block_languages)

        for offset in range(block_size):
            day_offset = (block_idx * block_size + offset) % 7
            response_date = str(
                START_DATE + np.timedelta64((week - 1) * 7 + day_offset, "D")
            )
            candidates.append(
                {
                    "province": province,
                    "district": district,
                    "week_number": week,
                    "response_date": response_date,
                    "theme": block_themes[offset],
                    "language": block_languages[offset],
                    "is_synthetic": True,
                }
            )

    return candidates[:pool_size]


def plan_additional_records(
    existing_records: list[dict[str, Any]],
    target_total: int = TARGET_TOTAL_RECORDS,
    minimum_per_district_week: int = MINIMUM_PER_DISTRICT_WEEK,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Plan additions using multiset deduplication without altering existing rows."""
    if target_total < 0:
        raise ValueError("target_total cannot be negative.")
    if minimum_per_district_week < 1:
        raise ValueError("minimum_per_district_week must be positive.")

    existing_synthetic = [
        record
        for record in existing_records
        if _is_synthetic(record.get("is_synthetic", True))
    ]
    existing_key_counts = Counter(_record_key(record) for record in existing_synthetic)

    counts_by_pair: Counter[tuple[str, int]] = Counter()
    counts_by_district: Counter[str] = Counter()
    for record in existing_synthetic:
        province = record.get("province")
        district = record.get("district")
        week = record.get("week_number", record.get("week"))
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

    candidate_pools: dict[tuple[str, int], list[dict[str, Any]]] = {}
    available_by_pair: dict[tuple[str, int], int] = {}
    for district in DISTRICTS:
        for week in range(1, N_WEEKS + 1):
            pair = (district, week)
            pool = _candidate_records(district, week)
            temp_counts = existing_key_counts.copy()
            filtered_pool: list[dict[str, Any]] = []
            for candidate in pool:
                key = _record_key(candidate)
                if temp_counts[key] > 0:
                    temp_counts[key] -= 1
                else:
                    filtered_pool.append(candidate)
            candidate_pools[pair] = pool
            available_by_pair[pair] = len(filtered_pool)

    quotas: Counter[tuple[str, int]] = Counter()
    for district in DISTRICTS:
        for week in range(1, N_WEEKS + 1):
            pair = (district, week)
            deficit = max(0, minimum_per_district_week - counts_by_pair[pair])
            if deficit > available_by_pair[pair]:
                raise RuntimeError(
                    f"Insufficient candidate records for {district}, week {week}; "
                    "reduce the requested minimum or increase pool size."
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
                week
                for week in range(1, N_WEEKS + 1)
                if quotas[(district, week)] < available_by_pair[(district, week)]
            ]
            if available_weeks:
                eligible_districts.append(district)
                projected_totals[district] = counts_by_district[district] + sum(
                    quotas[(district, week)] for week in range(1, N_WEEKS + 1)
                )

        if not eligible_districts:
            raise RuntimeError(
                "The requested total exceeds the available synthetic candidate pool."
            )

        minimum_total = min(projected_totals.values())
        least_covered = [
            district
            for district in eligible_districts
            if projected_totals[district] == minimum_total
        ]
        weights = np.asarray(
            [district_weights[DISTRICTS.index(district)] for district in least_covered],
            dtype=float,
        )
        district = str(allocation_rng.choice(least_covered, p=_normalized(weights)))
        eligible_weeks = [
            week
            for week in range(1, N_WEEKS + 1)
            if quotas[(district, week)] < available_by_pair[(district, week)]
        ]
        week_counts = [
            counts_by_pair[(district, week)] + quotas[(district, week)]
            for week in eligible_weeks
        ]
        minimum_week_count = min(week_counts)
        least_covered_weeks = [
            week
            for week, count in zip(eligible_weeks, week_counts)
            if count == minimum_week_count
        ]
        week = int(allocation_rng.choice(least_covered_weeks))
        quotas[(district, week)] += 1
        remaining -= 1

    records: list[dict[str, Any]] = []
    skipped_duplicates = 0
    remaining_existing = existing_key_counts.copy()
    for district in DISTRICTS:
        for week in range(1, N_WEEKS + 1):
            needed = quotas[(district, week)]
            if needed == 0:
                continue
            added_for_pair = 0
            for candidate in candidate_pools[(district, week)]:
                key = _record_key(candidate)
                if remaining_existing[key] > 0:
                    remaining_existing[key] -= 1
                    skipped_duplicates += 1
                    continue
                records.append(candidate)
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


def export_csv(
    records: list[dict[str, Any]],
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> Path:
    """Write synthetic records to CSV compatible with both analysis.py and privacy.py."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "response_id",
        "province",
        "district",
        "week",
        "week_number",
        "response_date",
        "theme",
        "language",
        "data_type",
        "is_synthetic",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            week_num = int(record.get("week_number", record.get("week", 1)))
            writer.writerow(
                {
                    "response_id": f"SYN-{index:05d}",
                    "province": record["province"],
                    "district": record["district"],
                    "week": week_num,
                    "week_number": week_num,
                    "response_date": str(record["response_date"])[:10],
                    "theme": record["theme"],
                    "language": record["language"],
                    "data_type": "synthetic",
                    "is_synthetic": True,
                }
            )
    return path


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
    parser = argparse.ArgumentParser(
        description="Expand Sri Lanka synthetic wellbeing dataset across all 25 districts."
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Generate data/synthetic_responses.csv locally without connecting to Supabase.",
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help="Output CSV path (default: data/synthetic_responses.csv).",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("SRI LANKA SYNTHETIC DEMONSTRATION DATA EXPANSION")
    print("Illustrative data only; not a representation of actual conditions.")
    print("=" * 68)

    full_records, _ = plan_additional_records([])
    csv_out = export_csv(full_records, args.csv_path)
    print(f"\nExported {len(full_records):,} synthetic records to {csv_out}")

    if args.csv_only:
        return

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
            batch = records[start : start + BATCH_SIZE]
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
