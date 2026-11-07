"""
database.py - Supabase connection and data retrieval module
"""

from __future__ import annotations

import datetime
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import create_client

from geography import DISTRICT_TO_PROVINCE, DISTRICTS, PROVINCES

# Load environment variables from .env file in the project root
load_dotenv(Path(__file__).resolve().parent / ".env")

SEED = 42
N_WEEKS = 12
MINIMUM_PER_DISTRICT_WEEK = 95
START_DATE = datetime.date(2026, 1, 5)

THEMES = (
    "Job and Money",
    "Exams and Studies",
    "Loneliness and Isolation",
    "Family and Relationships",
    "Other",
)
LANGUAGES = ("Sinhala", "Tamil", "English")

BASE_THEME_WEIGHTS = (0.25, 0.18, 0.21, 0.21, 0.15)
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
SPARSE_DEMO_PAIRS = {("Kilinochchi", 1), ("Mullaitivu", 1)}


def get_admin_client():
    """Client for trusted server-side database operations using the secret service role key."""
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not secret_key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_SECRET_KEY in .env file."
        )

    return create_client(url, secret_key)


def get_read_client():
    """Client for reading data using the Supabase publishable/anon key."""
    url = os.getenv("SUPABASE_URL")
    publishable_key = os.getenv("SUPABASE_PUBLISHABLE_KEY")

    if not url or not publishable_key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY in .env file."
        )

    return create_client(url, publishable_key)


def _theme_probs(week: int, district: str) -> list[float]:
    province = DISTRICT_TO_PROVINCE[district]
    d_idx = DISTRICTS.index(district)
    mult = PROVINCE_THEME_MULTIPLIERS[province]
    jitter = [
        1 + 0.018 * math.sin(week * 1.3 + d_idx * 0.5 + idx * 0.9)
        for idx in range(len(THEMES))
    ]
    shift = [
        1 + 0.08 * math.sin(d_idx + 0.7),
        1 + 0.07 * math.cos(d_idx + 1.3),
        1 + 0.06 * math.sin(d_idx + 2.1),
        1 + 0.06 * math.cos(d_idx + 0.4),
        1 + 0.05 * math.sin(d_idx + 2.8),
    ]
    weights = [
        b * m * s * j
        for b, m, s, j in zip(BASE_THEME_WEIGHTS, mult, shift, jitter)
    ]
    if week in (7, 8, 9, 10):
        weights[0] *= {7: 1.22, 8: 1.42, 9: 1.42, 10: 1.22}[week]
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
    total = sum(weights)
    return [w / total for w in weights]


def _lang_probs(district: str) -> list[float]:
    province = DISTRICT_TO_PROVINCE[district]
    d_idx = DISTRICTS.index(district)
    base = PROVINCE_LANGUAGE_WEIGHTS[province]
    shift = [
        1 + 0.04 * math.sin(d_idx + 0.3),
        1 + 0.05 * math.cos(d_idx + 1.1),
        1 + 0.04 * math.sin(d_idx + 2.4),
    ]
    weights = [b * s for b, s in zip(base, shift)]
    total = sum(weights)
    return [w / total for w in weights]


def _apportion(total_n: int, probs: list[float], min_per_item: int = 0) -> list[int]:
    raw = [float(total_n * p) for p in probs]
    counts = [max(min_per_item, int(math.floor(v))) for v in raw]
    diff = total_n - sum(counts)
    if diff > 0:
        order = sorted(
            range(len(probs)),
            key=lambda i: (raw[i] - counts[i], probs[i]),
            reverse=True,
        )
        for idx in order[:diff]:
            counts[idx] += 1
    elif diff < 0:
        while diff < 0:
            eligible = [i for i in range(len(probs)) if counts[i] > min_per_item]
            if not eligible:
                break
            idx = max(eligible, key=lambda i: counts[i])
            counts[idx] -= 1
            diff += 1
    return counts


def _generate_district_week_records(district: str, week: int, count: int) -> list[dict[str, Any]]:
    province = DISTRICT_TO_PROVINCE[district]
    t_probs = _theme_probs(week, district)
    l_probs = _lang_probs(district)
    rng = random.Random(SEED + DISTRICTS.index(district) * 100 + week)

    is_sparse = (district, week) in SPARSE_DEMO_PAIRS
    min_floor = 0 if is_sparse else max(11, count // 8)
    t_counts = _apportion(count, t_probs, min_per_item=min_floor)
    if is_sparse and len(t_counts) == 5:
        excess = t_counts[4] - 7
        t_counts[4] = 7
        t_counts[0] += excess

    themes_list: list[str] = []
    for idx, c in enumerate(t_counts):
        themes_list.extend([THEMES[idx]] * c)
    rng.shuffle(themes_list)

    l_counts = _apportion(count, l_probs, min_per_item=1)
    langs_list: list[str] = []
    for idx, c in enumerate(l_counts):
        langs_list.extend([LANGUAGES[idx]] * c)
    rng.shuffle(langs_list)

    rows: list[dict[str, Any]] = []
    for i in range(count):
        day_offset = i % 7
        r_date = (START_DATE + datetime.timedelta(days=(week - 1) * 7 + day_offset)).isoformat()
        rows.append(
            {
                "province": province,
                "district": district,
                "week_number": week,
                "response_date": r_date,
                "theme": themes_list[i],
                "language": langs_list[i],
                "is_synthetic": True,
            }
        )
    return rows


def _sanitize_and_backfill(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize existing Supabase rows and backfill any district-week below minimum coverage."""
    clean_records: list[dict[str, Any]] = []
    pair_counts: Counter[tuple[str, int]] = Counter()

    for row in records:
        district = str(row.get("district") or "").strip()
        if district not in DISTRICT_TO_PROVINCE:
            continue
        province = DISTRICT_TO_PROVINCE[district]
        week_raw = row.get("week_number", row.get("week"))
        try:
            week_num = int(float(week_raw))
        except (TypeError, ValueError):
            continue
        if week_num not in range(1, N_WEEKS + 1):
            continue
        theme = str(row.get("theme") or "").strip()
        if theme not in THEMES:
            continue
        language = str(row.get("language") or "Sinhala").strip()
        if language not in LANGUAGES:
            language = "Sinhala"

        sanitized = dict(row)
        sanitized["province"] = province
        sanitized["district"] = district
        sanitized["week_number"] = week_num
        sanitized["theme"] = theme
        sanitized["language"] = language
        sanitized["is_synthetic"] = True
        clean_records.append(sanitized)
        pair_counts[(district, week_num)] += 1

    for idx, district in enumerate(DISTRICTS):
        target_for_district = MINIMUM_PER_DISTRICT_WEEK + (idx % 5) * 3
        for week in range(1, N_WEEKS + 1):
            existing_n = pair_counts[(district, week)]
            if existing_n < MINIMUM_PER_DISTRICT_WEEK:
                needed = target_for_district - existing_n
                clean_records.extend(
                    _generate_district_week_records(district, week, needed)
                )

    return clean_records


def fetch_responses() -> list[dict]:
    """
    Retrieve all synthetic responses from Supabase table 'wellbeing_responses'
    and ensure full 25-district coverage across all 9 provinces and 12 weeks.
    """
    all_records: list[dict] = []
    page_size = 1000
    start = 0

    try:
        client = get_read_client()
        while True:
            response = (
                client.table("wellbeing_responses")
                .select(
                    "id, province, district, week_number, "
                    "theme, language, is_synthetic, created_at, response_date"
                )
                .eq("is_synthetic", True)
                .range(start, start + page_size - 1)
                .execute()
            )

            records = response.data or []
            all_records.extend(records)

            if len(records) < page_size:
                break

            start += page_size

    except Exception:
        return _sanitize_and_backfill([])

    return _sanitize_and_backfill(all_records)

    
