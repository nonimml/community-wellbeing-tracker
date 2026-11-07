"""
database.py - Supabase connection and data retrieval module
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

from geography import DISTRICTS

# Load environment variables from .env file in the project root
load_dotenv(Path(__file__).resolve().parent / ".env")
LOCAL_CSV_PATH = Path(__file__).resolve().parent / "data" / "synthetic_responses.csv"


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


def _load_local_csv_records() -> list[dict]:
    """Load synthetic fallback/supplement records from data/synthetic_responses.csv if present."""
    if not LOCAL_CSV_PATH.exists():
        return []
    records: list[dict] = []
    with LOCAL_CSV_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            district = (row.get("district") or "").strip()
            province = (row.get("province") or "").strip()
            week_str = (row.get("week_number") or row.get("week") or "").strip()
            if not district or not province or not week_str:
                continue
            try:
                week_number = int(float(week_str))
            except ValueError:
                continue
            records.append(
                {
                    "province": province,
                    "district": district,
                    "week_number": week_number,
                    "response_date": (row.get("response_date") or "").strip()[:10],
                    "theme": (row.get("theme") or "").strip(),
                    "language": (row.get("language") or "").strip(),
                    "is_synthetic": True,
                }
            )
    return records


def _ensure_full_district_coverage(records: list[dict]) -> list[dict]:
    """Ensure all 25 Sri Lankan districts have synthetic demonstration coverage.

    If Supabase currently holds only the initial Western Province records
    (Colombo, Gampaha, Kalutara) before the admin expansion script has been
    executed against Supabase, this deterministically appends the planned
    synthetic records for the remaining districts so the dashboard renders
    full 25-district coverage immediately.
    """
    covered_districts = {
        str(r.get("district", "")).strip()
        for r in records
        if r.get("district")
    }
    if len(covered_districts & set(DISTRICTS)) >= len(DISTRICTS) and len(records) >= 25_000:
        return records

    try:
        from generate_data import plan_additional_records

        additional, _ = plan_additional_records(records)
        return records + additional
    except Exception:
        csv_records = _load_local_csv_records()
        if csv_records:
            missing_rows = [
                r for r in csv_records if r.get("district") not in covered_districts
            ]
            return records + missing_rows
        return records


def fetch_responses() -> list[dict]:
    """
    Retrieve all synthetic responses from Supabase table 'wellbeing_responses'.
    Uses pagination to bypass row limits. Filters strictly for is_synthetic = TRUE.
    Automatically supplements missing district coverage using the deterministic
    synthetic planner if Supabase has not yet been expanded to all 25 districts.
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

    except Exception as e:
        fallback = _ensure_full_district_coverage([])
        if fallback:
            return fallback
        raise ConnectionError(
            f"Failed to connect to Supabase or fetch records: {e}"
        ) from e

    return _ensure_full_district_coverage(all_records)
