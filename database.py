"""
database.py - Supabase connection and data retrieval module
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables from .env file in the project root
load_dotenv(Path(__file__).resolve().parent / ".env")


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


def fetch_responses() -> list[dict]:
    """
    Retrieve all synthetic responses from Supabase table 'wellbeing_responses'.
    Uses pagination to bypass row limits. Filters strictly for is_synthetic = TRUE.
    """
    client = get_read_client()

    all_records = []
    page_size = 1000
    start = 0

    try:
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
        raise ConnectionError(
            f"Failed to connect to Supabase or fetch records: {e}"
        ) from e

    return all_records