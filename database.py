import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).resolve().parent / ".env")


def get_admin_client():
    """Client for trusted server-side database operations."""
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not secret_key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_SECRET_KEY in .env"
        )

    return create_client(url, secret_key)


def get_read_client():
    """Client for reading data using the publishable key."""
    url = os.getenv("SUPABASE_URL")
    publishable_key = os.getenv("SUPABASE_PUBLISHABLE_KEY")

    if not url or not publishable_key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY in .env"
        )

    return create_client(url, publishable_key)


def fetch_responses():
    """Retrieve every response, in pages, from Supabase."""
    client = get_read_client()

    all_records = []
    page_size = 1000
    start = 0

    while True:
        response = (
            client.table("wellbeing_responses")
            .select(
                "id, province, district, week_number, "
                "theme, language, is_synthetic, created_at"
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

    return all_records

def get_admin_client():
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not secret_key:
        raise ValueError("Missing Supabase credentials")

    return create_client(url, secret_key)