
import numpy as np
from database import get_admin_client

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 42
TARGET_TOTAL = 5000
N_WEEKS = 12
BATCH_SIZE = 500

START_DATE = np.datetime64("2026-01-05")

DISTRICTS = ["Colombo", "Gampaha", "Kalutara"]
DISTRICT_PROBS = [0.40, 0.35, 0.25]

THEMES = [
    "Job and Money",
    "Exams and Studies",
    "Loneliness and Isolation",
    "Family and Relationships",
    "Other",
]

LANGUAGES = ["Sinhala", "Tamil", "English"]
LANGUAGE_PROBS = [0.55, 0.15, 0.30]

# Original Western Province baseline theme probabilities
BASELINE_THEMES = [
    0.26,  # Job and Money
    0.18,  # Exams and Studies
    0.22,  # Loneliness and Isolation
    0.22,  # Family and Relationships
    0.12,  # Other
]

# Original scenario: financial stress rises in weeks 7-10
WESTERN_JOB_BOOST = {
    7: 0.08,
    8: 0.16,
    9: 0.16,
    10: 0.08,
}


def get_theme_probabilities(week):
    probabilities = np.array(
        BASELINE_THEMES,
        dtype=float
    )

    if week in WESTERN_JOB_BOOST:
        probabilities[0] += WESTERN_JOB_BOOST[week]

    return probabilities / probabilities.sum()


def generate_records():

    rng = np.random.default_rng(SEED)

    records = []

    # Distribute exactly 5,000 records across 12 weeks
    weekly_counts = rng.multinomial(
        TARGET_TOTAL,
        np.ones(N_WEEKS) / N_WEEKS
    )

    for week, number_of_records in enumerate(
        weekly_counts,
        start=1
    ):

        theme_probabilities = get_theme_probabilities(week)

        for _ in range(number_of_records):

            district = rng.choice(
                DISTRICTS,
                p=DISTRICT_PROBS
            )

            theme = rng.choice(
                THEMES,
                p=theme_probabilities
            )

            language = rng.choice(
                LANGUAGES,
                p=LANGUAGE_PROBS
            )

            day_offset = int(rng.integers(0, 7))

            response_date = (
                START_DATE
                + np.timedelta64((week - 1) * 7 + day_offset, "D")
            )

            records.append({
                "province": "Western",
                "district": district,
                "week_number": week,
                "response_date": str(response_date),
                "theme": theme,
                "language": language,
                "is_synthetic": True
            })

    return records


def main():

    print("=" * 55)
    print("WESTERN PROVINCE SYNTHETIC DATA GENERATOR")
    print("=" * 55)

    client = get_admin_client()

    # Prevent accidental duplicate insertion
    existing = (
        client.table("wellbeing_responses")
        .select("id")
        .limit(1)
        .execute()
    )

    if existing.data:
        print("\nSTOPPED: Database already contains records.")
        print("No new data was inserted.")
        print("Do not delete records unless you intentionally")
        print("want to reset your demonstration dataset.")
        return

    print("\nGenerating synthetic records...")

    records = generate_records()

    print(f"Generated: {len(records)} records")
    print("Province: Western")
    print("Districts: Colombo, Gampaha, Kalutara")
    print("Weeks: 1-12")

    print("\nConnecting to Supabase...")

    # Insert records in batches
    for start in range(0, len(records), BATCH_SIZE):

        batch = records[start:start + BATCH_SIZE]

        client.table("wellbeing_responses").insert(
            batch
        ).execute()

        inserted = min(start + BATCH_SIZE, len(records))

        print(f"Uploaded {inserted}/{len(records)} records")

    print("\n" + "=" * 55)
    print("SUCCESS: DATA UPLOADED TO SUPABASE")
    print("=" * 55)

    print(f"Total records: {len(records)}")
    print("Province: Western")
    print("Districts: 3")
    print("Weeks: 12")
    print("Data type: Synthetic")
    print("CSV file: Not created")


if __name__ == "__main__":
    main()
