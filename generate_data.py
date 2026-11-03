import os

import numpy as np  
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42                      # fixed seed -> reproducible output
TARGET_TOTAL = 2000            # approximate number of records
N_WEEKS = 12
START_DATE = pd.Timestamp("2026-01-05")   # arbitrary Monday; week 1 starts here
OUTPUT_PATH = os.path.join("data", "synthetic_responses.csv")
PRIVACY_THRESHOLD = 5          # illustrative minimum count per province-week

THEMES = [
    "Job and Money",
    "Exams and Studies",
    "Loneliness and Isolation",
    "Family and Relationships",
    "Other",
]
LANGUAGES = ["Sinhala", "Tamil", "English"]

# Share of the overall volume per province (arbitrary, sums to 1 for the
# provinces other than Northern, which is handled separately below).
PROVINCE_VOLUME_SHARE = {
    "Western": 0.26,
    "Central": 0.14,
    "Southern": 0.14,
    "Eastern": 0.09,
    "North Western": 0.12,
    "North Central": 0.08,
    "Uva": 0.08,
    "Sabaragamuwa": 0.09,
}

# Baseline theme probabilities per province, in THEMES order.
# Provinces other than the scenario ones get deliberately different mixes.
BASELINE_THEMES = {
    "Western":       [0.26, 0.18, 0.22, 0.22, 0.12],
    "Central":       [0.20, 0.20, 0.22, 0.26, 0.12],
    "Southern":      [0.22, 0.20, 0.20, 0.26, 0.12],   # kept stable
    "Northern":      [0.24, 0.20, 0.24, 0.20, 0.12],
    "Eastern":       [0.30, 0.14, 0.18, 0.28, 0.10],
    "North Western": [0.18, 0.26, 0.16, 0.28, 0.12],
    "North Central": [0.32, 0.12, 0.20, 0.24, 0.12],
    "Uva":           [0.24, 0.22, 0.28, 0.16, 0.10],
    "Sabaragamuwa":  [0.20, 0.24, 0.18, 0.24, 0.14],
}

# Language probabilities per province, in LANGUAGES order (assumptions only).
LANGUAGE_PROBS = {
    "Western":       [0.55, 0.15, 0.30],
    "Central":       [0.65, 0.20, 0.15],
    "Southern":      [0.85, 0.03, 0.12],
    "Northern":      [0.02, 0.93, 0.05],
    "Eastern":       [0.30, 0.60, 0.10],
    "North Western": [0.80, 0.10, 0.10],
    "North Central": [0.88, 0.05, 0.07],
    "Uva":           [0.78, 0.15, 0.07],
    "Sabaragamuwa":  [0.82, 0.10, 0.08],
}

# Scenario weeks (1-indexed) and the extra probability mass shifted to a theme.
# Western: financial stress rises in weeks 7-10, peaking in weeks 8-9.
WESTERN_JOB_BOOST = {7: 0.08, 8: 0.16, 9: 0.16, 10: 0.08}
# Central: exam stress rises in weeks 5-8, peaking in weeks 6-7.
CENTRAL_EXAM_BOOST = {5: 0.08, 6: 0.18, 7: 0.18, 8: 0.08}


def theme_probs(province: str, week: int) -> np.ndarray:
    """Return theme probabilities for a province-week, applying scenarios."""
    p = np.array(BASELINE_THEMES[province], dtype=float)

    if province == "Western" and week in WESTERN_JOB_BOOST:
        p[THEMES.index("Job and Money")] += WESTERN_JOB_BOOST[week]
    elif province == "Central" and week in CENTRAL_EXAM_BOOST:
        p[THEMES.index("Exams and Studies")] += CENTRAL_EXAM_BOOST[week]

    return p / p.sum()          # renormalise so probabilities sum to 1


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []

    # Reserve a small number of records for Northern (below), the rest are
    # spread over the other provinces by their volume share.
    expected_northern_total = 2 * N_WEEKS
    remaining = TARGET_TOTAL - expected_northern_total

    for province in BASELINE_THEMES:
        for week in range(1, N_WEEKS + 1):
            if province == "Northern":
                # Scenario 4: only 1-3 responses per week, always below the
                # privacy threshold, so the dashboard should suppress them.
                n = int(rng.integers(1, 4))
            else:
                expected = remaining * PROVINCE_VOLUME_SHARE[province] / N_WEEKS
                n = int(rng.poisson(expected))

            if n == 0:
                continue

            themes = rng.choice(THEMES, size=n, p=theme_probs(province, week))
            langs = rng.choice(LANGUAGES, size=n, p=LANGUAGE_PROBS[province])
            day_offsets = rng.integers(0, 7, size=n)
            dates = START_DATE + pd.to_timedelta((week - 1) * 7 + day_offsets, unit="D")

            rows.append(pd.DataFrame({
                "province": province,
                "week": week,
                "response_date": dates,
                "theme": themes,
                "language": langs,
            }))

    df = pd.concat(rows, ignore_index=True)

    # Sort chronologically (stable tie-breaks), then assign sequential IDs.
    df = df.sort_values(["response_date", "province"], kind="stable").reset_index(drop=True)
    df.insert(0, "response_id", [f"SYN-{i:05d}" for i in range(1, len(df) + 1)])
    df["response_date"] = df["response_date"].dt.strftime("%Y-%m-%d")
    df["data_type"] = "synthetic"   # explicit label on every row

    # Save (create data/ if needed).
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print("=" * 60)
    print("SYNTHETIC DATASET - not real survey data")
    print("=" * 60)
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Records:  {len(df)}")
    print(f"Dates:    {df['response_date'].min()} to {df['response_date'].max()}")
    print(f"Weeks:    {df['week'].nunique()}")

    print("\nRecords per province:")
    print(df["province"].value_counts().to_string())

    print("\nRecords per theme:")
    print(df["theme"].value_counts().to_string())

    print("\nRecords per language:")
    print(df["language"].value_counts().to_string())

    print("\nWestern - share of 'Job and Money' by week:")
    w = df[df["province"] == "Western"]
    print((w["theme"].eq("Job and Money").groupby(w["week"]).mean().round(2)).to_string())

    print("\nCentral - share of 'Exams and Studies' by week:")
    c = df[df["province"] == "Central"]
    print((c["theme"].eq("Exams and Studies").groupby(c["week"]).mean().round(2)).to_string())

    print(f"\nNorthern - responses per week (privacy threshold = {PRIVACY_THRESHOLD}):")
    n_counts = (df[df["province"] == "Northern"].groupby("week").size()
                .reindex(range(1, N_WEEKS + 1), fill_value=0))
    print(n_counts.to_string())
    print(f"Northern weeks below threshold: {(n_counts < PRIVACY_THRESHOLD).sum()} of {N_WEEKS}")


if __name__ == "__main__":
    main()