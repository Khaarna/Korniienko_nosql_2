
import json
import random
from collections import Counter
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm


INPUT_FILE = Path("arxiv-metadata-oai-snapshot.json")
OUTPUT_FILE = Path("data/arxiv_subset.parquet")

MAX_RECORDS = 10_000
RANDOM_SEED = 42

# Target sizes for year and category groups
QUOTAS = {
    ("before_2015", "cs.LG"): 1000,
    ("before_2015", "other"): 1500,
    ("2015_2019", "cs.LG"): 1000,
    ("2015_2019", "other"): 1500,
    ("2020_2021", "cs.LG"): 1000,
    ("2020_2021", "other"): 1500,
    ("2022_plus", "cs.LG"): 1000,
    ("2022_plus", "other"): 1500,
}


def clean_text(value):
    # Remove unnecessary whitespace and line breaks
    return " ".join(str(value or "").split())


def extract_year(paper):
    # Use the first arXiv submission date
    versions = paper.get("versions", [])

    if versions:
        try:
            return parsedate_to_datetime(
                versions[0]["created"]
            ).year
        except (KeyError, ValueError, TypeError, IndexError):
            pass

    # Fall back to the last update date
    return int(paper.get("update_date", "2000")[:4])


def format_authors(paper):
    parsed = paper.get("authors_parsed", [])

    if parsed:
        authors = []

        for entry in parsed[:10]:
            last = clean_text(entry[0]) if len(entry) > 0 else ""
            first = clean_text(entry[1]) if len(entry) > 1 else ""

            if last:
                authors.append(f"{last} {first}".strip())

        if authors:
            return ", ".join(authors)

    return clean_text(paper.get("authors", ""))


def get_period(year):
    if year < 2015:
        return "before_2015"
    if year <= 2019:
        return "2015_2019"
    if year <= 2021:
        return "2020_2021"
    return "2022_plus"


def reservoir_add(reservoir, record, count, capacity, rng):
    # Keep a uniform random sample from a data stream
    if len(reservoir) < capacity:
        reservoir.append(record)
    else:
        position = rng.randrange(count)

        if position < capacity:
            reservoir[position] = record


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    rng = random.Random(RANDOM_SEED)

    # Maintain separate samples for each group
    reservoirs = {key: [] for key in QUOTAS}
    seen = Counter()

    # Global fallback sample for groups with insufficient records
    fallback = []
    total_valid = 0

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Reading arXiv metadata"):
            if not line.strip():
                continue

            paper = json.loads(line)

            title = clean_text(paper.get("title", ""))
            abstract = clean_text(paper.get("abstract", ""))

            if not title or not abstract:
                continue

            categories = paper.get("categories", "").split()
            primary_category = (
                categories[0] if categories else "unknown"
            )

            year = extract_year(paper)

            record = {
                "id": paper["id"],
                "title": title,
                "abstract": abstract,
                "authors": format_authors(paper),
                "year": year,
                "category": primary_category,
            }

            category_group = (
                "cs.LG"
                if primary_category == "cs.LG"
                else "other"
            )

            group = (get_period(year), category_group)

            total_valid += 1
            seen[group] += 1

            reservoir_add(
                reservoirs[group],
                record,
                seen[group],
                QUOTAS[group],
                rng,
            )

            reservoir_add(
                fallback,
                record,
                total_valid,
                MAX_RECORDS,
                rng,
            )

    # Combine stratified samples
    selected = [
        record
        for group_records in reservoirs.values()
        for record in group_records
    ]

    # Fill remaining places from the global sample
    selected_ids = {record["id"] for record in selected}

    for record in fallback:
        if len(selected) >= MAX_RECORDS:
            break

        if record["id"] not in selected_ids:
            selected.append(record)
            selected_ids.add(record["id"])

    if len(selected) != MAX_RECORDS:
        raise RuntimeError(
            f"Expected {MAX_RECORDS} papers, got {len(selected)}"
        )

    # Shuffle and save the selected records
    rng.shuffle(selected)

    df = pd.DataFrame(selected)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nValid papers in source: {total_valid}")
    print(f"Selected papers: {len(df)}")

    print("\nTop 10 categories:")
    print(df["category"].value_counts().head(10))

    print("\nDistribution by period:")
    print(
        df["year"]
        .apply(get_period)
        .value_counts()
        .sort_index()
    )

    print("\nPublication years:")
    print(df["year"].describe())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nExample paper:")
    print(df.iloc[0].to_dict())

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
