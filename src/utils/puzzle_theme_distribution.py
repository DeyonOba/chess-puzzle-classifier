
from collections import Counter
import csv
import json
import pathlib


def calculate_themes_distribution(csv_path: str):
    themes_counter = Counter()
    num_samples = 0
    filepath = pathlib.Path(csv_path).resolve()

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if len(row) > 7:
                themes = row[7].strip().split(" ")
                themes_counter.update(themes)
                num_samples += 1

    return dict(themes_counter.most_common()), num_samples


def save_themes_count(themes: dict, output_path: str) -> None:
    output_file = pathlib.Path().resolve().joinpath(output_path)
    with open(output_file, "w") as f:
        json.dump(themes, f, indent=4)