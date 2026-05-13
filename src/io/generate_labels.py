import json
from pathlib import Path


def generate_labels(useful_dir="data/useful", not_useful_dir="data/not-useful", output_file="data/labels.json"):
    labels = {}

    # Label all PDFs in "useful" folder
    for pdf in Path(useful_dir).glob("*.pdf"):
        labels[f"{pdf.stem}.txt"] = "useful"

    # Label all PDFs in "not_useful" folder
    for pdf in Path(not_useful_dir).glob("*.pdf"):
        labels[f"{pdf.stem}.txt"] = "not useful"

    # Save labels to JSON file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4)

    print(f"labels.json created with {len(labels)} entries at {output_file}")


if __name__ == "__main__":
    generate_labels()
