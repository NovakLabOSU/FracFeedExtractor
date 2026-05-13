from pathlib import Path


def load_processed_text(directory="data/processed-text"):
    texts = []
    for txt_file in Path(directory).glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


if __name__ == "__main__":
    texts = load_processed_text()
    print(f"Loaded {len(texts)} text files.")
    if texts:
        print(f"First file preview:\n{texts[0][:300]}...")
