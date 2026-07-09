"""Benchmark extraction accuracy against ground-truth labels.

Reads a ground-truth CSV and a directory of per-paper JSON results produced
by the pipeline, then computes per-field accuracy.

Usage::

    python scripts/benchmark.py \\
        --results  data/results/metrics/ \\
        --ground-truth data/ground_truth.csv \\
        [--tolerance 0]          # exact match only (default)
        [--mismatch-log out.csv] # write detailed mismatch log

Ground-truth CSV format
-----------------------
The first column must be ``filename`` (the PDF or txt filename, with or without
extension). Remaining columns are field names matching the FIELDS registry in
``src/config.py`` (e.g. ``species_name``, ``num_empty``, ``num_sampled``, …).
Leave a cell blank to indicate the field is null/unknown for that paper.

Result JSON format
------------------
Standard pipeline output: a JSON object with a ``records`` list. Each record
is a dict with the same field names as the ground-truth CSV. When a paper
produced multiple records (e.g. two species), the first record is used for
comparison — or the record whose ``species_name`` matches the ground truth.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIELDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_ground_truth(csv_path: Path) -> dict[str, dict]:
    """Return {filename_stem: {field: value_or_None}} from the ground-truth CSV."""
    records = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            if not filename:
                continue
            stem = Path(filename).stem
            record = {}
            for field in FIELDS:
                raw = row.get(field.name, "").strip()
                record[field.name] = raw if raw else None
            records[stem] = record
    return records


def _load_result(json_path: Path) -> dict | None:
    """Return the first record dict from a pipeline result JSON, or None."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        records = data.get("records", [])
        return records[0] if records else None
    except Exception as e:
        print(f"  [WARN] Could not read {json_path.name}: {e}", file=sys.stderr)
        return None


def _best_record(result_json: Path, gt_species: str | None) -> dict | None:
    """Return the record from a result JSON that best matches gt_species."""
    try:
        data = json.loads(result_json.read_text(encoding="utf-8"))
        records = data.get("records", [])
    except Exception:
        return None
    if not records:
        return None
    if gt_species:
        for rec in records:
            if str(rec.get("species_name", "")).strip().lower() == gt_species.strip().lower():
                return rec
    return records[0]


def _values_match(predicted, truth, tolerance: float) -> bool:
    """Return True if predicted and truth are equivalent within tolerance."""
    if truth is None and predicted is None:
        return True
    if truth is None or predicted is None:
        return False
    # Try numeric comparison within tolerance
    try:
        p_num = float(str(predicted).strip())
        t_num = float(str(truth).strip())
        if tolerance == 0:
            return p_num == t_num
        return abs(p_num - t_num) <= tolerance
    except (ValueError, TypeError):
        pass
    # String comparison (case-insensitive strip)
    return str(predicted).strip().lower() == str(truth).strip().lower()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def benchmark(results_dir: Path, ground_truth_path: Path, tolerance: float, mismatch_log: Path | None):
    gt = _load_ground_truth(ground_truth_path)
    if not gt:
        print("[ERROR] Ground-truth CSV is empty or has no 'filename' column.", file=sys.stderr)
        sys.exit(1)

    field_names = [f.name for f in FIELDS]
    correct: dict[str, int] = {f: 0 for f in field_names}
    total: dict[str, int] = {f: 0 for f in field_names}
    mismatches: list[dict] = []
    papers_matched = 0
    papers_missing = 0

    for stem, gt_record in gt.items():
        # Find the matching result JSON (try stem_results.json, stem.json, etc.)
        candidates = list(results_dir.glob(f"{stem}*.json"))
        if not candidates:
            print(f"  [MISS] No result JSON for {stem!r}", file=sys.stderr)
            papers_missing += 1
            continue

        pred_record = _best_record(candidates[0], gt_record.get("species_name"))
        if pred_record is None:
            print(f"  [MISS] Empty result for {stem!r}", file=sys.stderr)
            papers_missing += 1
            continue

        papers_matched += 1
        for field in field_names:
            gt_val = gt_record.get(field)
            pr_val = pred_record.get(field)
            # Skip fields the ground truth left blank
            if gt_val is None:
                continue
            total[field] += 1
            if _values_match(pr_val, gt_val, tolerance):
                correct[field] += 1
            else:
                mismatches.append(
                    {
                        "paper": stem,
                        "field": field,
                        "ground_truth": gt_val,
                        "predicted": pr_val,
                    }
                )

    # ── Print accuracy table ──────────────────────────────────────────────
    print(f"\nBenchmark results ({papers_matched} papers matched, {papers_missing} missing)\n")
    header = f"{'Field':<25}  {'Correct':>7}  {'Total':>7}  {'Accuracy':>9}"
    print(header)
    print("-" * len(header))
    overall_correct = overall_total = 0
    for field in field_names:
        n_corr = correct[field]
        n_tot = total[field]
        overall_correct += n_corr
        overall_total += n_tot
        acc = f"{100 * n_corr / n_tot:.1f}%" if n_tot else "—"
        print(f"  {field:<23}  {n_corr:>7}  {n_tot:>7}  {acc:>9}")
    print("-" * len(header))
    overall_acc = f"{100 * overall_correct / overall_total:.1f}%" if overall_total else "—"
    print(f"  {'OVERALL':<23}  {overall_correct:>7}  {overall_total:>7}  {overall_acc:>9}\n")

    # ── Mismatch log ──────────────────────────────────────────────────────
    if mismatch_log and mismatches:
        with open(mismatch_log, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["paper", "field", "ground_truth", "predicted"])
            writer.writeheader()
            writer.writerows(mismatches)
        print(f"Mismatch log written to {mismatch_log} ({len(mismatches)} mismatches)\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline extraction accuracy against ground-truth labels.")
    parser.add_argument(
        "--results",
        required=True,
        type=str,
        help="Directory containing per-paper result JSON files (e.g. data/results/metrics/).",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        type=str,
        help="Path to ground-truth CSV (columns: filename + one per extraction field).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0,
        help="Numeric tolerance for field comparison (default: 0 = exact match).",
    )
    parser.add_argument(
        "--mismatch-log",
        type=str,
        default=None,
        help="If set, write a CSV of all mismatches to this path.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        print(f"[ERROR] Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    gt_path = Path(args.ground_truth)
    if not gt_path.is_file():
        print(f"[ERROR] Ground-truth CSV not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    mismatch_log = Path(args.mismatch_log) if args.mismatch_log else None

    benchmark(results_dir, gt_path, args.tolerance, mismatch_log)


if __name__ == "__main__":
    main()
