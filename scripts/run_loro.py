#!/usr/bin/env python
"""Leave-one-recording-out evaluation of own wake-strong recordings.

For each own train recording with reviewed reps, train a model that never sees
that recording (RepCount train + the other own recordings), then count reps on it.
Folds are resumable: an existing runs/loro/<source>/model.pt is reused.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own", type=Path, default=REPO / "data_rtmpose" / "own")
    parser.add_argument("--dataset", type=Path, default=REPO / "data_rtmpose" / "dataset.npz")
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "loro.json")
    parser.add_argument("--runs", type=Path, default=REPO / "runs" / "loro")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--params", type=Path, default=None,
                        help="per-exercise decode params for evaluation")
    args = parser.parse_args()

    sources = json.loads((args.own / "sources.json").read_text())
    folds = []
    for source_id, entry in sorted(sources.items()):
        if entry["split"] != "train" or not entry["pairs"]:
            continue
        if args.source and source_id not in args.source:
            continue
        folds.append(source_id)

    results = []
    for position, source_id in enumerate(folds, 1):
        run_dir = args.runs / source_id
        model_path = run_dir / "model.pt"
        if not model_path.exists():
            log(f"[{position}/{len(folds)}] training fold excluding {source_id[:8]}")
            subprocess.run([sys.executable, str(REPO / "scripts" / "train_rtmpose.py"),
                            "--dataset", str(args.dataset), "--out", str(run_dir),
                            "--exclude", source_id, "--epochs", str(args.epochs),
                            "--patience", str(args.patience), "--alpha", str(args.alpha)], check=True)
        else:
            log(f"[{position}/{len(folds)}] reusing fold model for {source_id[:8]}")
        evaluation = run_dir / "eval.json"
        command = [sys.executable, str(REPO / "scripts" / "eval_rtmpose.py"),
                   "--model", str(model_path), "--out", str(evaluation),
                   "own", "--own", str(args.own), "--source", source_id]
        if args.params:
            command += ["--params", str(args.params)]
        subprocess.run(command, check=True)
        payload = json.loads(evaluation.read_text())
        results.extend(payload["rows"])

    exercise = {}
    for row in results:
        bucket = exercise.setdefault(row["exercise"], [])
        bucket.append(row)
    summary = {}
    for name, rows in sorted(exercise.items()):
        summary[name] = {
            "recordings": len(rows),
            "reference_reps": sum(r["reference_count"] for r in rows),
            "predicted_reps": sum(r["predicted_count"] for r in rows),
            "mae": sum(r["mae"] for r in rows) / len(rows),
            "obo": sum(r["obo"] for r in rows) / len(rows),
            "f1": sum(r["timing"]["f1"] for r in rows) / len(rows),
            "precision": sum(r["timing"]["precision"] for r in rows) / len(rows),
            "recall": sum(r["timing"]["recall"] for r in rows) / len(rows),
        }
    output = {"folds": folds, "rows": results, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1))
    log(f"LORO done: {len(folds)} folds, {len(results)} recordings -> {args.out}")
    for name, stats in summary.items():
        log(f"  {name:13} mae={stats['mae']:.2f} obo={stats['obo']:.2f} f1={stats['f1']:.2f} "
            f"pred/ref={stats['predicted_reps']}/{stats['reference_reps']}")


if __name__ == "__main__":
    main()
