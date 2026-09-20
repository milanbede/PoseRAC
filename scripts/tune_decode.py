#!/usr/bin/env python
"""Tune repetition decode thresholds on LORO folds (own data only).

Each fold model never saw its held-out recording, so the grid selected here is
validation evidence, not test evidence. The chosen per-exercise parameters are
written for eval_rtmpose.py --params.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import normalize_sequence  # noqa: E402
from rtmpose_lib import CLASSES, count_repetitions, load_model, repetition_events  # noqa: E402
from eval_rtmpose import match_events  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ENTER = [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.78]
EXIT = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
MOMENTUM = [0.0, 0.1, 0.2, 0.4]
BASE = (0.78, 0.4, 0.4)
EXERCISE_CLASS = {"push_up": "push_up", "jumping_jack": "jump_jack", "jab": "jab"}


def fold_probabilities(own_dir, runs_dir, source_id, device):
    model_path = runs_dir / source_id / "model.pt"
    if not model_path.exists():
        return None
    model, _ = load_model(model_path, device)
    data = np.load(own_dir / "poses" / (source_id + ".npz"))
    features, mask = normalize_sequence(data["xy"], data["score"], data["valid"])
    probs = np.zeros((len(features), len(CLASSES)), np.float32)
    if mask.any():
        with torch.inference_mode():
            probs[mask] = torch.sigmoid(model(torch.from_numpy(features[mask]).to(device))).cpu().numpy()
    pairs = json.loads((own_dir / "poses" / (source_id + ".pairs.json")).read_text())
    reference = sorted(int(pair["completion_us"]) for pair in pairs)
    return {"probs": probs, "mask": mask, "ts": data["ts"], "reference": reference,
            "exercise": json.loads((own_dir / "sources.json").read_text())[source_id]["exercise"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own", type=Path, default=REPO / "data_rtmpose" / "own")
    parser.add_argument("--runs", type=Path, default=REPO / "runs" / "loro")
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "decode_tuning.json")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cache = {}
    for directory in sorted(args.runs.glob("*")):
        source_id = directory.name
        payload = fold_probabilities(args.own, args.runs, source_id, device)
        if payload is None:
            continue
        cache[source_id] = payload
        print(f"  {source_id[:8]} {payload['exercise']} frames={len(payload['mask'])}", flush=True)
    if not cache:
        raise SystemExit("No LORO fold models found")

    results = {}
    for exercise in sorted({payload["exercise"] for payload in cache.values()}):
        class_name = EXERCISE_CLASS.get(exercise)
        if class_name is None:
            continue
        class_index = CLASSES.index(class_name)
        folds = {sid: payload for sid, payload in cache.items() if payload["exercise"] == exercise}
        best, baseline, grid = None, None, []
        for enter, exit_, momentum in itertools.product(ENTER, EXIT, MOMENTUM):
            errors, f1s, maes = [], [], []
            for sid, payload in folds.items():
                count = count_repetitions(payload["probs"][:, class_index], payload["mask"],
                                          enter=enter, exit=exit_, momentum=momentum)
                events = repetition_events(payload["probs"][:, class_index], payload["ts"], payload["mask"],
                                           enter=enter, exit=exit_, momentum=momentum)
                reference = payload["reference"]
                errors.append(abs(count - len(reference)))
                maes.append(abs(len(reference) - count) / (len(reference) + 1e-9))
                f1s.append(match_events(reference, events)["f1"])
            row = {"enter": enter, "exit": exit_, "momentum": momentum,
                   "recordings": len(folds), "mean_abs_error": float(np.mean(errors)),
                   "mean_mae": float(np.mean(maes)), "mean_f1": float(np.mean(f1s))}
            grid.append(row)
            if best is None or (row["mean_abs_error"], -row["mean_f1"]) < (best["mean_abs_error"], -best["mean_f1"]):
                best = row
        baseline = next(row for row in grid if (row["enter"], row["exit"], row["momentum"]) == BASE)
        results[exercise] = {"best": best, "baseline": baseline, "grid": grid}
        print(f"{exercise:13} baseline mae={baseline['mean_mae']:.2f} f1={baseline['mean_f1']:.2f} | "
              f"best enter={best['enter']} exit={best['exit']} momentum={best['momentum']} "
              f"mae={best['mean_mae']:.2f} f1={best['mean_f1']:.2f}")
    params = {exercise: {key: value["best"][key] for key in ("enter", "exit", "momentum")}
              for exercise, value in results.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"params": params, "results": results}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
