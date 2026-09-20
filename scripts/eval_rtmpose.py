#!/usr/bin/env python
"""Evaluate a native-17 PoseRAC checkpoint.

Subcommands:
  repcount   MAE/OBO on RepCount test videos (oracle-class and predicted-class)
  own        per-recording counts on own wake-strong sources, with timing F1
  holdout    RepCount video holdout (same protocol, for model selection checks)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import normalize_sequence  # noqa: E402
from rtmpose_lib import CLASSES, count_repetitions, load_model, repetition_events  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REPCOUNT = Path("/Volumes/Vibefare NAS/Software Projects/wake-strong/datasets/RepCount_pose")
DEFAULT_OWN = REPO / "data_rtmpose" / "own"
TOLERANCE_US = 500_000
REPCOUNT_CLASSES = [name for name in CLASSES if name != "jab"]


def log(message):
    print(message, flush=True)


def read_repcount_labels(root, split=None, label_file=None):
    path = Path(label_file) if label_file else (
        Path(root) / "annotation" / ("test.csv" if split == "test" else "video_train.csv"))
    labels = {}
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 4 or row[1] == "type" or not row[3].strip():
                continue
            labels[row[2]] = {"count": int(float(row[3])), "type": row[1]}
    return labels


@torch.inference_mode()
def probabilities(model, npz, device):
    data = np.load(npz)
    features, mask = normalize_sequence(data["xy"], data["score"], data["detected"])
    probs = np.zeros((len(features), len(CLASSES)), np.float32)
    if mask.any():
        x = torch.from_numpy(features[mask]).to(device)
        probs[mask] = torch.sigmoid(model(x)).cpu().numpy()
    return probs, mask, data["timestamp_us"]


def match_events(ground_truth, predicted, tolerance=TOLERANCE_US):
    used, matches, timing = set(), 0, []
    for guess in predicted:
        best, best_delta = None, tolerance + 1
        for index, reference in enumerate(ground_truth):
            if index in used:
                continue
            delta = abs(guess - reference)
            if delta <= tolerance and delta < best_delta:
                best, best_delta = index, delta
        if best is not None:
            used.add(best)
            matches += 1
            timing.append(best_delta)
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matched": matches,
            "predicted": len(predicted), "reference": len(ground_truth),
            "mean_timing_error_us": float(np.mean(timing)) if timing else None}


def eval_repcount(args, model, device):
    labels = read_repcount_labels(args.root, "test", getattr(args, "label_file", None))
    directory = Path(args.test_dir)
    rows = []
    for path in sorted(directory.glob("*.npz")):
        name = path.stem + ".mp4"
        if name not in labels:
            continue
        gt = labels[name]["count"]
        probs, mask, _ = probabilities(model, path, device)
        mean_probs = probs[mask].mean(axis=0) if mask.any() else np.zeros(len(CLASSES), np.float32)
        predicted_class = max(REPCOUNT_CLASSES, key=lambda c: float(mean_probs[CLASSES.index(c)]))
        predicted = count_repetitions(probs[:, CLASSES.index(predicted_class)], mask)
        gt_class = labels[name]["type"]
        known = count_repetitions(probs[:, CLASSES.index(gt_class)], mask) if gt_class in REPCOUNT_CLASSES else None
        oracle = None
        for class_name in REPCOUNT_CLASSES:
            count = count_repetitions(probs[:, CLASSES.index(class_name)], mask)
            error = abs(gt - count)
            if oracle is None or error < oracle["error"]:
                oracle = {"class": class_name, "count": count, "error": error}
        rows.append({
            "video": name, "gt": gt, "gt_type": labels[name]["type"],
            "predicted_class": predicted_class, "predicted_count": predicted,
            "gt_class_count": known,
            "gt_class_mae": abs(gt - known) / (gt + 1e-9) if known is not None else None,
            "gt_class_obo": (1.0 if abs(gt - known) <= 1 else 0.0) if known is not None else None,
            "oracle_class": oracle["class"], "oracle_count": oracle["count"],
            "oracle_mae": oracle["error"] / (gt + 1e-9),
            "oracle_obo": 1.0 if abs(gt - oracle["count"]) <= 1 else 0.0,
            "predicted_mae": abs(gt - predicted) / (gt + 1e-9),
            "predicted_obo": 1.0 if abs(gt - predicted) <= 1 else 0.0,
        })
        if len(rows) % 20 == 0:
            log(f"  {len(rows)} videos")
    summary = {
        "videos": len(rows),
        "oracle_mae": float(np.mean([r["oracle_mae"] for r in rows])) if rows else None,
        "oracle_obo": float(np.mean([r["oracle_obo"] for r in rows])) if rows else None,
        "predicted_class_mae": float(np.mean([r["predicted_mae"] for r in rows])) if rows else None,
        "predicted_class_obo": float(np.mean([r["predicted_obo"] for r in rows])) if rows else None,
        "gt_class_mae": float(np.mean([r["gt_class_mae"] for r in rows if r["gt_class_mae"] is not None])) if rows else None,
        "gt_class_obo": float(np.mean([r["gt_class_obo"] for r in rows if r["gt_class_obo"] is not None])) if rows else None,
        "oracle_class_matches_gt_type": float(np.mean([r["oracle_class"] == r["gt_type"] for r in rows])) if rows else None,
    }
    return {"summary": summary, "rows": rows}


def eval_own_recording(model, device, own_dir, source_id, enter=0.78, exit=0.4, momentum=0.4):
    sources = json.loads((own_dir / "sources.json").read_text())
    entry = sources[source_id]
    exercise = entry["exercise"]
    class_name = {"push_up": "push_up", "jumping_jack": "jump_jack", "jab": "jab"}.get(exercise)
    if class_name is None:
        return None
    pairs = json.loads((own_dir / "poses" / (source_id + ".pairs.json")).read_text())
    data = np.load(own_dir / "poses" / (source_id + ".npz"))
    features, mask = normalize_sequence(data["xy"], data["score"], data["valid"])
    probs = np.zeros((len(features), len(CLASSES)), np.float32)
    if mask.any():
        with torch.inference_mode():
            probs[mask] = torch.sigmoid(model(torch.from_numpy(features[mask]).to(device))).cpu().numpy()
    class_index = CLASSES.index(class_name)
    count = count_repetitions(probs[:, class_index], mask, enter=enter, exit=exit, momentum=momentum)
    events = repetition_events(probs[:, class_index], data["ts"], mask,
                               enter=enter, exit=exit, momentum=momentum)
    reference = sorted(int(pair["completion_us"]) for pair in pairs)
    timing = match_events(reference, events)
    ground_truth = len(pairs)
    return {
        "source_id": source_id, "exercise": exercise, "split": entry["split"],
        "frames": int(len(mask)), "valid_frames": int(mask.sum()),
        "predicted_count": count, "reference_count": ground_truth,
        "count_error": count - ground_truth,
        "mae": abs(ground_truth - count) / (ground_truth + 1e-9),
        "obo": 1.0 if abs(ground_truth - count) <= 1 else 0.0,
        "timing": timing,
    }


def eval_own(args, model, device, source_ids=None):
    enter = getattr(args, "enter", 0.78)
    exit_ = getattr(args, "exit", 0.4)
    momentum = getattr(args, "momentum", 0.4)
    params = {}
    if getattr(args, "params", None):
        params = json.loads(Path(args.params).read_text()).get("params", {})
    sources = json.loads((args.own / "sources.json").read_text())
    directory = args.own / "poses"
    rows = []
    for path in sorted(directory.glob("*.npz")):
        source_id = path.stem
        if source_id not in sources or not (directory / (source_id + ".pairs.json")).exists():
            continue
        if source_ids and source_id not in source_ids:
            continue
        tuning = params.get(sources[source_id]["exercise"], {})
        result = eval_own_recording(model, device, args.own, source_id,
                                    enter=tuning.get("enter", enter),
                                    exit=tuning.get("exit", exit_),
                                    momentum=tuning.get("momentum", momentum))
        if result is None:
            continue
        rows.append(result)
        log(f"  {source_id[:8]} {result['exercise']:13} pred={result['predicted_count']:3} "
            f"gt={result['reference_count']:3} f1={result['timing']['f1']:.2f}")
    return {"rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true")
    sub = parser.add_subparsers(dest="mode", required=True)

    repcount = sub.add_parser("repcount")
    repcount.add_argument("--root", type=Path, default=DEFAULT_REPCOUNT)
    repcount.add_argument("--test-dir", type=Path, default=REPO / "data_rtmpose" / "repcount" / "test")
    repcount.add_argument("--label-file", type=Path, default=None,
                          help="override label csv (e.g. video_train.csv for holdout videos)")

    own = sub.add_parser("own")
    own.add_argument("--own", type=Path, default=DEFAULT_OWN)
    own.add_argument("--source", action="append", default=None)
    own.add_argument("--enter", type=float, default=0.78)
    own.add_argument("--exit", type=float, default=0.4)
    own.add_argument("--momentum", type=float, default=0.4)
    own.add_argument("--params", type=Path, default=None,
                     help="per-exercise decode params from tune_decode.py")

    args = parser.parse_args()
    device = torch.device("cpu") if args.cpu else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, config = load_model(args.model, device)
    log(f"loaded {args.model} on {device}")
    if args.mode == "repcount":
        result = eval_repcount(args, model, device)
    else:
        result = eval_own(args, model, device, source_ids=set(args.source) if args.source else None)
    result["model"] = str(args.model)
    result["config"] = config
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        log(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
