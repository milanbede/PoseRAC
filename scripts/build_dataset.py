#!/usr/bin/env python
"""Assemble the combined RepCount_pose + own wake-strong native-17 dataset.

Classes (9): front_raise, pull_up, squat, bench_pressing, jump_jack, situp,
push_up, pommelhorse, jab.

Output: data_rtmpose/dataset.npz
  X (N,34) float32, Y (N,9) float32 multi-hot, metric (N,) int64,
  class_name (N,) <U16, source (N,) <U64, frame (N,) int64, part (N,) <U16
plus data_rtmpose/dataset_meta.json with counts, holdout videos and provenance.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import normalize_sequence  # noqa: E402
from phase_alignment import POLICY, align_examples  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CLASSES = ["front_raise", "pull_up", "squat", "bench_pressing", "jump_jack",
           "situp", "push_up", "pommelhorse", "jab"]
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
OWN_CLASS = {"push_up": "push_up", "jumping_jack": "jump_jack", "jab": "jab"}
HOLDOUT_FRACTION = 0.1
SEED = 42


def log(message):
    print(message, flush=True)


_POSE_TRAIN_CACHE = {}


def lookup_video(root, name):
    """(exercise, salient1, salient2) from pose_train.csv for one video."""
    if not _POSE_TRAIN_CACHE:
        with open(Path(root) / "annotation" / "pose_train.csv", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 4 or row[1] == "type":
                    continue
                indices = [int(float(value)) for value in row[3:] if value not in ("", "nan")]
                _POSE_TRAIN_CACHE[row[2]] = (row[1], indices[::2], indices[1::2])
    return _POSE_TRAIN_CACHE.get(name, (None, [], []))


def load_repcount(repcount_dir, root):
    manifest_path = repcount_dir / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest and manifest.get("complete"):
        records = manifest["videos"]
    else:  # extraction still running (or crashed): use every completed npz
        records = [{"video": path.stem + ".mp4", "path": f"train_salient/{path.name}",
                    "exercise": exercise, "salient1": s1, "salient2": s2}
                   for path in sorted((repcount_dir / "train_salient").glob("*.npz"))
                   for exercise, s1, s2 in [lookup_video(root, path.stem + ".mp4")]]
    records = [record for record in records if (repcount_dir / record["path"]).exists()]
    manifest = {"videos": records}
    names = sorted(record["video"] for record in manifest["videos"])
    rng = random.Random(SEED)
    shuffled = names[:]
    rng.shuffle(shuffled)
    holdout = set(shuffled[:max(1, int(len(shuffled) * HOLDOUT_FRACTION))])
    examples = []
    for record in manifest["videos"]:
        exercise = record["exercise"]
        if exercise not in CLASS_INDEX:
            log(f"  unknown RepCount class {exercise!r} in {record['video']}; skipped")
            continue
        data = np.load(repcount_dir / record["path"])
        features, mask = normalize_sequence(data["xy"], data["score"], data["detected"])
        position = {int(frame): index for index, frame in enumerate(data["frame"])}
        part = "repcount_val" if record["video"] in holdout else "repcount_train"
        class_index = CLASS_INDEX[exercise]
        for indices, target in ((record["salient1"], 1.0), (record["salient2"], 0.0)):
            for frame in indices:
                index = position.get(int(frame))
                if index is None or not mask[index]:
                    continue
                label = np.zeros(len(CLASSES), np.float32)
                if target:
                    label[class_index] = 1.0
                examples.append((features[index], label, class_index, exercise,
                                 record["video"], int(frame), part))
    return examples, sorted(holdout)


def load_own(own_dir):
    sources = json.loads((own_dir / "sources.json").read_text())
    all_pairs = json.loads((own_dir / "pairs.json").read_text())
    examples = []
    counts = {}
    for source_id, pairs in sorted(all_pairs.items()):
        entry = sources[source_id]
        class_name = OWN_CLASS.get(entry["exercise"])
        if class_name is None:
            continue
        split = entry["split"]
        if split not in {"train", "validation"}:
            continue  # Test poses and labels are not training/preprocessing inputs.
        part = {"train": "own_train", "test": "own_test", "validation": "own_val"}.get(split, "own_other")
        data = np.load(own_dir / "poses" / (source_id + ".npz"))
        features, mask = normalize_sequence(data["xy"], data["score"], data["valid"])
        class_index = CLASS_INDEX[class_name]
        kept = 0
        for pair in pairs:
            for key, target in (("opposite_index", 1.0), ("return_index", 0.0)):
                index = int(pair[key])
                if index < 0 or index >= len(mask) or not mask[index]:
                    continue
                label = np.zeros(len(CLASSES), np.float32)
                if target:
                    label[class_index] = 1.0
                examples.append((features[index], label, class_index, class_name,
                                 source_id, index, part))
                kept += 1
        counts[source_id] = kept
    return examples, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repcount", type=Path, default=REPO / "data_rtmpose" / "repcount")
    parser.add_argument("--root", type=Path,
                        default=Path("/Volumes/Vibefare NAS/Software Projects/wake-strong/datasets/RepCount_pose"))
    parser.add_argument("--own", type=Path, default=REPO / "data_rtmpose" / "own")
    parser.add_argument("--out", type=Path, default=REPO / "data_rtmpose")
    args = parser.parse_args()

    repcount, holdout = load_repcount(args.repcount, args.root)
    own, own_counts = load_own(args.own)
    log(f"repcount examples: {len(repcount)} (holdout videos: {len(holdout)})")
    log(f"own examples: {len(own)} across {len(own_counts)} recordings")

    examples, alignment = align_examples(repcount + own)
    X = np.stack([e[0] for e in examples]).astype(np.float32)
    Y = np.stack([e[1] for e in examples]).astype(np.float32)
    metric = np.asarray([e[2] for e in examples], np.int64)
    class_name = np.asarray([e[3] for e in examples], dtype="U16")
    source = np.asarray([e[4] for e in examples], dtype="U64")
    frame = np.asarray([e[5] for e in examples], np.int64)
    part = np.asarray([e[6] for e in examples], dtype="U16")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "dataset.npz", X=X, Y=Y, metric=metric,
                        class_name=class_name, source=source, frame=frame, part=part)
    parts, counts = np.unique(part, return_counts=True)
    positives = {CLASSES[index]: int((Y[Y[:, index] > 0][:, index]).shape[0]) for index in range(len(CLASSES))}
    meta = {
        "classes": CLASSES,
        "examples": int(len(X)),
        "parts": {str(k): int(v) for k, v in zip(parts, counts)},
        "positive_salient1_per_class": positives,
        "own_pairs_per_recording": own_counts,
        "holdout_videos": holdout,
        "holdout_fraction": HOLDOUT_FRACTION,
        "seed": SEED,
        "phase_policy": POLICY,
        "phase_alignment": alignment,
        "sha256": hashlib.sha256((args.out / "dataset.npz").read_bytes()).hexdigest(),
    }
    (args.out / "dataset_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"parts: {meta['parts']}")
    log(f"positive salient1 per class: {positives}")
    log(f"dataset -> {args.out / 'dataset.npz'}")


if __name__ == "__main__":
    main()
