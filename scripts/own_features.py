#!/usr/bin/env python
"""Build native-17 features and salient-pair labels for the own wake-strong data.

Reads the live ProtonDrive store (annotations are only current there; the NAS
.ws-data copy is a stale pre-migration snapshot).

Outputs under --out:
  poses/<source_id>.npz   ts, xy (pixels), score, valid, provenance
  pairs.json              per valid_rep opposite(1)/return(0) pairs, coco-17
  sources.json            split/exercise/revision metadata for every source

Pose inputs: existing rtmpose-performance derived runs (33-slot array reduced to
COCO-17), plus extractor npz files for recordings without a performance run
(`video` mode of rtmpose_extract.py, --name <source_id>).
"""
import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_STORE = Path("/Users/milan-bede/Library/CloudStorage/ProtonDrive-milan.bede95@proton.me-folder/Wake Strong Data/.ws-data")
DEFAULT_OUT = REPO / "data_rtmpose" / "own"
PREFERRED_RUN = "264b0ff1-ee60-47bd-bf67-e0bfedaa26c7"

# COCO-17 index -> 33-slot index used by the ws-data derived files.
COCO_TO_REVIEW = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
REVIEW_TO_COCO = {slot: coco for coco, slot in enumerate(COCO_TO_REVIEW)}
MIN_SCORE = 0.3
SHOULDERS = (5, 6)
HIPS = (11, 12)
BODY = tuple(range(5, 17))

EXERCISE_CLASS = {"push_up": "push_up", "jumping_jack": "jump_jack", "jab": "jab"}


def log(message):
    print(message, flush=True)


def latest_annotation(store, source_id):
    folder = store / "annotations" / source_id
    files = [p for p in folder.glob("*.json") if p.name[0].isdigit()]
    if not files:
        return None
    files.sort(key=lambda p: int(p.name.split("-")[0]))
    wrapped = json.loads(files[-1].read_text())
    annotation = wrapped.get("annotation", wrapped)
    annotation["_file"] = str(files[-1])
    return annotation


def load_derived_performance(store, source_id):
    folder = store / "derived" / source_id
    candidates = sorted(folder.glob("poses-rtmpose-performance-*.jsonl.gz"),
                        key=lambda p: (PREFERRED_RUN not in p.name, p.name))
    for path in candidates:
        try:
            handle = gzip.open(path, "rt")
            rows = []
            image_size = None
            for line in handle:
                row = json.loads(line)
                if row.get("type") == "header":
                    image_size = row.get("image_size") or image_size
                    continue
                rows.append(row)
            handle.close()
        except (OSError, EOFError):
            continue
        if rows:
            return path, rows, image_size
    return None, None, None


def rows_to_arrays(rows, image_size):
    """Convert 33-slot landmark rows to native COCO-17 pixels."""
    longest = float(max(image_size)) if image_size else 1.0
    stamps, xy, score, valid = [], [], [], []
    for row in rows:
        stamps.append(int(row["timestamp_us"]))
        poses = row.get("poses") or []
        if not poses:
            xy.append(np.zeros((17, 2), np.float32))
            score.append(np.zeros(17, np.float32))
            valid.append(False)
            continue
        landmarks = np.asarray(poses[0]["landmarks"], np.float32).reshape(-1, 4)
        points = np.zeros((17, 2), np.float32)
        conf = np.zeros(17, np.float32)
        for slot, coco in REVIEW_TO_COCO.items():
            points[coco] = landmarks[slot, :2] * longest
            conf[coco] = landmarks[slot, 3]
        ok = bool(conf[list(SHOULDERS) + list(HIPS)].min() >= MIN_SCORE)
        xy.append(points)
        score.append(conf)
        valid.append(ok)
    return (np.asarray(stamps, np.int64), np.asarray(xy, np.float32),
            np.asarray(score, np.float32), np.asarray(valid, bool))


def load_extracted(out, source_id):
    path = out / "videos" / (source_id + ".npz")
    if not path.exists():
        return None
    with np.load(path) as data:
        return (data["timestamp_us"].astype(np.int64), data["xy"].astype(np.float32),
                data["score"].astype(np.float32), data["detected"].astype(bool), path)


def approved_spans(annotation):
    return [span for span in annotation.get("content_intervals", [])
            if span.get("approved") and span.get("kind") == "workout"]


def cycles(annotation):
    """Mirror ws_data/saliency.py cycles() for one annotation."""
    result = {}
    events = annotation.get("events", [])
    for span in approved_spans(annotation):
        previous = span["start_us"]
        ordered = sorted((e for e in events if span["start_us"] <= e["completion_us"] < span["end_us"]),
                         key=lambda e: e["completion_us"])
        for index, event in enumerate(ordered):
            window = ordered[max(0, index - 1):index + 2]
            gaps = [b["completion_us"] - a["completion_us"]
                    for a, b in zip(window, window[1:])
                    if a.get("exercise") == event.get("exercise") == b.get("exercise")
                    and b["completion_us"] > a["completion_us"]]
            period = min(gaps) if gaps else event["completion_us"] - previous
            start = max(previous, event["completion_us"] - period,
                        event["start_us"] if event["start_us"] < event["completion_us"] else previous)
            if event.get("kind") == "valid_rep":
                result[event["id"]] = (event, start)
            previous = event["completion_us"]
    return result


def body(xy, score, valid):
    if not valid:
        return None
    if score[list(SHOULDERS) + list(HIPS)].min() < MIN_SCORE:
        return None
    hip = (xy[11] + xy[12]) / 2.0
    scale = float(np.linalg.norm((xy[5] + xy[6]) / 2.0 - hip))
    if scale < 1e-5:
        return None
    return (xy - hip) / scale, score


def derive_pairs(annotation, stamps, xy, score, valid):
    """Return (pairs, warnings); opposite=salient1 (label 1), return=salient0 (label 0)."""
    pairs, warnings = [], []
    bodies = [body(xy[i], score[i], valid[i]) for i in range(len(stamps))]
    for event_id, (event, start) in cycles(annotation).items():
        exercise = event.get("exercise")
        completion = event["completion_us"]
        candidates = [i for i, t in enumerate(stamps) if start <= t <= completion]
        if not candidates:
            warnings.append(f"{event_id}: no frames in cycle")
            continue
        last = candidates[-1]
        reference = bodies[last]
        midpoint = start + (completion - start) / 2
        reference_available = reference is not None and completion - stamps[last] <= 100_000
        contiguous = []
        for i in reversed(candidates[:-1]) if reference_available else []:
            if bodies[i] is None or stamps[i + 1] - stamps[i] > 350_000:
                break
            contiguous.append(i)
        scores = []
        assert reference is not None
        for i in contiguous:
            points, confidence = bodies[i]
            weights = np.minimum(confidence, reference[1])[5:]
            weights = np.where(weights >= MIN_SCORE, weights, 0)
            distance = float(np.sqrt(np.sum(np.sum((points[5:] - reference[0][5:]) ** 2, axis=1) * weights)
                                     / max(weights.sum(), 1)))
            timing = max(0.0, 1 - abs(stamps[i] - midpoint) / max(1, (completion - start) / 2))
            scores.append((distance, timing, i))
        if not scores or max(s[0] for s in scores) < .1:
            before_return = candidates[:-1]
            if not before_return:
                warnings.append(f"{event_id}: cycle too short")
                continue
            opposite = min(before_return, key=lambda i: abs(stamps[i] - midpoint))
            heuristic = "cadence-midpoint-v1"
            warnings.append(f"{event_id}: timing estimate; keypoint evidence weak")
        else:
            maximum = max(s[0] for s in scores)
            opposite = max(scores, key=lambda s: .85 * s[0] / maximum + .15 * s[1])[2]
            heuristic = "cadence-motion-v2"
        if not valid[opposite] or not valid[last]:
            warnings.append(f"{event_id}: missing landmark at pair")
            continue
        pairs.append({"event_id": event_id, "exercise": exercise, "opposite_us": int(stamps[opposite]),
                      "return_us": int(stamps[last]), "opposite_index": int(opposite),
                      "return_index": int(last), "heuristic": heuristic,
                      "cycle_start_us": int(start), "completion_us": int(completion)})
    return pairs, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-unlabeled", action="store_true",
                        help="also build pose arrays for sources without a performance run (needs extractor npz)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    poses_dir = args.out / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads((args.store / "splits.json").read_text())["assignments"]
    sources = {}
    for path in sorted((args.store / "sources").glob("*.json")):
        source = json.loads(path.read_text())
        sources[source["id"]] = source

    index, pose_provenance = {}, {}
    for source_id, source in sorted(sources.items()):
        extracted = load_extracted(args.out, source_id)
        derived_path, rows, image_size = load_derived_performance(args.store, source_id)
        if extracted is not None and (derived_path is None or extracted[4].stat().st_mtime > derived_path.stat().st_mtime):
            stamps, xy, score, valid, path = extracted
            provenance = {"kind": "extractor", "path": str(path)}
        elif derived_path is not None:
            stamps, xy, score, valid = rows_to_arrays(rows, image_size)
            provenance = {"kind": "derived", "path": str(derived_path), "image_size": image_size}
        else:
            if args.include_unlabeled:
                log(f"  {source_id[:8]}: no performance poses, no extractor npz; skipped")
            continue
        annotation = latest_annotation(args.store, source_id)
        pairs, warnings = ([], [])
        if annotation:
            pairs, warnings = derive_pairs(annotation, stamps, xy, score, valid)
        np.savez_compressed(poses_dir / (source_id + ".npz"), ts=stamps, xy=xy, score=score, valid=valid)
        entry = {
            "source_id": source_id,
            "exercise": source.get("exercise"),
            "class_name": EXERCISE_CLASS.get(source.get("exercise") or "", None),
            "split": splits.get(source_id, {}).get("category", source.get("split")),
            "duration_us": source.get("duration_us"),
            "training_eligible": source.get("training_eligible"),
            "training_protected": source.get("training_protected"),
            "annotation_revision": annotation.get("revision") if annotation else None,
            "annotation_file": annotation.get("_file") if annotation else None,
            "moves_review": (annotation or {}).get("reviews", {}).get("moves"),
            "source_sha256": source.get("sha256"),
            "pose": provenance,
            "frames": int(len(stamps)),
            "valid_frames": int(valid.sum()),
            "pairs": len(pairs),
            "warnings": warnings,
        }
        index[source_id] = entry
        pose_provenance[source_id] = provenance["kind"]
        (poses_dir / (source_id + ".pairs.json")).write_text(json.dumps(pairs, indent=1))
        log(f"  {source_id[:8]} {entry['exercise']:14} split={entry['split']:10} frames={entry['frames']:5} "
            f"valid={entry['valid_frames']:5} pairs={len(pairs):3} ({provenance['kind']})")
        for warning in warnings[:3]:
            log(f"      warning: {warning}")

    (args.out / "sources.json").write_text(json.dumps(index, indent=1))
    all_pairs = {sid: json.loads((poses_dir / (sid + ".pairs.json")).read_text())
                 for sid in index if (poses_dir / (sid + ".pairs.json")).exists()}
    (args.out / "pairs.json").write_text(json.dumps(all_pairs, indent=1))
    total_pairs = sum(len(v) for v in all_pairs.values())
    log(f"own features done: {len(index)} sources, {total_pairs} salient pairs -> {args.out}")


if __name__ == "__main__":
    main()
