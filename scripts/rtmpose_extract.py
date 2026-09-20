#!/usr/bin/env python
"""RTMPose (native 17-joint) feature extraction for RepCount_pose and own videos.

Runs with rtmlib 0.0.16 + onnxruntime 1.22.1 (morning-forge tool venv is known good).
All outputs are resumable: an existing per-video npz is skipped.

Examples:
  python scripts/rtmpose_extract.py repcount-train
  python scripts/rtmpose_extract.py repcount-test            # ~5.4h on M3/CoreML
  python scripts/rtmpose_extract.py video --video path.mp4 --fps 15 --out data_rtmpose/own/videos
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtmpose_assets import COCO_JOINTS, HIPS, SHOULDERS, Detector, model_paths, fingerprint  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REPCOUNT = Path("/Volumes/Vibefare NAS/Software Projects/wake-strong/datasets/RepCount_pose")
DEFAULT_OUT = REPO / "data_rtmpose" / "repcount"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def video_frames(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return capture, fps, width, height, count


def extract_video(detector, path, wanted=None, stride=1):
    """Yield (frame_index, timestamp_us, xy, score, detected) for selected frames."""
    import cv2
    capture, fps, width, height, count = video_frames(path)
    wanted_set = None if wanted is None else set(int(i) for i in wanted)
    index = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if wanted_set is not None and index not in wanted_set:
                continue
            if index % stride:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector(rgb)
            if result is None:
                xy = np.zeros((17, 2), np.float32)
                score = np.zeros(17, np.float32)
                detected = False
            else:
                xy, score = result
                detected = True
            yield index, int(round(index / fps * 1e6)), xy, score, detected
    finally:
        capture.release()
    if wanted_set is not None and count:
        missing = sorted(i for i in wanted_set if i >= index + 1)
        if missing:
            log(f"  warning: {len(missing)} requested frames beyond end of {Path(path).name}: {missing[:5]}")


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_pose_train(csv_path):
    """Return {video: (exercise, salient1_indices, salient2_indices)}."""
    videos = {}
    with open(csv_path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 4 or row[1] == "type":
                continue
            labels = [value for value in row[3:] if value not in ("", "nan")]
            indices = [int(float(value)) for value in labels]
            videos[row[2]] = (row[1], indices[::2], indices[1::2])
    return videos


def shard(sequence, shard, shards):
    if shards <= 1:
        return sequence
    return [item for index, item in enumerate(sequence) if index % shards == shard]


def run_repcount_train(args, detector, provenance):
    video_dir = Path(args.root) / "video" / "train"
    videos = load_pose_train(Path(args.root) / "annotation" / "pose_train.csv")
    out_dir = Path(args.out) / "train_salient"
    items = shard(sorted(videos.items()), args.shard, args.shards)
    records = []
    done = 0
    started = time.time()
    for position, (name, (exercise, s1, s2)) in enumerate(items, 1):
        if args.limit and position > args.limit:
            break
        target = out_dir / (Path(name).stem + ".npz")
        wanted = sorted(set(s1) | set(s2))
        if target.exists():
            records.append({"video": name, "exercise": exercise, "path": str(target.relative_to(Path(args.out))),
                            "salient1": s1, "salient2": s2, "reused": True})
            done += 1
            continue
        path = video_dir / name
        frames, stamps, xys, scores, detected = [], [], [], [], []
        actual = set()
        for index, stamp, xy, score, ok in extract_video(detector, path, wanted=wanted):
            frames.append(index)
            stamps.append(stamp)
            xys.append(xy)
            scores.append(score)
            detected.append(ok)
            actual.add(index)
        missing = [i for i in wanted if i not in actual]
        if missing:
            log(f"  warning: {name}: missing frames {missing[:5]}")
        save_npz(target, {
            "frame": np.asarray(frames, np.int32),
            "timestamp_us": np.asarray(stamps, np.int64),
            "xy": np.asarray(xys, np.float32).reshape(-1, 17, 2),
            "score": np.asarray(scores, np.float32).reshape(-1, 17),
            "detected": np.asarray(detected, bool),
        })
        records.append({"video": name, "exercise": exercise, "path": str(target.relative_to(Path(args.out))),
                        "salient1": s1, "salient2": s2, "reused": False})
        done += 1
        if position % 25 == 0 or position == len(videos):
            elapsed = time.time() - started
            log(f"  train {position}/{len(videos)} videos ({done} saved) {elapsed/60:.1f} min")
    meta = {"mode": "repcount-train", "root": str(args.root), "videos": records,
            "provenance": provenance, "complete": True, "expected": len(videos),
            "shard": args.shard, "shards": args.shards}
    (Path(args.out) / ("train_manifest_shard" + str(args.shard) + ".json" if args.shards > 1
                       else "train_manifest.json")).write_text(json.dumps(meta, indent=1))
    log(f"repcount-train done: {len(records)} videos -> {out_dir}")


def run_repcount_test(args, detector, provenance):
    video_dir = Path(args.root) / "video" / "test"
    out_dir = Path(args.out) / "test"
    names = shard(sorted(p.name for p in video_dir.glob("*.mp4")), args.shard, args.shards)
    if args.limit:
        names = names[:args.limit]
    started = time.time()
    total_frames = 0
    records = []
    for position, name in enumerate(names, 1):
        target = out_dir / (Path(name).stem + ".npz")
        if target.exists() and not args.force:
            with np.load(target) as data:
                count = int(data["frame"].shape[0])
            records.append({"video": name, "path": str(target.relative_to(Path(args.out))), "frames": count, "reused": True})
            total_frames += count
            continue
        path = video_dir / name
        frames, stamps, xys, scores, detected = [], [], [], [], []
        for index, stamp, xy, score, ok in extract_video(detector, path, stride=args.stride):
            frames.append(index)
            stamps.append(stamp)
            xys.append(xy)
            scores.append(score)
            detected.append(ok)
        save_npz(target, {
            "frame": np.asarray(frames, np.int32),
            "timestamp_us": np.asarray(stamps, np.int64),
            "xy": np.asarray(xys, np.float32).reshape(-1, 17, 2),
            "score": np.asarray(scores, np.float32).reshape(-1, 17),
            "detected": np.asarray(detected, bool),
        })
        total_frames += len(frames)
        records.append({"video": name, "path": str(target.relative_to(Path(args.out))), "frames": len(frames), "reused": False})
        elapsed = time.time() - started
        done = position
        rate = elapsed / max(done, 1)
        log(f"  test {done}/{len(names)} {name} frames={len(frames)} "
            f"elapsed={elapsed/60:.1f}m eta={(len(names)-done)*rate/60:.1f}m total_frames={total_frames}")
    meta = {"mode": "repcount-test", "root": str(args.root), "stride": args.stride,
            "videos": records, "total_frames": total_frames, "provenance": provenance}
    (Path(args.out) / "test_manifest.json").write_text(json.dumps(meta, indent=1))
    log(f"repcount-test done: {len(records)} videos / {total_frames} frames")


def run_video(args, detector, provenance):
    out_dir = Path(args.out)
    path = Path(args.video)
    capture, fps, width, height, count = video_frames(path)
    capture.release()
    stride = max(1, int(round(fps / args.fps))) if args.fps else args.stride
    frames, stamps, xys, scores, detected = [], [], [], [], []
    for index, stamp, xy, score, ok in extract_video(detector, path, stride=stride):
        frames.append(index)
        stamps.append(stamp)
        xys.append(xy)
        scores.append(score)
        detected.append(ok)
        if index % 150 == 0:
            log(f"  {path.name}: frame {index} ({len(frames)} kept) detected={ok}")
    stem = args.name or path.stem
    target = out_dir / (stem + ".npz")
    save_npz(target, {
        "frame": np.asarray(frames, np.int32),
        "timestamp_us": np.asarray(stamps, np.int64),
        "xy": np.asarray(xys, np.float32).reshape(-1, 17, 2),
        "score": np.asarray(scores, np.float32).reshape(-1, 17),
        "detected": np.asarray(detected, bool),
    })
    (out_dir / (stem + ".meta.json")).write_text(json.dumps({
        "video": str(path), "fps": fps, "width": width, "height": height, "source_frames": count,
        "stride": stride, "kept_frames": len(frames), "provenance": provenance}, indent=1))
    log(f"video done: {target} ({len(frames)} frames, stride {stride})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--variant", default="rtmpose-performance")
    shared.add_argument("--cpu", action="store_true", help="disable CoreML EP")
    shared.add_argument("--threads", type=int, default=4)
    shared.add_argument("--detect-stride", type=int, default=3,
                        help="run the person detector every N frames, carrying boxes in between (1 = every frame)")

    train = sub.add_parser("repcount-train", parents=[shared])
    train.add_argument("--root", type=Path, default=DEFAULT_REPCOUNT)
    train.add_argument("--out", type=Path, default=DEFAULT_OUT)
    train.add_argument("--limit", type=int, default=0)
    train.add_argument("--shard", type=int, default=0)
    train.add_argument("--shards", type=int, default=1)

    test = sub.add_parser("repcount-test", parents=[shared])
    test.add_argument("--root", type=Path, default=DEFAULT_REPCOUNT)
    test.add_argument("--out", type=Path, default=DEFAULT_OUT)
    test.add_argument("--stride", type=int, default=1)
    test.add_argument("--limit", type=int, default=0)
    test.add_argument("--force", action="store_true")
    test.add_argument("--shard", type=int, default=0)
    test.add_argument("--shards", type=int, default=1)

    video = sub.add_parser("video", parents=[shared])
    video.add_argument("--video", type=Path, required=True)
    video.add_argument("--out", type=Path, required=True)
    video.add_argument("--name", default="")
    video.add_argument("--fps", type=float, default=0, help="target sampling fps (0 = native)")
    video.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log(f"loading {args.variant} (coreml={not args.cpu}, detect_stride={args.detect_stride})")
    detector = Detector(args.variant, coreml=not args.cpu, threads=args.threads,
                        detect_stride=args.detect_stride)
    provenance = {"detector": detector.provenance, "assets": {k: str(v) for k, v in model_paths(args.variant).items()},
                  "fingerprint": fingerprint(args.variant)}
    (args.out / "extractor_provenance.json").write_text(json.dumps(provenance, indent=1))
    log(json.dumps(detector.provenance["models"], indent=1))
    if args.mode == "repcount-train":
        run_repcount_train(args, detector, provenance)
    elif args.mode == "repcount-test":
        run_repcount_test(args, detector, provenance)
    else:
        run_video(args, detector, provenance)


if __name__ == "__main__":
    main()
