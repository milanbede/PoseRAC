#!/usr/bin/env python
"""Export the native-17 checkpoint as a ws-data PoseRAC candidate.

Writes ~/.local/share/ws-data/models/poserac/candidates/<id>/{weights.pth,manifest.json}.
The review tool currently builds a 99-dim MediaPipe-adapter network, so consuming
this candidate requires a native-34 adapter branch in
morning-forge ws_data/poserac.py (adapter id below); the manifest records
everything that branch needs.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from rtmpose_lib import CLASSES

REPO = Path(__file__).resolve().parent.parent
DEFAULT_HOME = Path.home() / ".local/share/ws-data" / "models" / "poserac"
DEFAULT_OFFICIAL = "469590b611bde3595eaf163b517263da634e2096"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "runs" / "final" / "model.pt")
    parser.add_argument("--id", default="rtmpose-native-9cls")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--report", type=Path, default=REPO / "reports" / "rtmpose_2026-09-20.json")
    args = parser.parse_args()

    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    target = args.home / "candidates" / args.id
    target.mkdir(parents=True, exist_ok=True)
    weights = target / "weights.pth"
    shutil.copyfile(args.checkpoint, weights)

    validation = None
    if args.report.exists():
        validation = json.loads(args.report.read_text())
    manifest = {
        "id": args.id,
        "sha256": digest(weights),
        "actions": CLASSES,
        "adapter": "coco17-native-34-v1",
        "dim": config.get("dim", 34),
        "heads": config.get("heads", 2),
        "enc_layer": config.get("enc_layer", 6),
        "pose_model": "rtmpose-performance",
        "base_sha256": None,
        "weak_supervision": False,
        "context_policy": "per-frame-v1",
        "seed": config.get("seed", 42),
        "epochs": config.get("fixed_epochs", 60),
        "learning_rate": config.get("lr", 1e-3),
        "experimental": True,
        "training": {"dataset_sha256": config.get("dataset_sha256"), "own_weight": config.get("own_weight")},
        "validation": validation,
        "source_checkpoint": str(args.checkpoint),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"exported candidate {args.id} -> {target}")


if __name__ == "__main__":
    main()
