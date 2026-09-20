#!/usr/bin/env python
"""Export the native-17 checkpoint as a ws-data PoseRAC candidate.

Writes ~/.local/share/ws-data/models/poserac/candidates/<id>/{weights.pth,manifest.json}
for the gm data-review tool (tool/dataset_review), which lists candidates under
the model dropdown. The manifest declares adapter coco17-native-34-v1 and the
action vocabulary the review tool uses (jumping_jack, not jump_jack).
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtmpose_assets import configuration  # noqa: E402
from rtmpose_lib import CLASSES  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_HOME = Path.home() / ".local/share/ws-data" / "models" / "poserac"
# Review-tool vocabulary: jump_jack -> jumping_jack, plus jab as the 9th output.
APP_ACTIONS = [name if name != "jump_jack" else "jumping_jack" for name in CLASSES]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_model_fingerprint(variant):
    """Match ws_data.rtmpose.fingerprint (encoded() formatting included)."""
    payload = json.dumps(configuration(variant), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256((payload + "\n").encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "runs" / "final" / "model.pt")
    parser.add_argument("--id", default="rtmpose-native-9cls")
    parser.add_argument("--pose-model", default="rtmpose-performance")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--report", type=Path, default=REPO / "reports" / "rtmpose_2026-09-20.json")
    parser.add_argument("--decode-tuning", type=Path, default=REPO / "reports" / "decode_tuning.json")
    args = parser.parse_args()

    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    target = args.home / "candidates" / args.id
    if target.exists():
        raise ValueError("Candidate already exists; choose a new id to preserve saved evaluations")
    target.mkdir(parents=True, exist_ok=True)
    weights = target / "weights.pth"
    temporary = weights.with_suffix(".tmp")
    torch.save(checkpoint["state_dict"], temporary)
    temporary.replace(weights)

    validation = json.loads(args.report.read_text()) if args.report.exists() else None
    decode_tuning = json.loads(args.decode_tuning.read_text())["params"] if args.decode_tuning.exists() else {}
    manifest = {
        "id": args.id,
        "sha256": digest(weights),
        "actions": APP_ACTIONS,
        "adapter": "coco17-native-34-v1",
        "dim": config.get("dim", 34),
        "heads": config.get("heads", 2),
        "enc_layer": config.get("enc_layer", 6),
        "batch_first": True,
        "pose_model": args.pose_model,
        "pose_model_sha256": pose_model_fingerprint(args.pose_model),
        "decode": decode_tuning,
        "base_sha256": None,
        "weak_supervision": False,
        "context_policy": "per-frame-v1",
        "seed": config.get("seed", 42),
        "epochs": config.get("fixed_epochs") or 60,
        "learning_rate": config.get("lr", 1e-3),
        "experimental": True,
        "training": {"dataset_sha256": config.get("dataset_sha256"), "own_weight": config.get("own_weight"),
                     "phase_policy": config.get("phase_policy"), "sampling_policy": config.get("sampling_policy")},
        "validation": validation,
        "source_checkpoint": str(args.checkpoint),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"exported candidate {args.id} -> {target}")
    print(f"  weights sha256: {manifest['sha256']}")
    print(f"  actions: {', '.join(manifest['actions'])}")


if __name__ == "__main__":
    main()
