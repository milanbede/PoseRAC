"""Read-only candidate comparison on a saved Data Review Status run.

Requires ws_data on PYTHONPATH. Does not extract poses, alter annotations, or
write into the live run cache. Test-mode manifests are rejected.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from rtmpose_lib import CLASSES, load_model


def main():
    from ws_data.pipeline_metrics import score
    from ws_data.pipeline_runner import exercise_schedule
    from ws_data.poserac import NATIVE_ADAPTER, decode_native, vectors

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "job", "cache", "model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["request"].get("evaluation_mode") != "status":
        raise ValueError("Only train/validation Status runs may be compared")
    job = json.loads(args.job.read_text())
    net, config = load_model(args.model)
    torch.set_num_threads(2)
    keys = ("matched", "missed", "extra", "false_counts")
    totals = defaultdict(lambda: {"before": dict.fromkeys(keys, 0), "after": dict.fromkeys(keys, 0)})
    rows = []
    for sid, prior in job["results"].items():
        frozen = manifest["inputs"][sid]
        if frozen["source"]["split"] not in {"train", "validation"}:
            raise ValueError("Test/unassigned source in Status manifest")
        if "artifact_fingerprint" not in prior:
            continue
        artifact = args.cache / prior["artifact_fingerprint"] / "inference.json"
        data = json.loads(artifact.read_text())
        predictions = []
        for span in exercise_schedule(frozen):
            exercise = span["exercise"]
            name = "jump_jack" if exercise == "jumping_jack" else exercise
            frames = [f for f in data["frames"] if span["start_us"] <= f["timestamp_us"] < span["end_us"]]
            values, valid = vectors(frames, frozen["pipeline"]["config"]["pose_model"], NATIVE_ADAPTER)
            with torch.inference_mode():
                probability = torch.cat([torch.sigmoid(net(torch.from_numpy(values[i:i+128])))
                                         for i in range(0, len(values), 128)])[:, CLASSES.index(name)].numpy()
            events = decode_native([f["timestamp_us"] for f in frames], probability, valid, exercise,
                                   enter=.78, exit=.4, momentum=.4)
            predictions.extend(e.model_dump() for e in events)
        current = score(frozen, data["frames"], predictions, data["image_size"], 0, "cached-pose-replay")
        exercise = frozen["source"]["exercise"]
        row = {"source_id": sid, "filename": frozen["source"]["filename"], "exercise": exercise,
               "split": frozen["source"]["split"], "before": {k: prior[k] for k in keys},
               "after": {k: current[k] for k in keys}, "predictions": len(predictions),
               "artifact_fingerprint": prior["artifact_fingerprint"]}
        for side in ("before", "after"):
            for key in keys:
                totals[exercise][side][key] += row[side][key]
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"config": config, "evaluation": "training-validation cached-pose diagnostic, not independent test",
              "decoder": {"enter": .78, "exit": .4, "momentum": .4}, "summary": dict(totals), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
