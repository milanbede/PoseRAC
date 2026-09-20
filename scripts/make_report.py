#!/usr/bin/env python
"""Merge evaluation artifacts into reports/rtmpose_<date>.{json,md}."""
import argparse
import datetime
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(path):
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return None


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-meta", type=Path, default=REPO / "data_rtmpose" / "dataset_meta.json")
    parser.add_argument("--final-model", type=Path, default=REPO / "runs" / "final" / "model.pt")
    parser.add_argument("--final-config", type=Path, default=REPO / "runs" / "final" / "config.json")
    parser.add_argument("--holdout", type=Path, default=REPO / "reports" / "holdout_eval.json")
    parser.add_argument("--repcount", type=Path, default=REPO / "reports" / "repcount_test.json")
    parser.add_argument("--loro", type=Path, default=REPO / "reports" / "loro.json")
    parser.add_argument("--own", type=Path, default=REPO / "reports" / "own_test.json")
    parser.add_argument("--extract-provenance", type=Path,
                        default=REPO / "data_rtmpose" / "repcount" / "extractor_provenance.json")
    parser.add_argument("--out-dir", type=Path, default=REPO / "reports")
    args = parser.parse_args()

    meta = load(args.dataset_meta)
    holdout = load(args.holdout)
    repcount = load(args.repcount)
    loro = load(args.loro)
    own = load(args.own)
    config = load(args.final_config)
    provenance = load(args.extract_provenance)

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": meta,
        "extraction_provenance": provenance,
        "model_config": config,
        "repcount_test": repcount["summary"] if repcount else None,
        "repcount_holdout": holdout["summary"] if holdout else None,
        "own_loro": loro["summary"] if loro else None,
        "own_test": None,
    }
    own_rows = [r for r in (own or {}).get("rows", []) if r.get("split") == "test"]
    if own:
        rows = own_rows
        report["own_test"] = {
            "recordings": len(rows),
            "predicted_reps": sum(r["predicted_count"] for r in rows),
            "reference_reps": sum(r["reference_count"] for r in rows),
            "f1": sum(r["timing"]["f1"] for r in rows) / len(rows) if rows else None,
            "count_errors": {r["source_id"][:8]: r["count_error"] for r in rows},
            "rows": rows,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    json_path = args.out_dir / f"rtmpose_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=1))

    lines = [f"# PoseRAC native-17 RTMPose training report ({stamp})", ""]
    if meta:
        lines += [
            "## Dataset",
            f"- classes: {', '.join(meta['classes'])}",
            f"- examples: {meta['examples']} {meta['parts']}",
            f"- positive salient poses per class: {meta['positive_salient1_per_class']}",
            f"- RepCount holdout videos: {len(meta['holdout_videos'])} ({meta['holdout_fraction']:.0%})",
            f"- dataset sha256: `{meta['sha256'][:16]}...`",
            "",
        ]
    if own:
        lines += ["## Own test split (held-out recordings only)", "| recording | exercise | predicted | reference | error | F1 |",
                  "|---|---|---|---|---|---|"]
        for row in own_rows:
            lines.append(f"| {row['source_id'][:8]} | {row['exercise']} | {row['predicted_count']} | "
                         f"{row['reference_count']} | {row['count_error']:+d} | {row['timing']['f1']:.2f} |")
        lines.append("")
    if loro:
        lines += ["## Own leave-one-recording-out", "| exercise | recordings | predicted/ref reps | MAE | OBO | F1 | precision | recall |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, stats in sorted(loro.get("summary", {}).items()):
            lines.append(f"| {name} | {stats['recordings']} | {stats['predicted_reps']}/{stats['reference_reps']} | "
                         f"{stats['mae']:.2f} | {stats['obo']:.2f} | {stats['f1']:.2f} | "
                         f"{stats['precision']:.2f} | {stats['recall']:.2f} |")
        lines.append("")
    if repcount:
        summary = repcount["summary"]
        lines += ["## RepCount_pose test", f"- videos: {summary['videos']}",
                  f"- oracle-class MAE / OBO: {fmt(summary['oracle_mae'])} / {fmt(summary['oracle_obo'])}",
                  f"- known-exercise (GT type) MAE / OBO: {fmt(summary.get('gt_class_mae'))} / {fmt(summary.get('gt_class_obo'))}",
                  f"- predicted-class MAE / OBO: {fmt(summary['predicted_class_mae'])} / {fmt(summary['predicted_class_obo'])}",
                  f"- oracle class matches GT type: {fmt(summary.get('oracle_class_matches_gt_type'))}",
                  ""]
    else:
        lines += ["## RepCount_pose test", "Not yet evaluated (RTMPose test extraction pending).", ""]
    lines += [
        "## Notes",
        "- Trained from scratch on native RTMPose COCO-17 features (34-dim); the official 99-dim MediaPipe",
        "  checkpoint cannot be warm-started into this input space.",
        "- Own data is weak-phase supervision derived from reviewed valid repetitions, oversampled 4x.",
        "- The 40-rep push-up benchmark (76db062c) was unrecoverable and is excluded.",
        "- RepCount test reports the upstream oracle-class protocol, the known-exercise variant (GT type), and",
        "  the predicted-class variant.",
        "- Own decode thresholds were tuned only on LORO validation folds; the own test recordings are untouched.",
        "- Two test clips reviewed with zero valid reps (1648b896 jab, 2e319d64 jumping_jack) contain 9 and 10",
        "  invalid_attempt events; counts there measure movement cycles, not form validity. In-sample train",
        "  recordings are omitted from the own test table.",
        "",
        f"Artifacts: `{json_path}`",
    ]
    md_path = args.out_dir / f"rtmpose_{stamp}.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
