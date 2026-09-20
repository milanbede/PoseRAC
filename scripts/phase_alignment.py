"""Align salient phases from saved geometry, without changing source annotations."""
from collections import defaultdict

import numpy as np

POLICY = "geometry-open-bent-positive-v1"


def phase_geometry(features, exercise):
    points = np.asarray(features).reshape(-1, 17, 2)
    if exercise == "jump_jack":
        # Positive means wrists above shoulders; coordinates are torso-scaled.
        return (points[:, [5, 6], 1] - points[:, [9, 10], 1]).mean(axis=1), .25
    angles = []
    for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
        a, b = points[:, shoulder] - points[:, elbow], points[:, wrist] - points[:, elbow]
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine = np.divide((a * b).sum(axis=1), denominator,
                           out=np.full(len(points), np.nan), where=denominator > 1e-8)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    # Positive means bent elbows. Both sides must have usable geometry.
    return -np.mean(angles, axis=0), 15.


def align_examples(examples):
    """Choose polarity independently per recording; reject unclear phase groups.

    Tuple schema is shared with build_dataset: X, Y, class index/name,
    source, frame, partition. Only the selected class label may change.
    """
    aligned = list(examples)
    groups = defaultdict(list)
    for index, example in enumerate(examples):
        if example[3] in {"jump_jack", "push_up"}:
            groups[(example[3], example[4], example[6])].append(index)
    rejected, report = set(), []
    for (exercise, source, part), indices in sorted(groups.items()):
        scores, margin = phase_geometry([examples[i][0] for i in indices], exercise)
        targets = np.asarray([examples[i][1][examples[i][2]] for i in indices])
        positive, negative = scores[targets == 1], scores[targets == 0]
        delta, agreement = None, 0.
        if len(positive) and len(negative) and np.isfinite(scores).all():
            delta = float(np.median(positive) - np.median(negative))
            direction = 1 if delta > 0 else -1
            agreement = float(np.mean(direction * (positive[:, None] - negative) > margin))
        if delta is None or abs(delta) <= margin or agreement < .8:
            decision = "excluded"
            rejected.update(indices)
        elif delta < 0:
            decision = "flipped"
            for i in indices:
                feature, label, cls, *rest = examples[i]
                label = label.copy()
                label[cls] = 1 - label[cls]
                aligned[i] = (feature, label, cls, *rest)
        else:
            decision = "kept"
        report.append({"exercise": exercise, "source": source, "part": part,
                       "decision": decision, "examples": len(indices),
                       "median_delta": delta, "agreement": agreement})
    return [e for i, e in enumerate(aligned) if i not in rejected], report
