"""Shared native-17 feature normalization for RTMPose pose sequences."""
import numpy as np

SHOULDERS = (5, 6)
HIPS = (11, 12)
BODY = tuple(range(5, 17))
NUM_JOINTS = 17
NUM_FEATURES = NUM_JOINTS * 2
MIN_SCORE = 0.3


def normalize_frame(xy, score):
    """Root-relative, torso-scaled 34-dim feature for a single frame.

    Returns (feature, valid). Invalid frames are all-zero and must be masked out.
    """
    xy = np.asarray(xy, np.float32)
    score = np.asarray(score, np.float32)
    if xy.shape != (NUM_JOINTS, 2) or score.shape != (NUM_JOINTS,):
        return np.zeros(NUM_FEATURES, np.float32), False
    if not np.isfinite(xy).all() or not np.isfinite(score).all():
        return np.zeros(NUM_FEATURES, np.float32), False
    if score[list(SHOULDERS) + list(HIPS)].min() < MIN_SCORE:
        return np.zeros(NUM_FEATURES, np.float32), False
    hip = (xy[11] + xy[12]) / 2.0
    shoulder = (xy[5] + xy[6]) / 2.0
    scale = float(np.linalg.norm(shoulder - hip))
    if scale < 1e-5:
        return np.zeros(NUM_FEATURES, np.float32), False
    return ((xy - hip) / scale).reshape(-1).astype(np.float32), True


def normalize_sequence(xy, score, valid=None):
    """Return (features (N,34), mask (N,)) for a sequence."""
    xy = np.asarray(xy, np.float32).reshape(-1, NUM_JOINTS, 2)
    score = np.asarray(score, np.float32).reshape(-1, NUM_JOINTS)
    n = len(xy)
    features = np.zeros((n, NUM_FEATURES), np.float32)
    mask = np.zeros(n, bool)
    if valid is not None:
        valid = np.asarray(valid, bool).reshape(-1)
    for i in range(n):
        if valid is not None and not valid[i]:
            continue
        features[i], mask[i] = normalize_frame(xy[i], score[i])
    return features, mask
