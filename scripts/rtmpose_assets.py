"""Pinned RTMPose ONNX assets (mirrors morning-forge ws_data/rtmpose.py).

The own-data pose runs were produced with these exact assets, so extraction for
RepCount and re-extraction of own recordings must use the same files.
"""
import hashlib
import os
from pathlib import Path

import numpy as np

RELEASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"

DETECTORS = {
    "yolox_nano": ("yolox_nano_8xb8-300e_humanart-40f6f0d0", "1450966de24902b18aada1a78913d7efd8fc8dcd51bd4d0d5591476bd4a38821", (416, 416)),
    "yolox_tiny": ("yolox_tiny_8xb8-300e_humanart-6f3252f9", "ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250", (416, 416)),
    "yolox_s": ("yolox_s_8xb8-300e_humanart-3ef259a7", "332e09ea9696e3401049b6c5314851db020b6430f85da159e65f37034ab3aee8", (640, 640)),
    "yolox_m": ("yolox_m_8xb8-300e_humanart-c2c7a14a", "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23", (640, 640)),
    "yolox_l": ("yolox_l_8xb8-300e_humanart-ce1d7a62", "0e7a169bb6b2f439c318f727fd894b1405c30280baca676e6a5c7d1f8cb52876", (640, 640)),
    "yolox_x": ("yolox_x_8xb8-300e_humanart-a39d44ed", "8e9ea96a176bd48501eaaa77216e49ee30794d2f8ba80c7b9862beca4ea972da", (640, 640)),
}

POSE = {
    "s": {"256x192": ("rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504", "9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d", (192, 256))},
    "m": {"256x192": ("rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504", "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c", (192, 256))},
    "l": {
        "256x192": ("rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504", "cff059fd58a2c0d5fabaddcd66a96abcfb327563bcb0149ea59c9de4a8990fe2", (192, 256)),
        "384x288": ("rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504", "62425affc6b9edba8f8a65f55f5e278fec5d905d0c90578e0f8503bdf3954587", (288, 384)),
    },
    "x": {"384x288": ("rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629", "df0c0fa91e9870b1515dcaff741fd76cc753dcfb12786862f61b243bae81cd52", (288, 384))},
}

DEFAULTS = {
    "lightweight": ("s", "256x192", "yolox_tiny"),
    "balanced": ("m", "256x192", "yolox_m"),
    "l": ("l", "384x288", "yolox_x"),
    "performance": ("x", "384x288", "yolox_x"),
}

# COCO-17 joint order.
COCO_JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
               "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
               "left_knee", "right_knee", "left_ankle", "right_ankle"]
SHOULDERS = (5, 6)
HIPS = (11, 12)

DEFAULT_VARIANT = "rtmpose-performance"
MODEL_HOME = Path(os.environ.get("WS_DATA_HOME", Path.home() / ".local/share/ws-data")) / "models" / "rtmpose"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parts(variant):
    if not variant.startswith("rtmpose-"):
        raise ValueError("Unknown RTMPose model")
    key = variant[len("rtmpose-"):]
    if key in DEFAULTS:
        return DEFAULTS[key]
    parts = key.rsplit("-", 2)
    if len(parts) != 3 or parts[0] not in POSE or parts[1] not in POSE[parts[0]] or parts[2] not in DETECTORS:
        raise ValueError("Unknown RTMPose model")
    return parts[0], parts[1], parts[2]


def configuration(variant):
    family, size, detector = _parts(variant)
    return {"det": DETECTORS[detector], "pose": POSE[family][size]}


def fingerprint(variant):
    import json
    return hashlib.sha256(json.dumps(configuration(variant), sort_keys=True).encode()).hexdigest()


def model_paths(variant):
    paths = {}
    for kind, (name, checksum, _) in configuration(variant).items():
        path = MODEL_HOME / kind / (name + ".onnx")
        if not path.is_file():
            raise ValueError(f"RTMPose asset missing: {path}")
        if digest(path) != checksum:
            raise ValueError(f"RTMPose asset hash mismatch: {path}")
        paths[kind] = path
    return paths


class Detector:
    """RTMPose person detector + body pose, CoreML-accelerated with CPU fallback.

    detect_stride amortizes the (expensive) person detector: boxes are carried for
    up to `detect_stride` frames, with a joint-confidence guard so carried boxes on
    an empty scene do not produce fabricated poses.
    """

    def __init__(self, variant=DEFAULT_VARIANT, coreml=True, threads=4, detect_stride=3, min_joint_score=0.3):
        try:
            import onnxruntime as ort
            from rtmlib import RTMPose, YOLOX
            from rtmlib.tools.base import BaseTool
        except ImportError as error:  # pragma: no cover
            raise SystemExit("Install extraction dependencies: rtmlib==0.0.16 onnxruntime==1.22.1") from error
        ort.disable_telemetry_events()
        ort.set_default_logger_severity(4)

        providers = (["CoreMLExecutionProvider", "CPUExecutionProvider"] if coreml
                     else ["CPUExecutionProvider"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 4

        class BoundBase(BaseTool):
            def __init__(self, onnx_model, model_input_size=None, mean=None, std=None, backend="onnxruntime", device="cpu"):
                self.onnx_model = onnx_model
                self.model_input_size = model_input_size
                self.mean, self.std = mean, std
                self.backend, self.device = backend, device
                self.session = ort.InferenceSession(onnx_model, sess_options=options, providers=providers)

        class BoundDetector(YOLOX, BoundBase):
            pass

        class BoundPose(RTMPose, BoundBase):
            pass

        config = configuration(variant)
        paths = model_paths(variant)
        self.detector_model = BoundDetector(str(paths["det"]), model_input_size=config["det"][2])
        self.pose_model = BoundPose(str(paths["pose"]), model_input_size=config["pose"][2], to_openpose=False)
        self.detect_stride = max(1, int(detect_stride))
        self.min_joint_score = float(min_joint_score)
        self._frame = 0
        self._boxes = None
        self.provenance = {
            "provider": "rtmpose",
            "variant": variant,
            "model_sha256": fingerprint(variant),
            "models": {kind: {"filename": path.name, "sha256": digest(path)} for kind, path in paths.items()},
            "rtmlib_version": __import__("importlib.metadata", fromlist=["version"]).version("rtmlib"),
            "onnxruntime_version": ort.__version__,
            "execution_providers": providers,
            "detect_stride": self.detect_stride,
            "min_joint_score": self.min_joint_score,
            "native_skeleton": "coco-17",
            "confidence": "keypoint score",
        }

    def __call__(self, rgb):
        """Return (xy, score) for the highest-scoring person or None."""
        import numpy as np
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        carried = self._boxes is not None and self._frame % self.detect_stride
        self._frame += 1
        if not carried:
            try:
                boxes = self.detector_model(bgr)
            except Exception as error:  # CoreML rejects the detector's zero-length NMS output.
                if "has zero elements" not in str(error):
                    raise
                self._boxes = None
                return None
            if len(boxes) == 0:
                self._boxes = None
                return None
            self._boxes = boxes
        keypoints, scores = self.pose_model(bgr, bboxes=self._boxes)
        if len(keypoints) == 0:
            return None
        scores = np.asarray(scores, dtype=np.float32)
        best = int(np.argmax(scores.mean(axis=1)))
        xy = np.asarray(keypoints[best], dtype=np.float32)
        score = scores[best]
        if score[list(SHOULDERS) + list(HIPS)].min() < self.min_joint_score:
            self._boxes = None
            return None
        return xy, score
