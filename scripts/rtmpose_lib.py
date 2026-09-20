"""Native-17 PoseRAC model and repetition decoding shared by train/eval."""
import numpy as np
import torch
from torch import nn

CLASSES = ["front_raise", "pull_up", "squat", "bench_pressing", "jump_jack",
           "situp", "push_up", "pommelhorse", "jab"]
DIM = 34
HEADS = 2
ENCODER_LAYERS = 6
NUM_CLASSES = len(CLASSES)


class PoseRACNative(nn.Module):
    """PoseRAC encoder adapted to native RTMPose 17-joint input (34-dim)."""

    def __init__(self, dim=DIM, heads=HEADS, enc_layer=ENCODER_LAYERS, num_classes=NUM_CLASSES):
        super().__init__()
        self.dim = dim
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True), num_layers=enc_layer)
        self.fc1 = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.dim)
        x = self.transformer_encoder(x)
        x = x.view(-1, self.dim)
        return self.fc1(x)

    def embeddings(self, x):
        x = x.view(-1, 1, self.dim)
        x = self.transformer_encoder(x)
        return x.view(-1, self.dim)


class ActionTrigger:
    """Upstream two-threshold trigger for salient-phase counting."""

    def __init__(self, enter_threshold=0.78, exit_threshold=0.4):
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.pose_entered = False

    def __call__(self, score):
        triggered = False
        if not self.pose_entered:
            self.pose_entered = score > self.enter_threshold
            return triggered
        if score < self.exit_threshold:
            self.pose_entered = False
            triggered = True
        return triggered


def count_repetitions(probabilities, mask, enter=0.78, exit=0.4, momentum=0.4):
    """Mirror upstream eval.py salient1/salient2 alternation counting.

    `probabilities` is P(salient1) per frame; `mask` marks usable frames.
    """
    trigger1 = ActionTrigger(enter, exit)
    trigger2 = ActionTrigger(enter, exit)
    classify_prob = 0.5
    count = 0
    curr = "holder"
    initial = "holder"
    for probability, usable in zip(probabilities, mask):
        if not usable:
            continue
        classify_prob = float(probability) * (1.0 - momentum) + momentum * classify_prob
        salient1 = trigger1(classify_prob)
        salient2 = trigger2(1.0 - classify_prob)
        if initial == "holder":
            if salient1:
                initial = "salient1"
            elif salient2:
                initial = "salient2"
        if initial == "salient1":
            if curr == "salient1" and salient2:
                count += 1
        else:
            if curr == "salient2" and salient1:
                count += 1
        if salient1:
            curr = "salient1"
        elif salient2:
            curr = "salient2"
    return count


def repetition_events(probabilities, timestamps, mask, enter=0.78, exit=0.4, momentum=0.4):
    """Return completion timestamps (upstream trigger semantics)."""
    trigger1 = ActionTrigger(enter, exit)
    trigger2 = ActionTrigger(enter, exit)
    classify_prob = 0.5
    curr = "holder"
    initial = "holder"
    events = []
    for probability, timestamp, usable in zip(probabilities, timestamps, mask):
        if not usable:
            continue
        classify_prob = float(probability) * (1.0 - momentum) + momentum * classify_prob
        salient1 = trigger1(classify_prob)
        salient2 = trigger2(1.0 - classify_prob)
        if initial == "holder":
            if salient1:
                initial = "salient1"
            elif salient2:
                initial = "salient2"
        if initial == "salient1":
            if curr == "salient1" and salient2:
                events.append(int(timestamp))
        else:
            if curr == "salient2" and salient1:
                events.append(int(timestamp))
        if salient1:
            curr = "salient1"
        elif salient2:
            curr = "salient2"
    return events


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = PoseRACNative(dim=config.get("dim", DIM), heads=config.get("heads", HEADS),
                          enc_layer=config.get("enc_layer", ENCODER_LAYERS),
                          num_classes=config.get("num_classes", NUM_CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, config
