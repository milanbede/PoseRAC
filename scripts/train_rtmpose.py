#!/usr/bin/env python
"""Train native-17 PoseRAC on RepCount_pose + own wake-strong data.

Modes:
  holdout   train on RepCount train + own train, early-stop on the RepCount
            video holdout (also used for LORO folds with --exclude)
  final     train on everything with a fixed epoch budget

The official 99-dim MediaPipe weights cannot be warm-started into the native
34-dim model, so this trains from scratch; RepCount is the broad pretraining set
and own data is oversampled.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtmpose_lib import CLASSES, DIM, ENCODER_LAYERS, HEADS, PoseRACNative  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OWN_WEIGHT = 4.0


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(path):
    data = np.load(path)
    return {key: data[key] for key in data.files}


def build_split(data, exclude=None, include_holdout=False):
    part = data["part"]
    source = data["source"]
    train_mask = np.isin(part, ["repcount_train", "own_train"])
    if exclude:
        train_mask &= ~(source == exclude)
    if include_holdout:
        train_mask |= part == "repcount_val"
        val_mask = np.zeros_like(train_mask)
    else:
        val_mask = part == "repcount_val"
    return train_mask, val_mask


def class_pos_weight(Y, train_mask):
    positives = Y[train_mask].sum(axis=0)
    total = int(train_mask.sum())
    weight = np.where(positives > 0, (total - positives) / np.maximum(positives, 1), 1.0)
    return torch.tensor(np.clip(weight, 1.0, 50.0), dtype=torch.float32)


def sampling_weights(part, train_mask, own_weight):
    return torch.tensor([own_weight if str(p) == "own_train" else 1.0
                         for p in part[train_mask]], dtype=torch.double)


def make_loader(X, Y, metric, part, train_mask, own_weight, batch_size, seed):
    indices = np.where(train_mask)[0]
    weights = sampling_weights(part, train_mask, own_weight)
    sampler = WeightedRandomSampler(weights, num_samples=len(indices), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    dataset = TensorDataset(torch.from_numpy(X[indices]), torch.from_numpy(Y[indices]),
                            torch.from_numpy(metric[indices]).long())
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def evaluate_loss(model, X, Y, mask, device, batch_size=512):
    model.eval()
    total, count = 0.0, 0
    loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for start in range(0, int(mask.sum()), batch_size):
            indices = np.where(mask)[0][start:start + batch_size]
            x = torch.from_numpy(X[indices]).to(device)
            y = torch.from_numpy(Y[indices]).to(device)
            loss = loss_fn(model(x), y)
            total += float(loss) * len(indices)
            count += len(indices)
    return total / max(count, 1)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_dataset(args.dataset)
    X, Y, metric = data["X"], data["Y"], data["metric"]
    train_mask, val_mask = build_split(data, exclude=args.exclude, include_holdout=args.include_holdout)
    log(f"train examples: {int(train_mask.sum())}, validation examples: {int(val_mask.sum())}")
    device = pick_device()
    model = PoseRACNative().to(device)
    pos_weight = class_pos_weight(Y, train_mask).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=6, mode="min")
    loader = make_loader(X, Y, metric, data["part"], train_mask, args.own_weight, args.batch_size, args.seed)

    miner = loss_metric = None
    if args.alpha > 0:
        from pytorch_metric_learning import losses, miners
        miner = miners.MultiSimilarityMiner()
        loss_metric = losses.TripletMarginLoss()

    best_loss, best_epoch, best_state, history = float("inf"), -1, None, []
    started = time.time()
    epochs = args.fixed_epochs or args.epochs
    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y, labels in loader:
            x, y, labels = x.to(device), y.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss_classify = loss_fn(logits, y)
            loss = loss_classify
            if miner is not None:
                unique = torch.unique(labels)
                if labels.numel() > 2 and len(unique) > 1:
                    embeddings = model.embeddings(x)
                    hard_pairs = miner(embeddings, labels)
                    loss = loss + args.alpha * loss_metric(embeddings, labels, hard_pairs)
            loss.backward()
            optimizer.step()
            running += float(loss_classify.detach()) * len(x)
            seen += len(x)
        train_loss = running / max(seen, 1)
        val_loss = evaluate_loss(model, X, Y, val_mask, device) if val_mask.any() else train_loss
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"]})
        improved = val_loss < best_loss - 1e-5
        if improved:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or improved:
            log(f"epoch {epoch:3d} train={train_loss:.5f} val={val_loss:.5f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}{' *' if improved else ''}")
        if args.fixed_epochs is None and epoch - best_epoch >= args.patience:
            log(f"early stop at epoch {epoch} (best {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"dim": DIM, "heads": HEADS, "enc_layer": ENCODER_LAYERS, "num_classes": len(CLASSES),
              "classes": CLASSES, "alpha": args.alpha, "lr": args.lr, "own_weight": args.own_weight,
              "seed": args.seed, "batch_size": args.batch_size, "exclude": args.exclude,
              "include_holdout": args.include_holdout, "fixed_epochs": args.fixed_epochs,
              "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()}
    metadata_path = Path(args.dataset).with_name("dataset_meta.json")
    if metadata_path.exists():
        config["phase_policy"] = json.loads(metadata_path.read_text()).get("phase_policy")
    config["sampling_policy"] = "partition-own-train-v1"
    checkpoint = {"state_dict": model.state_dict(), "config": config,
                  "best_val_loss": best_loss, "best_epoch": best_epoch}
    torch.save(checkpoint, args.out / "model.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=1))
    (args.out / "config.json").write_text(json.dumps(config, indent=1))
    log(f"saved {args.out / 'model.pt'} (best val {best_loss:.5f} @ epoch {best_epoch}, "
        f"{time.time() - started:.0f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=REPO / "data_rtmpose" / "dataset.npz")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "holdout")
    parser.add_argument("--exclude", default=None, help="own source_id held out for LORO folds")
    parser.add_argument("--include-holdout", action="store_true",
                        help="fold the RepCount video holdout into training (final refit)")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--own-weight", type=float, default=OWN_WEIGHT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--fixed-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
