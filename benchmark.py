from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

from main import Model


class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


@dataclass
class Confusion:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class ColaMetrics:
    loss: float
    mcc: float
    confusion: Confusion
    examples: int


def load_cola(filepath: str) -> list[tuple[bytes, int]]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"CoLA file not found at '{filepath}'. Provide --cola-train-path/--cola-eval-path or disable CoLA mode."
        )

    data: list[tuple[bytes, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            data.append((parts[3].encode("utf-8"), int(parts[1])))

    if not data:
        raise ValueError(
            f"No usable CoLA rows found in '{filepath}'. Expected tab-separated rows with 4 columns."
        )
    return data


def mcc(confusion: Confusion) -> float:
    import math

    tp, tn, fp, fn = confusion.tp, confusion.tn, confusion.fp, confusion.fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0


def _final_state_for_text(model: Model, text_bytes: bytes) -> mx.array:
    dummies = [mx.zeros((model.dim,)) for _ in range(model.layercount)]
    for b in text_bytes:
        x = model.encoder(mx.array(b))
        for j, layer in enumerate(model.layers):
            x, state, _ = layer(x, dummies[j])
            layer.states = mx.stop_gradient(state)
    return model.layers[-1].states


def _update_confusion(confusion: Confusion, predicted_class: int, label: int) -> None:
    if predicted_class == 1 and label == 1:
        confusion.tp += 1
    elif predicted_class == 0 and label == 0:
        confusion.tn += 1
    elif predicted_class == 1 and label == 0:
        confusion.fp += 1
    elif predicted_class == 0 and label == 1:
        confusion.fn += 1


def train_cola_head(
    model: Model,
    train_data: Iterable[tuple[bytes, int]],
    *,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    progress_every: int = 200,
    progress_cb: Callable[[dict], None] | None = None,
) -> tuple[Classification, ColaMetrics]:
    model.freeze()
    head = Classification(model.dim)
    headopt = opt.AdamW(learning_rate=learning_rate)

    def loss_fn(params: dict, state: mx.array, target: int):
        head.update(params)
        logits = head(state)
        loss = nn.losses.cross_entropy(logits[None, :], mx.array([target])).mean()
        return loss, logits

    last_metrics = ColaMetrics(loss=0.0, mcc=0.0, confusion=Confusion(), examples=0)
    train_data = list(train_data)
    for epoch in range(epochs):
        confusion = Confusion()
        total_loss = 0.0
        for i, (example_bytes, label) in enumerate(train_data, start=1):
            final_state = _final_state_for_text(model, example_bytes)
            (loss, logits), grads = mx.value_and_grad(loss_fn, argnums=0)(head.trainable_parameters(), final_state, label)
            headopt.update(head, grads)
            mx.eval(head.parameters(), headopt.state)

            total_loss += float(loss.item())
            predicted_class = int(mx.argmax(logits).item())
            _update_confusion(confusion, predicted_class, label)

            score = mcc(confusion)
            if progress_cb is not None and (i == 1 or i % progress_every == 0 or i == len(train_data)):
                progress_cb(
                    {
                        "phase": "train",
                        "epoch": epoch + 1,
                        "step": i,
                        "loss": total_loss / i,
                        "mcc": score,
                        "tp": confusion.tp,
                        "tn": confusion.tn,
                        "fp": confusion.fp,
                        "fn": confusion.fn,
                    }
                )

        avg_loss = total_loss / len(train_data)
        last_metrics = ColaMetrics(
            loss=avg_loss,
            mcc=mcc(confusion),
            confusion=confusion,
            examples=len(train_data),
        )

    return head, last_metrics


def evaluate_cola_head(
    model: Model,
    head: Classification,
    eval_data: Iterable[tuple[bytes, int]],
    *,
    progress_every: int = 200,
    progress_cb: Callable[[dict], None] | None = None,
) -> ColaMetrics:
    eval_data = list(eval_data)
    confusion = Confusion()
    total_loss = 0.0

    for i, (example_bytes, label) in enumerate(eval_data, start=1):
        final_state = _final_state_for_text(model, example_bytes)
        logits = head(final_state)
        loss = nn.losses.cross_entropy(logits[None, :], mx.array([label])).mean()
        total_loss += float(loss.item())

        predicted_class = int(mx.argmax(logits).item())
        _update_confusion(confusion, predicted_class, label)
        score = mcc(confusion)

        if progress_cb is not None and (i == 1 or i % progress_every == 0 or i == len(eval_data)):
            progress_cb(
                {
                    "phase": "eval",
                    "step": i,
                    "loss": total_loss / i,
                    "mcc": score,
                    "tp": confusion.tp,
                    "tn": confusion.tn,
                    "fp": confusion.fp,
                    "fn": confusion.fn,
                }
            )

    return ColaMetrics(
        loss=total_loss / len(eval_data),
        mcc=mcc(confusion),
        confusion=confusion,
        examples=len(eval_data),
    )


def run(
    checkpoint_path: str = "smaller-4.5m.safetensors",
    train_path: str = "CoLA/original/raw/in_domain_train.tsv",
    eval_path: str | None = None,
    epochs: int = 3,
) -> None:
    model = Model(dim=512, layers=16, temp=0.75, lr=5e-4)
    model.load(checkpoint_path)

    train_data = load_cola(train_path)
    head, train_metrics = train_cola_head(model, train_data, epochs=epochs, progress_cb=lambda x: print(x))
    print(
        f"train: loss={train_metrics.loss:.6f} mcc={train_metrics.mcc:.6f} "
        f"tp={train_metrics.confusion.tp} tn={train_metrics.confusion.tn} "
        f"fp={train_metrics.confusion.fp} fn={train_metrics.confusion.fn}"
    )

    if eval_path:
        eval_data = load_cola(eval_path)
        eval_metrics = evaluate_cola_head(model, head, eval_data, progress_cb=lambda x: print(x))
        print(
            f"eval: loss={eval_metrics.loss:.6f} mcc={eval_metrics.mcc:.6f} "
            f"tp={eval_metrics.confusion.tp} tn={eval_metrics.confusion.tn} "
            f"fp={eval_metrics.confusion.fp} fn={eval_metrics.confusion.fn}"
        )


if __name__ == "__main__":
    run()
