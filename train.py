"""Train a Quick, Draw! classifier and export browser-compatible CNN weights.

Dependencies: numpy, torch. Each class is a grayscale 28x28 Quick, Draw! .npy file.
Source: https://github.com/googlecreativelab/quickdraw-dataset (CC BY 4.0).
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LABELS = {
    "cat": ("Кошка", "кошку"),
    "fish": ("Рыба", "рыбу"),
    "airplane": ("Самолёт", "самолёт"),
    "apple": ("Яблоко", "яблоко"),
    "bicycle": ("Велосипед", "велосипед"),
    "mushroom": ("Гриб", "гриб"),
    "sailboat": ("Парусник", "парусник"),
    "umbrella": ("Зонт", "зонт"),
}
SAMPLES_PER_CLASS = 5000
SEED = 42


def make_splits(data_dir, names):
    # Shuffle within each class so each split has the same class balance.
    rng = np.random.default_rng(SEED)
    images = [[], [], []]
    labels = [[], [], []]
    for label, name in enumerate(names):
        rows = np.load(data_dir / f"{name}.npy", mmap_mode="r")
        if rows.ndim != 2 or rows.shape[1] != 784 or len(rows) < SAMPLES_PER_CLASS:
            raise ValueError(f"Expected at least {SAMPLES_PER_CLASS} rows of 784 pixels for {name}")
        shuffled = rng.permutation(SAMPLES_PER_CLASS)
        for split_index, indices in enumerate(np.split(shuffled, [4000, 4500])):
            images[split_index].append(rows[indices])
            labels[split_index].append(np.full(len(indices), label, dtype=np.int64))

    splits = []
    for x, y in zip(images, labels):
        pixels = torch.from_numpy(np.concatenate(x)).reshape(-1, 1, 28, 28)
        targets = torch.from_numpy(np.concatenate(y))
        splits.append(TensorDataset(pixels, targets))
    return splits


def make_cnn(class_count):
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, class_count),
    )


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    predictions = []
    for pixels, _ in loader:
        predictions.append(model(pixels.float() / 255).argmax(dim=1))
    labels = loader.dataset.tensors[1]
    predictions = torch.cat(predictions)
    accuracy = float((predictions == labels).float().mean())
    return accuracy, predictions


def fit(model, train, valid, epochs, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_weights = None
    for epoch in range(1, epochs + 1):
        model.train()
        for pixels, labels in train:
            optimizer.zero_grad()
            loss = criterion(model(pixels.float() / 255), labels)
            loss.backward()
            optimizer.step()
        validation_accuracy, _ = evaluate(model, valid)
        print(f"{name} {epoch:02}/{epochs:02} | validation {validation_accuracy:.2%}", flush=True)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_weights = deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


def export_results(model, test, names, accuracy, baseline, predictions, output):
    pixels, labels = test.dataset.tensors
    confusion = np.zeros((len(names), len(names)), dtype=int)
    np.add.at(confusion, (labels.numpy(), predictions.numpy()), 1)
    examples = []
    # Use the first held-out drawing of each class for the demo.
    with torch.inference_mode():
        for index in range(len(names)):
            example_id = int(torch.where(labels == index)[0][0])
            sketch = pixels[example_id]
            logits = model(sketch.unsqueeze(0).float() / 255)
            examples.append({
                "pixels": sketch.flatten().tolist(),
                "probabilities": logits.softmax(dim=1)[0].tolist(),
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": [{"id": name, "name": LABELS[name][0], "draw": LABELS[name][1]} for name in names],
        "weights": {name: tensor.flatten().tolist() for name, tensor in model.state_dict().items()},
        "metrics": {
            "accuracy": accuracy,
            "baseline": baseline,
            "testCount": len(labels),
            "confusion": confusion.tolist(),
        },
        "examples": examples,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    torch.save({"weights": model.state_dict(), "classes": names}, output.with_suffix(".pt"))
    print(f"Baseline {baseline:.2%} | CNN {accuracy:.2%} on {len(labels)} test drawings", flush=True)
    print(confusion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--classes", nargs="+", choices=LABELS, default=list(LABELS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("model.json"))
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.classes)) != len(args.classes) or len(args.classes) < 2:
        parser.error("Use at least one epoch and at least two distinct classes")

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    train, valid, test = make_splits(args.data, args.classes)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=256)
    test_loader = DataLoader(test, batch_size=256)
    print(f"Train {len(train)} | validation {len(valid)} | test {len(test)}", flush=True)

    linear = nn.Sequential(nn.Flatten(), nn.Linear(784, len(args.classes)))
    fit(linear, train_loader, valid_loader, 20, "Linear")
    model = fit(make_cnn(len(args.classes)), train_loader, valid_loader, args.epochs, "CNN")
    baseline, _ = evaluate(linear, test_loader)
    accuracy, predictions = evaluate(model, test_loader)
    export_results(model, test_loader, args.classes, accuracy, baseline, predictions, args.output)


if __name__ == "__main__":
    main()
