"""Evaluate the trained classifier on the held-out test set.

Writes to results/:
    metrics.json            accuracy, precision, recall, F1 per class
    confusion_matrix.png
    training_curves.png     (if models/training_history.json exists)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402
from tensorflow import keras  # noqa: E402

from smart_atm.config import CLASS_NAMES, MODELS_DIR, ROOT  # noqa: E402


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Denied", "Granted"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key in zip(axes, ("accuracy", "loss")):
        ax.plot(history.get(key, []), label="train")
        ax.plot(history.get(f"val_{key}", []), label="validation")
        ax.set_title(key.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "dataset" / "test")
    ap.add_argument("--model", type=Path, default=MODELS_DIR / "smart_atm_classifier.keras")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    model = keras.models.load_model(args.model)
    ds = keras.utils.image_dataset_from_directory(
        args.data, labels="inferred", label_mode="binary", class_names=list(CLASS_NAMES),
        image_size=(args.img_size, args.img_size), batch_size=32, shuffle=False,
    )
    y_true = np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
    y_prob = model.predict(ds, verbose=0).ravel()
    y_pred = (y_prob >= args.threshold).astype(int)

    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES),
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = {"test_images": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
               "threshold": args.threshold, "per_class": report, "confusion_matrix": cm.tolist()}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, args.out / "confusion_matrix.png")

    hist_path = args.model.parent / "training_history.json"
    if hist_path.exists():
        plot_history(json.loads(hist_path.read_text()), args.out / "training_curves.png")

    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES), zero_division=0))
    print(f"Accuracy: {metrics['accuracy']:.2%} on {len(y_true)} test images")
    print(f"Results saved to {args.out}/")


if __name__ == "__main__":
    main()
