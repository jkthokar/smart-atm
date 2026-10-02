"""Train the face-occlusion classifier (MobileNetV2 transfer learning).

Stage 1 trains only the new head with the backbone frozen.
Stage 2 fine-tunes the top layers of the backbone with a small learning rate.

    python train.py --data dataset/train --epochs 10 --fine-tune-epochs 5
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from smart_atm.classifier import build_model
from smart_atm.config import CLASS_NAMES, MODELS_DIR, ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "dataset" / "train")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "smart_atm_classifier.keras")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--fine-tune-epochs", type=int, default=5)
    ap.add_argument("--fine-tune-layers", type=int, default=30, help="backbone layers to unfreeze")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    common = dict(
        directory=args.data, labels="inferred", label_mode="binary", class_names=list(CLASS_NAMES),
        image_size=(args.img_size, args.img_size), batch_size=args.batch,
        validation_split=args.val_split, seed=args.seed,
    )
    train_ds = keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(subset="validation", **common)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model, base = build_model(args.img_size, weights=None if args.no_pretrained else "imagenet")
    metrics = ["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    print("\n== Stage 1: training classification head ==")
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="binary_crossentropy", metrics=metrics)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    history = {k: [float(x) for x in v] for k, v in h1.history.items()}
    if args.fine_tune_epochs > 0:
        print(f"\n== Stage 2: fine-tuning top {args.fine_tune_layers} backbone layers ==")
        base.trainable = True
        for layer in base.layers[: -args.fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers:  # keep BatchNorm statistics frozen for stable fine-tuning
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(args.lr / 10), loss="binary_crossentropy", metrics=metrics)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=callbacks)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(float(x) for x in v)

    model.save(args.out)
    (args.out.parent / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nSaved model to {args.out}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
