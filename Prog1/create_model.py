"""Train and save the small Keras HDF5 model used by the demo app.

The demo has no external labelled dataset, so the training set is generated
from the documented signal rule. This keeps the artifact reproducible and
makes its validation score meaningful for this demo signal, not for a real
world domain.
"""

import json
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.h5"
METRICS_PATH = ROOT / "model_metrics.json"


def main() -> None:
    tf.keras.utils.set_random_seed(42)
    features = tf.random.uniform((2400, 3), minval=-2.0, maxval=2.0, seed=42)
    signal = features @ tf.constant([[3.0], [-2.0], [1.2]]) - 0.35
    labels = tf.cast(signal >= 0.0, tf.float32)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3,), name="input_vector"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="prediction"),
        ],
        name="simple_signal_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    history = model.fit(
        features.numpy(),
        labels.numpy(),
        validation_split=0.2,
        epochs=80,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    test_features = tf.random.stateless_uniform(
        (4000, 3), seed=(2026, 9), minval=-2.0, maxval=2.0
    )
    test_signal = test_features @ tf.constant([[3.0], [-2.0], [1.2]]) - 0.35
    test_labels = tf.cast(test_signal >= 0.0, tf.float32).numpy().ravel()
    probabilities = model.predict(test_features, verbose=0).ravel()
    classes = (probabilities >= 0.5).astype("float32")
    confidence = tf.maximum(probabilities, 1.0 - probabilities).numpy()
    reliable = confidence >= 0.65
    metrics = {
        "validation_accuracy": round(float(history.history["val_accuracy"][-1]), 4),
        "holdout_accuracy": round(float((classes == test_labels).mean()), 4),
        "reliable_accuracy": round(float((classes[reliable] == test_labels[reliable]).mean()), 4),
        "coverage": round(float(reliable.mean()), 4),
        "samples": int(len(test_labels)),
        "scope": "synthetic demo signal only",
    }
    model.save(MODEL_PATH, include_optimizer=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Created {MODEL_PATH} ({MODEL_PATH.stat().st_size:,} bytes); "
        f"holdout accuracy: {metrics['holdout_accuracy']:.3f}; "
        f"reliable accuracy: {metrics['reliable_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
