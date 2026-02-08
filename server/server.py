import numpy as np
import tensorflow as tf

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(_file_).resolve().parents[1]))

from model.model_maker import ModelMaker
from connection.server_conn import FLServer

NUM_CLIENTS = 2
ROUNDS = 2


def fed_avg(client_weights):
    new_weights = []
    for weights in zip(*client_weights):
        new_weights.append(np.mean(weights, axis=0))
    return new_weights


if _name_ == "_main_":

    MODEL_META = {
        "model_type": "light",   # or "convnext"
        "variant": "tiny",
        "pretrained": False,
        "fine_tune": False,
        "input_shape": (224,224,3),
        "num_classes": 10,
    }

    maker = ModelMaker(
        input_shape=MODEL_META["input_shape"],
        num_classes=MODEL_META["num_classes"]
    )

    model, global_weights = maker.build(
        model_type=MODEL_META["model_type"],
        variant=MODEL_META.get("variant"),
        pretrained=MODEL_META.get("pretrained"),
        fine_tune=MODEL_META.get("fine_tune"),
    )

    model.set_weights(global_weights)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    server = FLServer(
        model=model,
        model_meta=MODEL_META,
        max_clients=NUM_CLIENTS
    )

    for r in range(ROUNDS):
        print(f"\n🌍 FL Round {r+1}")
        print(f"⏳ Waiting for {NUM_CLIENTS} clients...")

        # BLOCK until all clients connect and send updates
        # FLServer.run_round performs aggregation and returns the global weights
        new_weights = server.run_round(client_sizes=[1000, 1000])

        # Basic sanity check
        if not isinstance(new_weights, list) or len(new_weights) == 0:
            raise RuntimeError("No aggregated weights returned by server")

        print("✅ Global model updated (aggregated on server)")

    print("\n🏁 Federated training complete")

    # Evaluate global model on CIFAR-10 test set
    try:
        print("🔎 Evaluating global model on CIFAR-10 test set")
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = tf.image.resize(x_test/255.0, (224,224))
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        loss, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
        print(f"📈 Global model accuracy: {acc*100:.2f}% (loss: {loss:.4f})")
    except Exception as e:
        print("⚠️  Evaluation failed:", e)

# End of server.py