import numpy as np
import tensorflow as tf
from model.model_maker import ModelMaker
from connection.server_conn import FLServer

NUM_CLIENTS = 2
ROUNDS = 2


def fed_avg(client_weights):
    new_weights = []
    for weights in zip(*client_weights):
        new_weights.append(np.mean(weights, axis=0))
    return new_weights


if __name__ == "__main__":

    MODEL_META = {
        "model_type": "vanilla",   # or "convnext"
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
        client_weights = server.run_round()

        # Safety check (important)
        if len(client_weights) < NUM_CLIENTS:
            raise RuntimeError("Not enough client updates received")

        print("🔄 Aggregating updates")
        new_weights = fed_avg(client_weights)

        model.set_weights(new_weights)
        print("✅ Global model updated")

    print("\n🏁 Federated training complete")
