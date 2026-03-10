import os
import sys
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from model.model_maker import ModelMaker
from connection.server_conn import FLServer


if __name__ == "__main__":

    MODEL_META = {
        "model_type": "convnext",
        "variant": "tiny",
        "pretrained": True,
        "fine_tune": False,
        "input_shape": (224, 224, 3),
        "num_classes": 10,
    }

    model_maker = ModelMaker(
        input_shape=MODEL_META["input_shape"],
        num_classes=MODEL_META["num_classes"],
    )

    model, _ = model_maker.build(
        model_type="convnext",
        variant=MODEL_META["variant"],
        pretrained=MODEL_META["pretrained"],
        fine_tune=MODEL_META["fine_tune"],
    )

    server = FLServer(
        model=model,
        model_meta=MODEL_META,
        max_clients=2,
        port=9999,
    )

    for rnd in range(3):
        print(f"\n🌍 Round {rnd+1}")
        server.run_round(client_sizes=[1000, 1000])

    print("\n✅ Training finished")
