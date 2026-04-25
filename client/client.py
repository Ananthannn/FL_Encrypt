import os
import sys
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from connection.client_conn import FLClient
from model.model_maker import ModelMaker
from model.train import train_model


def build_model(meta):
    maker = ModelMaker(
        input_shape=tuple(meta["input_shape"]),
        num_classes=meta["num_classes"]
    )

    model, _ = maker.build(
        model_type=meta["model_type"],
        variant=meta["variant"],
        pretrained=meta["pretrained"],
        fine_tune=meta["fine_tune"],
    )

    return model


if __name__ == "__main__":

    (x, y), _ = tf.keras.datasets.cifar10.load_data()

    x = tf.image.resize(x[:1000] / 255.0, (224, 224))
    y = tf.keras.utils.to_categorical(y[:1000], 10)

    client = FLClient()
    client.connect()

    meta = client.receive_model_metadata()
    model = build_model(meta)

    global_weights = client.receive_global_weights()
    model.set_weights(global_weights)

    print("🧠 Training locally")

    local_weights = train_model(
        x, y, model,
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=("accuracy",),
        epochs=2,
        batch_size=16,
    )

    print("📤 Sending update")

    client.send_local_weights(local_weights)
    client.close()
