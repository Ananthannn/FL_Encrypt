import tensorflow as tf
import numpy as np
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

    # NEW: split dataset
    client_id = int(input("Enter client id (1 or 2): "))
    start = (client_id-1)*1000
    end = client_id*1000

    x = tf.image.resize(x[start:end] / 255.0, (224, 224))
    y = tf.keras.utils.to_categorical(y[start:end], 10)

    client = FLClient()
    client.connect()

    meta = client.receive_model_metadata()
    model = build_model(meta)

    global_weights = client.receive_global_weights()
    model.set_weights(global_weights)

    print("🧠 Training locally")
    local_weights, history = train_model(
        x, y, model,
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=("accuracy",),
        epochs=1,
        batch_size=8,
    )

    print("📤 Sending update")
    client.send_local_weights(local_weights)
    client.close()
