import sys
import tensorflow as tf
from pathlib import Path

# Add the parent directory of 'client' to the search path
path_root = Path(_file_).parents[1]
sys.path.append(str(path_root))
from connection.client_conn import FLClient
from model.model_maker import ModelMaker
from model.train import train_model

def build_model(meta):
    maker = ModelMaker(
        input_shape=tuple(meta["input_shape"]),
        num_classes=meta["num_classes"]
    )
    model, _ = maker.build(**meta)
    return model

if _name_ == "_main_":

    # Each client loads ITS OWN data
    (x, y), _ = tf.keras.datasets.cifar10.load_data()

    x = tf.image.resize(x[:1000]/255.0, (224,224))
    y = tf.keras.utils.to_categorical(y[:1000], 10)

    client = FLClient()
    client.connect()

    meta = client.receive_model_metadata()
    model = build_model(meta)

    global_weights = client.receive_global_weights()
    model.set_weights(global_weights)

    print("🧠 Local training")
    local_weights, history = train_model(
        x, y, model,
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=("accuracy",),
        epochs=5,
        batch_size=8,
    )

    print("📤 Sending weights")
    client.send_local_weights(local_weights)
    client.close()