import tensorflow as tf
from model.model_maker import ModelMaker
from connection.server_conn import FLServer

if __name__ == "__main__":

    MODEL_META = {
        "model_type": "convnext",
        "variant": "tiny",
        "pretrained": False,
        "fine_tune": False,
        "input_shape": (224, 224, 3),
        "num_classes": 10,
    }

    maker = ModelMaker(
        input_shape=MODEL_META["input_shape"],
        num_classes=MODEL_META["num_classes"],
    )

    model, _ = maker.build("convnext")

    server = FLServer(model=model, model_meta=MODEL_META, max_clients=2)

    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = tf.image.resize(x_test[:2000]/255.0, (224,224))
    y_test = tf.keras.utils.to_categorical(y_test[:2000],10)

    for rnd in range(2):
        print(f"\n🌍 Round {rnd+1}")
        server.run_round(client_sizes=[1000, 1000])

        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"🌍 Global Accuracy: {acc:.4f}")

    print("✅ Training finished")
