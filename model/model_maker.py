import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, GlobalAveragePooling2D
)

class ModelMaker:
    def __init__(self, input_shape=(224,224,3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes

    # ---------------------------
    # Vanilla CNN (NEW)
    # ---------------------------
    def vanilla_cnn(self):
        model = Sequential([
            Conv2D(32, 3, activation="relu", input_shape=self.input_shape),
            MaxPooling2D(),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(self.num_classes, activation="softmax")
        ])
        return model, model.get_weights()

    # ---------------------------
    # ConvNeXt
    # ---------------------------
    def make_convnext(self, variant="tiny", pretrained=False, fine_tune=False):
        backbone = tf.keras.applications.ConvNeXtTiny

        base_model = backbone(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=self.input_shape
        )

        base_model.trainable = fine_tune

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(self.num_classes, activation="softmax")
        ])
        return model, model.get_weights()

    # ---------------------------
    # Build Interface
    # ---------------------------
    def build(self, model_type="convnext", **kwargs):
        if model_type == "convnext":
            return self.make_convnext(**kwargs)
        elif model_type == "vanilla":
            return self.vanilla_cnn()
        else:
            raise ValueError("Unknown model type")
