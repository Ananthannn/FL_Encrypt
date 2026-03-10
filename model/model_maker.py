import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


class ModelMaker:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def make_convnext(self, variant="tiny", pretrained=True, fine_tune=False):

        variant = variant.lower()

        if variant == "tiny":
            backbone_fn = tf.keras.applications.ConvNeXtTiny
        elif variant == "base":
            backbone_fn = tf.keras.applications.ConvNeXtBase
        elif variant == "large":
            backbone_fn = tf.keras.applications.ConvNeXtLarge
        else:
            raise ValueError("variant must be 'tiny', 'base', or 'large'")

        base_model = backbone_fn(
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

    def build(self, model_type, **kwargs):
        if model_type == "convnext":
            return self.make_convnext(**kwargs)
        else:
            raise ValueError("Only 'convnext' supported.")
