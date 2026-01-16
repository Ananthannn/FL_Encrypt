import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, GlobalAveragePooling2D
)


class ModelMaker:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    # -------------------------------------------------
    # Vanilla CNN (from scratch)
    # -------------------------------------------------
    def vanilla_cnn(self, conv_config):
        model = Sequential()

        for i, cfg in enumerate(conv_config):
            model.add(
                Conv2D(
                    filters=cfg["filters"],
                    kernel_size=cfg.get("kernel", 3),
                    activation="relu",
                    padding="same",
                    input_shape=self.input_shape if i == 0 else None
                )
            )

            if cfg.get("pool", False):
                model.add(MaxPooling2D())

            if "dropout" in cfg:
                model.add(Dropout(cfg["dropout"]))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))

        return model, model.get_weights()

    # -------------------------------------------------
    # ConvNeXt (GENERALIZED & CORRECT)
    # -------------------------------------------------
    def make_convnext(self, variant="tiny", pretrained=True, fine_tune=False):
        """
        variant: 'tiny' | 'base' | 'BBC'
        pretrained: use ImageNet weights or not
        fine_tune: unfreeze backbone or not
        """

        if variant == "tiny":
            backbone = tf.keras.applications.ConvNeXtTiny
        elif variant == "base":
            backbone = tf.keras.applications.ConvNeXtBase
        elif variant == "BBC":
            backbone = tf.keras.applications.ConvNeXtLarge
        else:
            raise ValueError("variant must be 'tiny' or 'base'")

        base_model = backbone(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze backbone for transfer learning
        base_model.trainable = fine_tune

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(self.num_classes, activation="softmax")
        ])
        
        return model, model.get_weights()

    def from_config(self , config):
        model = tf.keras.models.model_from_config(config)
        return model

    def build(self, model_type, **kwargs):
        if model_type == "vanilla_cnn":
            return self.vanilla_cnn(**kwargs)

        elif model_type == "convnext":
            return self.make_convnext(**kwargs)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def get_model_metadata(self, model_type, **kwargs):
        return {
            "model_type": model_type,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            **kwargs
        }