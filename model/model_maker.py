import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, GlobalAveragePooling2D,
    SeparableConv2D, BatchNormalization, ReLU, Dropout
)

class ModelMaker:
    def _init_(self, input_shape=(224,224,3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes

    # --------------------------------------------------
    # 1️⃣ Vanilla CNN (Baseline model)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2️⃣ ConvNeXt (Advanced model)
    # --------------------------------------------------
    def make_convnext(self, variant="tiny", pretrained=True, fine_tune=False):
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

    # --------------------------------------------------
    # 3️⃣ Lightweight / Mobile-friendly CNN
    # --------------------------------------------------
    def light_cnn(self, width_multiplier: float = 0.5, variant: str = None, **kwargs):
        """Small, efficient CNN using depthwise separable convolutions.
        Designed to be fast on CPU and integrated GPUs (Intel UHD), with
        a low parameter count and small memory footprint.

        Args:
            width_multiplier: shrink channels (0.25 - 1.0). Lower = faster.
            variant: optional named variant (e.g. 'tiny', 'small', 'base') to
                     choose a preset width_multiplier.
            **kwargs: ignored (keeps the API tolerant to extra MODEL_META keys)
        """
        # Allow variant names from MODEL_META to override width_multiplier
        if variant is not None:
            variant_map = {"tiny": 0.25, "small": 0.5, "base": 1.0}
            width_multiplier = variant_map.get(variant, width_multiplier)

        inputs = tf.keras.Input(shape=self.input_shape)

        def block(x, filters, kernel=3, pool=True):
            x = SeparableConv2D(filters, kernel, padding="same", use_bias=False)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            if pool:
                x = MaxPooling2D()(x)
            return x

        f = max(8, int(32 * width_multiplier))
        x = block(inputs, f)
        f = max(16, int(f * 2))
        x = block(x, f)
        f = max(32, int(f * 2))
        x = block(x, f)

        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        return model, model.get_weights()

    # --------------------------------------------------
    # 4️⃣ Unified build interface
    # --------------------------------------------------
    def build(self, model_type="convnext", **kwargs):
        if model_type == "convnext":
            return self.make_convnext(**kwargs)
        elif model_type == "vanilla":
            return self.vanilla_cnn()
        elif model_type == "light":
            return self.light_cnn(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")