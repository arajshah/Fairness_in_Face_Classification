from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_multitask_model(
    *,
    image_size: int = 224,
    weights: str | None = "imagenet",
    base_trainable: bool = False,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build a shared-backbone, multi-output model."""
    base = ResNet50(weights=weights, include_top=False, input_shape=(image_size, image_size, 3))
    base.trainable = bool(base_trainable)

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)

    race_output = Dense(7, activation="softmax", name="race_output")(x)
    gender_output = Dense(2, activation="softmax", name="gender_output")(x)

    model = Model(inputs=base.input, outputs=[race_output, gender_output])
    return model, base


