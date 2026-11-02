"""Model and image loading shared by the notebook and the Streamlit app."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMG_SIZE, SEED


# ---------- images ----------

def load_image(path):
    """Read a stored 256x256 JPEG and resize it to the model input size."""
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    return tf.image.resize(img, (IMG_SIZE, IMG_SIZE))


def make_dataset(paths, labels=None, training=False, batch_size=32):
    """tf.data pipeline: files -> 224x224 float images in [0, 255]."""
    ds = tf.data.Dataset.from_tensor_slices(list(paths))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if labels is not None:
        ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(np.asarray(labels, "float32"))))
    if training:
        ds = ds.shuffle(2048, seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def pil_to_array(img):
    """Uploaded image (PIL) -> (1, 224, 224, 3) array, same steps as in prepare_data."""
    from src.prepare_data import standardize
    img = standardize(img).resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype="float32")[None]


# ---------- model ----------

def make_backbone():
    """MobileNetV2 pretrained on ImageNet, without its classifier.

    Output: one 1280-number vector per image (global average pooling).
    """
    return keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights="imagenet", pooling="avg",
    )


def build_model():
    """Augmentation -> scaling -> frozen MobileNetV2 -> dropout -> sigmoid.

    Augmentation and scaling are layers of the model, so the app can feed
    raw pixels and augmentation is switched off automatically at inference.
    """
    augment = keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.1, seed=SEED),
        layers.RandomContrast(0.1, seed=SEED),
    ], name="augmentation")

    backbone = make_backbone()
    backbone.trainable = False

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)  # [0, 255] -> [-1, 1], what MobileNetV2 expects
    x = backbone(x, training=False)  # keep BatchNorm statistics frozen, also while fine-tuning
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="brainrot_classifier")


def unfreeze_top(model, n_layers=30):
    """Stage 2: let the last n layers of MobileNetV2 adapt to our images.

    BatchNorm layers stay frozen: otherwise they replace the ImageNet statistics
    with statistics of small batches and training becomes unstable.
    """
    backbone = model.get_layer("mobilenetv2_1.00_224")
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def extract_embeddings(paths, batch_size=64):
    """Frozen MobileNetV2 features for the 'embeddings + logistic regression' model."""
    backbone = make_backbone()
    scale = layers.Rescaling(1 / 127.5, offset=-1)
    feats = [backbone(scale(batch), training=False).numpy()
             for batch in make_dataset(paths, batch_size=batch_size)]
    return np.concatenate(feats)
