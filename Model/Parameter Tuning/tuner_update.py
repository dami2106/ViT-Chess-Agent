import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import Sequence
from tensorflow import keras
import numpy as np
import random
import gzip
from sklearn.model_selection import train_test_split
import tensorflow.keras.callbacks as callbacks
import keras_tuner

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


model_name = "vit"
dataset = "800_Elo"
side = "from"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

def vit(hp):
    num_classes = 64
    input_shape = (8, 8, 14)

    patch_size = hp.Int("patch_size", min_value=2, max_value=8, step=2)
    num_patches = (8 // patch_size) ** 2

    projection_dim = hp.Int("proj_dim", min_value=16, max_value=256, step=16)

    num_heads = hp.Int("num_heads", min_value=2, max_value=6, step=1)

    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]

    transformer_layers = hp.Int("trans_layers", min_value=2, max_value=6, step=1)

    head_one = hp.Int("head_one", min_value=64, max_value=2048, step=64)
    head_two = hp.Int("head_two", min_value=64, max_value=2048, step=64)

    mlp_head_units = [head_one, head_two]

    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    # %%
    class Patches(layers.Layer):
        def __init__(self, patch_size):
            super().__init__()
            self.patch_size = patch_size

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches

    # %%
    class PatchEncoder(layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super().__init__()
            self.num_patches = num_patches
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded

    # %%
    def create_vit_classifier():
        inputs = layers.Input(shape=input_shape)
        # Create patches.
        patches = Patches(patch_size)(inputs)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)

        return model


    return create_vit_classifier()

class EarlyStoppingByLoss(keras.callbacks.Callback):
    def __init__(self, max_loss):
        self.max_loss = max_loss

    def on_train_batch_end(self, batch, logs=None):
        if logs["loss"] >= self.max_loss:
            self.model.stop_training = True

def invalid_loss(y_true, y_pred):
    return keras.losses.CategoricalCrossentropy()(y_true, y_pred) + 2000000


def invalid_network(hp):
    model = vit(hp)
    model.compile(optimizer="adam", loss=invalid_loss, metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)
    return model

def valid_network(hp):
    model = vit(hp)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001, decay=0.0001
    ), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)
    return model


def build_model(hp):
    try:
        model = valid_network(hp)
    except:
        model = invalid_network(hp)
    return model

print(f"Tuning {model_name}_{dataset}_{side}.")
path = ''
f = gzip.GzipFile(f"{path}{dataset}/boards.npy.gz", "r")
board = np.load(f)[:1000000]
f.close()

f = gzip.GzipFile(f"{path}{dataset}/{side}.npy.gz", "r")
labels = np.load(f)[:1000000]
f.close()

board = np.transpose(board, (0, 2, 3, 1))

X_train, X_validate, y_train, y_validate = train_test_split(board, labels, test_size=0.1, random_state=SEED)
board = None
labels = None

train_gen = DataGenerator(X_train, y_train, 2048)
valid_gen = DataGenerator(X_validate, y_validate, 2048)

tuner = keras_tuner.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=100,
    factor=3,
    seed=SEED,
    hyperband_iterations=5,
    directory='tuner',
    project_name='vit',
    max_consecutive_failed_trials=15,
    overwrite=True)

tuner.search_space_summary()

callbacks = [EarlyStoppingByLoss(900000)]

tuner.search(
    train_gen,
    epochs=10,
    callbacks=callbacks,
    steps_per_epoch=len(train_gen),
    validation_data=valid_gen
    )