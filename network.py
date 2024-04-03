"""Network functions.

Classes
---------
Functions
---------

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D

def compile_model(x_train, y_train, settings):
    # NORMALIZATION LAYER
    inputs = Input(shape=x_train.shape[1:])
    layers = inputs

    # DEFINE THE MODEL ARCHITECTURE
    # feed-forward network
    if settings["network_type"] == "ffn":
        assert len(settings["hiddens"]) == len(settings["act_fun"]) == len(settings["ridge_param"])

        layers = Flatten()(layers)
        layers = Dropout(rate=settings["dropout_rate"][0], seed=settings["rng_seed"])(layers)

        for hidden, activation, ridge in zip(settings["hiddens"], settings["act_fun"], settings["ridge_param"]):
            layers = Dense(
                hidden,
                activation=activation,
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=ridge),
                bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
            )(layers)

    # convolutional neural network
    elif settings["network_type"] == "cnn":
        assert len(settings["kernels"]) == len(settings["kernel_act"])
        assert len(settings["hiddens"]) == len(settings["act_fun"])

        for kernel, kernel_act in zip(settings["kernels"], settings["kernel_act"]):
            layers = Conv2D(
                kernel,
                (settings["kernel_size"], settings["kernel_size"]),
                use_bias=True,
                activation=kernel_act,
                padding="same",
                bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
            )(layers)
            layers = MaxPooling2D((2, 2))(layers)

        # make final dense layers
        layers = Flatten()(layers)
        for hidden, activation in zip(settings["hiddens"], settings["act_fun"]):
            layers = Dense(
                hidden,
                activation=activation,
                use_bias=True,
                bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
            )(layers)

    else:
        raise NotImplementedError()

    # DEFINE THE OUTPUT LAYER
    output_layer = Dense(
        y_train.shape[-1],
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
    )(layers)

    # CONSTRUCT THE MODEL
    model = Model(inputs, output_layer)

    # COMPILE THE MODEL
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=settings["learning_rate"]), \
        loss=tf.keras.losses.MeanSquaredError(), \
        metrics=["mae", "mse"] \
    )

    return model