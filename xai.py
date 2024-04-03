"""Functions for XAI analysis.

Functions
---------
get_gradients(inputs, top_pred_idx=None)
get_integrated_gradients(inputs, baseline=None, num_steps=50, top_pred_idx=None)
random_baseline_integrated_gradients(inputs, num_steps=50, num_runs=5, top_pred_idx=None)

"""

import tensorflow as tf
import numpy as np


def get_gradients(model, inputs, pred_idx=None):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        inputs: 2D/3D/4D matrix of samples
        pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    inputs = tf.cast(inputs, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)

        # Run the forward pass of the layer and record operations
        # on GradientTape.
        preds = model(inputs, training=False)

        # For classification, grab the top class
        if pred_idx is not None:
            preds = preds[:, pred_idx]

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(preds, inputs)
    return np.squeeze(grads)
