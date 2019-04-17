import tensorflow as tf
import torch
from models.training import is_pytorch


def mse(y_true, y_pred):
    """Mean squared error loss function."""

    if is_pytorch(y_true):
        return ((y_true - y_pred) ** 2).mean()
    else:
        return tf.reduce_mean((y_true - y_pred) ** 2)
