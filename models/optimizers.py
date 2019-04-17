import tensorflow as tf
from tensorflow import keras
import torch
from models.training import is_pytorch


def adam(model, learning_rate=0.0001):
    """Adam optimizer for both pytorch and tensorflow models."""

    if is_pytorch(model):
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
