import tensorflow as tf
import torch
from torch.nn import functional as F
from models.training import is_pytorch


def cross_entropy_logits(y_true, y_pred):
    """Mean squared error loss function."""

    if is_pytorch(y_true):
        return F.cross_entropy(y_pred, y_true)
    else:
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
