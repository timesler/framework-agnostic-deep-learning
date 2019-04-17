import tensorflow as tf
from tensorflow import keras
import torch
import inspect


def is_pytorch(obj):
    """Check if object belongs to the pytorch or tensorflow framework.
    
    Args:
        obj (object): Object to check.
    
    Returns:
        bool: True if object is from pytorch, False if from tensorflow.
    """

    if torch.nn.Module in inspect.getmro(obj.__class__):
        return True
    elif torch.Tensor in inspect.getmro(obj.__class__):
        return True
    elif keras.Model in inspect.getmro(obj.__class__):
        return False
    elif tf.Tensor in inspect.getmro(obj.__class__):
        return False
    else:
        raise Exception(
            'Model should be a object that inherits from an appropriate pytorch or '
            'tensorflow class.'
        )


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


def train_batch(model, optimizer, loss_fun, x_batch, y_batch):
    """Update model with a single batch.
    
    Args:
        model (object): Model.
        optimizer (object): Optimizer
        loss_fun (object): Loss function.
        x_batch (object): Batch features.
        y_batch (object): Batch responses.
    
    Returns:
        float: Average loss for the batch, detached from the relevant automatic gradient engine.
    """

    mod = 'pytorch' if is_pytorch(model) else 'tensorflow'

    with tf.GradientTape() if mod == 'tensorflow' else NullContextManager() as tape:
        pred_batch = model(x_batch, training=True)
        loss_batch = loss_fun(y_batch, pred_batch)

        if mod == 'tensorflow':
            grad_batch = tape.gradient(loss_batch, model.trainable_variables)
            optimizer.apply_gradients(zip(grad_batch, model.trainable_variables), tf.train.get_or_create_global_step())
            loss_batch = loss_batch.numpy()
        elif mod == 'pytorch':
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_batch = loss_batch.detach().numpy()
    
    return loss_batch


def train(model, optimizer, loss_fun, loader, epochs: int=1):
    """Train and evaluate a model.
    
    Args:
        model (object): Model.
        optimizer (object): Optimizer
        loss_fun (object): Loss function.
        loader (object): Data loader.
        epochs (int, optional): Defaults to 1. Number of training epochs.
    
    Returns:
        list: Loss for each training batch.
    """
    
    batch_cnt = sum(1 for i in loader)
    losses = []
    for epoch in range(epochs):

        loss_epoch = 0
        for i, (x_batch, y_batch) in enumerate(loader):

            loss_batch = train_batch(model, optimizer, loss_fun, x_batch, y_batch)

            losses.append(loss_batch)
            loss_epoch = (loss_epoch * i + loss_batch) / (i + 1)
            print(f'\rEpoch {epoch + 1} ({i + 1}/{batch_cnt}): {loss_epoch}', end='')
        
        print('')

    return losses