import tensorflow as tf
from tensorflow import keras
import torch
from torch.utils import data
import numpy as np


def create_dataset(x: np.ndarray, y: np.ndarray, mod: str='pytorch'):
    """Create a dataset object. These objects are used to iterate through the samples of a data source.
    
    Args:
        x (np.ndarray): Feature dataset.
        y (np.ndarray): Response dataset.
        mod (str, optional): Defaults to 'pytorch'. Neural network module to use.
    
    Returns:
        Either a pytorch or tensorflow dataset object.
    """

    if mod == 'pytorch':
        return data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    elif mod == 'tensorflow':
        return tf.data.Dataset.from_tensor_slices((x, y))
    else:
        raise Exception('Parameter "mod" should be one of "pytorch" or "tensorflow".')


def create_loader(x: np.ndarray, y: np.ndarray, batch_size: int, mod: str='pytorch'):
    """Create a data loader object. These objects are used to iterate through batches of a data source.
    
    Args:
        x (np.ndarray): Feature dataset.
        y (np.ndarray): Response dataset.
        batch_size (int): Batch size.
        mod (str, optional): Defaults to 'pytorch'. Neural network module to use.
    
    Returns:
        Either a pytorch or tensorflow data loader object.
    """

    dataset = create_dataset(x, y, mod)

    if mod == 'pytorch':
        return data.DataLoader(dataset, batch_size=batch_size)
    else:
        return dataset.batch(batch_size)
