import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import random_uniform
from torch import nn
import pandas as pd
import numpy as np

tf.enable_eager_execution()


class Regressor_tf(tf.keras.Model):
    """This is a MLP regressor class for tensorflow."""

    def __init__(self, input_dim: int, output_dim: int):
        """Constructor for Regressor_tf class.
        
        Args:
            input_dim (int): Input dimension (number of features).
            output_dim (int): Output dimension (number of responses).
        """

        super(Regressor_tf, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        initializer = random_uniform(-np.sqrt(1/input_dim), np.sqrt(1/input_dim))
        self.batchNorm1 = layers.BatchNormalization()
        self.dense1 = layers.Dense(64, input_shape=(input_dim,), kernel_initializer=initializer)
        self.relu1 = layers.Activation('relu')
        self.dense2 = layers.Dense(32, kernel_initializer=initializer)
        self.relu2 = layers.Activation('relu')
        self.dense3 = layers.Dense(output_dim, kernel_initializer=initializer)
    
    def call(self, x, training=True):

        x = self.batchNorm1(x, training=training)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)

        return x


class Regressor_pt(nn.Module):
    """This is a MLP regressor class for pytorch."""

    def __init__(self, input_dim: int, output_dim: int):
        """Constructor for Regressor_pt class.
        
        Args:
            input_dim (int): Input dimension (number of features).
            output_dim (int): Output dimension (number of responses).
        """

        super(Regressor_pt, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.batchNorm1 = nn.BatchNorm1d(input_dim)
        self.dense1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(32, output_dim)
    
    def forward(self, x, training=True):

        if training:
            self.train()
        else:
            self.eval()
            
        x = self.batchNorm1(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)

        return x


def Regressor(input_dim: int, output_dim: int, mod: str='pytorch'):
    """Wrapper function for the conditional construction of Regressor_tf or Regressor_pt objects.
    
    Args:
        input_dim (int): Input dimension (number of features).
        output_dim (int): Output dimension (number of responses).
        mod (str, optional): Defaults to 'pytorch'. Neural network module to use.
    
    Returns:
        Object of class Regressor_tf or Regressor_pt.
    """

    if mod == 'pytorch':
        return Regressor_pt(input_dim, output_dim)
    elif mod == 'tensorflow':
        return Regressor_tf(input_dim, output_dim)
    else:
        raise Exception('Parameter "mod" should be one of "pytorch" or "tensorflow".')


class Classifier_tf(tf.keras.Model):
    """This is a MLP classifier class for tensorflow."""

    def __init__(self, input_dim: int, output_dim: int):
        """Constructor for Classifier_tf class.
        
        Args:
            input_dim (int): Input dimension (number of features).
            output_dim (int): Output dimension (number of responses).
        """

        super(Classifier_tf, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        initializer = random_uniform(-np.sqrt(1/input_dim), np.sqrt(1/input_dim))
        self.batchNorm1 = layers.BatchNormalization()
        self.dense1 = layers.Dense(64, input_shape=(input_dim,), kernel_initializer=initializer)
        self.relu1 = layers.Activation('relu')
        self.dense2 = layers.Dense(32, kernel_initializer=initializer)
        self.relu2 = layers.Activation('relu')
        self.dense3 = layers.Dense(output_dim, kernel_initializer=initializer)
        self.softmax1 = layers.Activation('softmax')
    
    def call(self, x, training=True):

        x = self.batchNorm1(x, training=training)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.softmax1(x)

        return x


class Classifier_pt(nn.Module):
    """This is a MLP classifier class for pytorch."""

    def __init__(self, input_dim: int, output_dim: int):
        """Constructor for Classifier_pt class.
        
        Args:
            input_dim (int): Input dimension (number of features).
            output_dim (int): Output dimension (number of responses).
        """

        super(Classifier_pt, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.batchNorm1 = nn.BatchNorm1d(input_dim)
        self.dense1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(32, output_dim)
        self.softmax1 = nn.Softmax()
    
    def forward(self, x, training=True):

        if training:
            self.train()
        else:
            self.eval()
            
        x = self.batchNorm1(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.softmax1(x)

        return x


def Classifier(input_dim: int, output_dim: int, mod: str='pytorch'):
    """Wrapper function for the conditional construction of Classifer_tf or Classifer_pt objects.
    
    Args:
        input_dim (int): Input dimension (number of features).
        output_dim (int): Output dimension (number of responses).
        mod (str, optional): Defaults to 'pytorch'. Neural network module to use.
    
    Returns:
        Object of class Classifer_tf or Classifer_pt.
    """

    if mod == 'pytorch':
        return Classifer_pt(input_dim, output_dim)
    elif mod == 'tensorflow':
        return Classifer_tf(input_dim, output_dim)
    else:
        raise Exception('Parameter "mod" should be one of "pytorch" or "tensorflow".')
        
