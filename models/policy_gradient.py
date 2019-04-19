import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import random_uniform
import torch
from torch import nn
import pandas as pd
import numpy as np

tf.enable_eager_execution()


class PolicyGradient_tf(tf.keras.Model):
    """This is a policy gradient model class for tensorflow.

    Note that in a true RL problem, the model inputs would include the current state, which is at least partially
    determined by the previous policy output. For illustration, in the models here, the policy output IS the state and
    so is fed back in directly.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """Constructor for PolicyGradient_tf class.
        
        Args:
            input_dim (int): Input dimension (number of features).
            output_dim (int): Output dimension (number of responses).
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.batchNorm1 = layers.BatchNormalization()
        self.dense1 = layers.Dense(
            64, input_shape=(input_dim+output_dim,),
            kernel_initializer=random_uniform(-np.sqrt(1/input_dim), np.sqrt(1/input_dim))
        )
        self.relu1 = layers.Activation('relu')
        self.dense2 = layers.Dense(32, kernel_initializer=random_uniform(-np.sqrt(1/64), np.sqrt(1/64)))
        self.relu2 = layers.Activation('relu')
        self.dense3 = layers.Dense(output_dim, kernel_initializer=random_uniform(-np.sqrt(1/32), np.sqrt(1/32)))
    
    def call(self, x, training=True):

        x = self.batchNorm1(x, training=training)

        out = []
        h = tf.zeros(self.output_dim)
        for x_i in x:
            
            x_i = tf.expand_dims(tf.concat([x_i, h], 0), 0)
            x_i = self.dense1(x_i)
            x_i = self.relu1(x_i)
            x_i = self.dense2(x_i)
            x_i = self.relu2(x_i)
            x_i = self.dense3(x_i)

            h = tf.squeeze(x_i)

            out.append(x_i)
        out = tf.concat(out, 0)

        return out


class PolicyGradient_pt(nn.Module):
    """This is a policy gradient model class for pytorch.

    Note that in a true RL problem, the model inputs would include the current state, which is at least partially
    determined by the previous policy output. For illustration, in the models here, the policy output IS the state and
    so is fed back in directly.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """Constructor for PolicyGradient_pt class.
        
        Args:
            input_dim (int): Input dimension (number of features).
            output_dim (int): Output dimension (number of responses).
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.batchNorm1 = nn.BatchNorm1d(input_dim)
        self.dense1 = nn.Linear(input_dim+output_dim, 64)
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
            
        out = []
        h = torch.zeros(self.output_dim)
        for x_i in x:

            x_i = torch.cat([x_i, h], 0).unsqueeze(0)
            x_i = self.dense1(x_i)
            x_i = self.relu1(x_i)
            x_i = self.dense2(x_i)
            x_i = self.relu2(x_i)
            x_i = self.dense3(x_i)
            
            h = x_i.squeeze()

            out.append(x_i)

        out = torch.cat(out, 0)

        return out


def PolicyGradient(input_dim: int, output_dim: int, mod: str='pytorch'):
    """Wrapper function for the conditional construction of PolicyGradient_tf or PolicyGradient_pt objects.
    
    Args:
        input_dim (int): Input dimension (number of features).
        output_dim (int): Output dimension (number of responses).
        mod (str, optional): Defaults to 'pytorch'. Neural network module to use.
    
    Returns:
        Object of class PolicyGradient_tf or PolicyGradient_pt.
    """

    if mod == 'pytorch':
        return PolicyGradient_pt(input_dim, output_dim)
    elif mod == 'tensorflow':
        return PolicyGradient_tf(input_dim, output_dim)
    else:
        raise Exception('Parameter "mod" should be one of "pytorch" or "tensorflow".')
        
