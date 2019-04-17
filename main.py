import numpy as np
import argparse
from datasets.dual_dataset import create_loader
from models.mlp import Regressor
from models.training import train
from models.optimizers import adam
from losses.continuous import mse


if __name__ == "__main__":
    
    # Default parameters
    samples = 10000
    features = 128
    responses = 8
    batch_size = 128
    epochs = 10
    framework = 'tensorflow'

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=samples, required=False)
    parser.add_argument('--features', type=int, default=features, required=False)
    parser.add_argument('--responses', type=int, default=responses, required=False)
    parser.add_argument('--batch_size', type=int, default=batch_size, required=False)
    parser.add_argument('--epochs', type=int, default=epochs, required=False)
    parser.add_argument('--framework', type=str, default=framework, required=False)
    args = parser.parse_args()
    locals().update(vars(args))

    # Generate some random data
    x = np.random.rand(samples, features).astype(np.float32)
    y = np.random.rand(samples, responses).astype(np.float32)

    # Create a data loader
    loader = create_loader(x, y, batch_size, framework)

    # Create a model and optimizer
    regressor = Regressor(x.shape[1], y.shape[1], framework)
    optimizer = adam(regressor, learning_rate=0.01)

    # Train model
    losses = train(regressor, optimizer, mse, loader, epochs)
