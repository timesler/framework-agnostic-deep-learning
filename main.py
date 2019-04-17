import numpy as np
import argparse
from datasets.dual_dataset import create_loader
from models.mlp import Regressor
from models.training import train
from models.optimizers import adam
from losses.continuous import mse
from losses.discrete import cross_entropy_logits


if __name__ == "__main__":
    
    # Default parameters
    classify = True
    samples = 10000
    features = 128
    responses = 8
    batch_size = 128
    epochs = 10
    framework = 'pytorch'

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify', action='store_true', default=False)
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
    if classify:
        y = np.argmax(y, axis=1).astype(np.int64)

    # Create a data loader
    loader = create_loader(x, y, batch_size, framework)

    # Create a model and optimizer
    model = Regressor(features, responses, framework)
    optimizer = adam(model, learning_rate=0.01)

    # Train model
    losses = train(model, optimizer, cross_entropy_logits if classify else mse, loader, epochs)
