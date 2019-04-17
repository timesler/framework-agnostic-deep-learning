# Framework-agnostic deep learning

This repo contains examples of using common code with both the Pytorch and Tensorflow ML/neural network frameworks.

Originally, this project was intended as a way to learn Tensorflow's eager execution mode by porting some typical 
Pytorch code. However, evidenced by the fact that it works with both frameworks, I have adopted this project structure as
a good starting structure for new modelling projects.

Given additional time, I plan to implement more framework-agnostic models, optimizers, and losses.

## Docker setup

The project contains the necessary pieces to spin up a docker container that doubles as a Jupyter server or an API, depending on the intended use. A number of docker shortcuts have been added to the `Makefile` in order to speed things up. However, if make is not installed, you can inspect the `Makefile` to get the relevant commands and run them manually.

Run `make build` to build the docker image. Then run `make jupyter` to start a docker container running jupyter lab. Navigate to localhost:38888 on the host machine to interact with the container.

## Running code inside docker

`main.py` contains a example implementation of framework-agnostic code. The script generates some random data then creates a data loader, model, optimizer and loss function. The model is then fit to the generated data. The script accepts the following optional command-line arguments:

* `--framework`: pytorch or tensorflow (default: pytorch)
* `--samples`: number of samples in generated random data (default: 10000)
* `--features`: number of input features (default: 128)
* `--responses`: number of output responses (default: 8)
* `--batch_size`: batch size (default: 128)
* `--epochs`: number of training epochs (default: 10)
* `--classify`: flag to switch between regression and classification

To run with pytorch: `python main.py --framework pytorch`.

To run with tensorflow: `python main.py --framework tensorflow`.
