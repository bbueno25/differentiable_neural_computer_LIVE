# Differentiable Neural Computer

## Introduction

We're going to build a Differentiable Neural Computer capable of learning the mapping between binary inputs and outputs. The point of this demo is to break the DNC down to its bare essentials so we can really understand how the architecture works.

DNCs can be viewed as a more general type of LSTM where the network learns how to use memory to understand data rather than attempting to learn the raw sequence relationships. This allows the network to be trained on a small amount of data and generalize to large amount of data without training, as well as handling inputs that were not necessarily seen during training --- a huge divergence from what was previously possible.

These findings are facilitated by the novel framework of memory which is fully differentiable (thus the name). Because of this property, the memory structure is able to be coupled with a neural network and trained by gradient descent or any other optimization method.

## Materials

## Procedures

run `python differentiable_neural_computer.py` in terminal.

## Authors

- **B. Bueno** - [bbueno5000](https://github.com/bbueno5000)

## Acknowledgements

- [claymcleod](https://github.com/claymcleod)
- [llSourcell](https://github.com/llSourcell)
