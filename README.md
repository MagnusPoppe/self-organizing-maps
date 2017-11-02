# The self organizing map algorithm
Module 4 of the course IT-3105 Artificial intelligence programming at NTNU. Self organizing maps are based on unsupervised,
competitive learning. For this project, the neural network is structured after the "Kohonen network".

### The self organizing map algorithm works in the following stages:

#### Setup
The setup stage is not a part of the self organizing map algorithm. In the setup stage the program reads in the
inputs from file. All inputs are normalized to a value between 0.0 --> 1.0. The network is then created, spawning all
output nodes.

#### 1. Initialization
In this stage, the algorihm connects all input nodes to all output nodes by creating a weight-connection between them.
These weights are used in the next stages.

#### 2. Competition
A random input is drawn. The neurons in the output layer now have to compute their distance from the input. The one
with the lowest distance wins and will be updated to grow even closer to the input.

#### 3. Cooperation
The winning neuron calculates the size of it's neighbourhood (by radius). The neighbours inside the neighbourhood
will also get their weights updated based on the winning node from stage 2. The neighbourhood will grow closer to the
winning node.

#### 4. Adaptation
In this stage, the learning rate and neighbourhood size is adjusted. They both have to decay to achieve the correct
results from the network.

*Steps 2-4 are repeated a set number of times.*