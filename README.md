# Artificial Neural Networks

A feed-forward multi-layer artificial neural network built to classify
three types of flowers: Iris Setosa, Iris Versicolour, and Iris Virginica.
The program trains the ANN using the dataset given in `ANN - Iris data.txt`
and classify the plants based on user input. 

## Language and libraries:

Written in:
   Python 3 
using the following libraries:
   numpy, random, time, csv, argparse, logging

## Directory Structure:
~~~~
   |--ann.py
   |—-README.md
~~~~
## Execution instructions
Change the directory to where the script is located (e.g: `$ cd ANN`)

Run the following command:
`$ python [-v] ann.py database_file`

The `-v` flag displays the status of the network, including all potentials
and weights, at each run of the network. It prints a vast amount of
information and is probably best turned off.

## Performance and Benchmark
Training the neural network usually takes about 30 seconds on a desktop
with an i7-4770k processor. 

The ANN is set to stop training when the validation MSE is less than 0.05.
The resulting neural network would usually have a testing accuracy of 97%.

The accuracy of the ANN tends to fluctuate rapidly when training first
begins. The reason for this is not yet understood. However, it always
converges and produces a functional ANN.

## Design Choices
### Architecture
The network has an input layer, a hidden layer, and an output layer. There
are 4 inputs, each representing an attribute of the flower. There are 4
neurons in the hidden layer. There are three neurons in the output layer,
each corresponding to a flower type and indicates the likelihood of a
flower belonging to a certain label given a feature set. The output is thus
an array of three floats that can range from 0 to 1.

### Feature set normalization
The attributes of each flower was divided by 10 for them to work better
with the sigmoid activation function. This choice is rather arbitrary, but
tends to result in an accurate ANN faster on experimentation. 

### Activation Function
The sigmoid function is chosen as the activation function.

### Biases
In this implementation, a bias is applied to the potentials of each neuron
in the hidden and output layers before they are passed into the activation
function. The weights between the bias and each neuron is different and
tuned during training. 

## Collaborators and resources:
- Lecture Slides
- Advice from Anita Lam
- numpy documentation

## Author
Jiayi (Frank) Ma (jiayi.ma@tufts.edu)
