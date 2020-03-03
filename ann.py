import argparse             # For Parsing Arguments
import logging              # For Debugging Functions
import numpy as np
import csv
import random
import time 


'''
Each dataset contains a feature set and their respective labels
'''
class DataSet:
    def __init__(self, feature_set, labels):
        self.feature_set = np.asarray(feature_set) / 10 # normalize
        self.labels = np.asarray(labels)

'''
The artificial neural network
'''
class ANN:
    def __init__(self, filename):
        """
        Creates a new ANN and load training data from file

        Args
        ----
        filename: string
            The name of the file containing training, validation, and
            testing data
        """
        feature_set = []        # temp for all feature sets
        labels = []             # temp for all labels

        with open(filename, newline='') as csvfile:
            data = list(csv.reader(csvfile))

        # convert text labels to a binary vector representation
        for instance in data:
            if len(instance) is not 0:
                if instance[-1] == "Iris-setosa":   
                    labels.append([1, 0, 0])
                elif instance[-1] == "Iris-versicolor":
                    labels.append([0, 1, 0])
                elif instance[-1] == "Iris-virginica":
                    labels.append([0, 0, 1])
                else:
                    print("Label not identified")
                feature_set.append(list(map(float, instance[:-1])))

        # split data 60:20:20 into training, validation, and testing sets
        self.training = DataSet(feature_set[0:30] + feature_set[50:80] + \
                                feature_set[100:130], labels[0:30] + \
                                labels[50:80] + labels[100:130])
        self.validation = DataSet(feature_set[30:40] + feature_set[80:90] + feature_set[140:150], labels[30:40] + labels[80:90] + labels[130:140])
        self.testing = DataSet(feature_set[40:50] + feature_set[90:100] + feature_set[140:150], labels[40:50] + labels[90:100] + labels[140:150])
        
        
        ''''
        Weights between the input and the hidden layer is represented by
        a 4 x 4 matrix, while weights between the hidden and the output 
        layer is represented by a 3 x 4 matrix (3 rows, 4 cols)
        '''
        np.random.seed(42)      # predicable "random" weights
        self.weights_input = np.random.rand(4,4)
        self.weights_hidden = np.random.rand(3,4)
        # an adjustable bias for each of the 7 neurons
        self.weights_biases = np.random.rand(7) 
        
        self.bias = 1
        self.lr = 0.1   # learning rate

        self.overfit_tolerance = 1000
        self.target_mse = 0.05

        logging.debug("Input weights initialized to be:")
        logging.debug(self.weights_input)
        logging.debug("Hidden weights initialized to be:")
        logging.debug(self.weights_hidden)
        logging.debug("Bias weights initialized to be:") 
        logging.debug(self.weights_biases)

    def train(self):
        """
        Trains the ANN

        Args
        ----
        None
        """
        overfit_count = 0
        training_mse = 1
        validation_mse = 1
        prev_training_mse = 1
        prev_validation_mse = 1
        
        verbose = input("Ready to train the ANN. This may take up to a minute. Display mse and accuracy of each generation? (y/n)\n")
        print("Training the neural network...")
        
        training_start_time = time.time()
        
        '''
        Train the neural network until we reach a target MSE of 0.05, or
        until there is an indication of overfitting
        '''
        while overfit_count < self.overfit_tolerance and validation_mse > self.target_mse:
            '''
            Training
            '''
            training_se = 0
            num_correct = 0
            for instance, label in zip(self.training.feature_set, self.training.labels):
                logging.debug("----------------------------Training-------------------------------")
                logging.debug("Feature set: ")
                logging.debug(instance)
                logging.debug("Label: ")
                logging.debug(label)

                '''
                Forward propagation
                '''
                logging.debug("\nForward propagation......................................")
                
                # Calculates the potentials of the hidden layer and
                # calculates output based on the activation function. 
                # potentials_hidden and output_hidden are both 4 x 1 matrices
                potentials_hidden = np.dot(self.weights_input, instance) + self.bias * self.weights_biases[:4]
                logging.debug("Potentials of hidden layer:")
                logging.debug(potentials_hidden)
                output_hidden = self.activation(potentials_hidden)
                logging.debug("Output of hidden layer:")
                logging.debug(output_hidden)

                # Calculates the potentials of the output layer and
                # calculates final output based on the activation function. 
                # potentials_final and output_final are both 3 x 1 matrices
                potentials_final = np.dot(self.weights_hidden, output_hidden) + self.bias * self.weights_biases[4:]
                logging.debug("Potentials of output layer:")
                logging.debug(potentials_final)
                output_final = self.activation(potentials_final)
                logging.debug("---> Output of output layer: ")
                logging.debug(output_final)
                
                # accumulate squared error
                training_se += np.sum((label - output_final)**2)
                
                # accumunate number of correct labels for accuracy calculation
                if label[np.argmax(output_final)] == 1:
                    num_correct += 1
                
                '''
                Backward propagation
                '''
                logging.debug("\nBackward propagation.....................................")
                
                # calculate the error in final output. error_output is a 3 x 1 matrix
                error_output = self.activation_d(output_final) * (label - output_final)
                logging.debug("Error of the output of the output layer: ")
                logging.debug(error_output) 
                # calculate new weights between the hidden layer and the output
                # layer. weights_hidden is a 3 x 4 matrix
                self.weights_hidden += self.lr * np.dot(error_output.reshape(3,1),output_hidden.reshape(4,1).T)
                logging.debug("New hidden layer weights:")
                logging.debug(self.weights_hidden)
                self.weights_biases[4:] += self.lr * self.bias * error_output

                # calculate error in the output of the hidden layer.
                # error_hidden is a 4 x 1 matrix 
                error_hidden = self.activation_d(output_hidden) * np.dot(self.weights_hidden.T, error_output)
                logging.debug("Error of the output of the hidden layer: ")
                logging.debug(error_hidden)
                # calculate new weights between the input layer and the hidden
                # layer. weights_input is a 4 x 4 matrix
                self.weights_input += self.lr * np.dot(error_hidden.reshape(4,1), instance.reshape(4,1).T)
                logging.debug("New input layer weights:")
                logging.debug(self.weights_input)
                self.weights_biases[:4] += self.lr * self.bias * error_hidden
            
            # accuracy and mse measurements
            training_mse = training_se / 90
            if verbose == "y":
                print("Training MSE = ", np.around(training_mse, decimals=4), end="     ")
                print("Training Accuracy: ", np.around(num_correct / 90 * 100, decimals=4), "%", end="     ")
            prev_training_mse = training_mse

            '''
            Validation
            '''
            validation_se = 0
            num_correct = 0
            for instance, label in zip(self.validation.feature_set, self.validation.labels):
                logging.debug("-----------------------------Validation------------------------------")
                logging.debug("Feature set: ")
                logging.debug(instance)
                logging.debug("Label: ")
                logging.debug(label)

                # Exactly the same as training, except without back propagation
                logging.debug("\nForward propagation......................................")
                potentials_hidden = np.dot(self.weights_input, instance) + self.bias * self.weights_biases[:4]
                logging.debug("Potentials of hidden layer:")
                logging.debug(potentials_hidden)
                output_hidden = self.activation(potentials_hidden)
                logging.debug("Output of hidden layer:")
                logging.debug(output_hidden)

                potentials_final = np.dot(self.weights_hidden, output_hidden) + self.bias * self.weights_biases[4:]
                logging.debug("Potentials of output layer:")
                logging.debug(potentials_final)
                output_final = self.activation(potentials_final)
                logging.debug("---> Output of output layer: ")
                logging.debug(output_final)
                
                validation_se += np.sum((label - output_final)**2)
                
                if label[np.argmax(output_final)] == 1:
                    num_correct += 1
            
            validation_mse = validation_se / 30
            if verbose == "y":
                print("Validation MSE = ", np.around(validation_mse, decimals=4), end="     ")
                print("Validation Accuracy: ", np.around(num_correct / 30 * 100, decimals=4), "%")
            prev_validation_mse = validation_mse
            

            # Train 1000 generations past the point when validation mse increase while
            # training mse decrease
            if validation_mse >= prev_validation_mse and training_mse < prev_training_mse:
                overfit_count += 1

        print("--- Training Time: %0.5f seconds ---" % (time.time() - training_start_time))
                
        '''
        Testing
        '''
        num_correct = 0
        for instance, label in zip(self.testing.feature_set, self.testing.labels):
            logging.debug("------------------------------Testing-----------------------------")
            logging.debug("Feature set: ")
            logging.debug(instance)
            logging.debug("Label: ")
            logging.debug(label)

            logging.debug("\nForward propagation......................................")
            
            potentials_hidden = np.dot(self.weights_input, instance) + self.bias * self.weights_biases[:4]
            logging.debug("Potentials of hidden layer:")
            logging.debug(potentials_hidden)
            output_hidden = self.activation(potentials_hidden)
            logging.debug("Output of hidden layer:")
            logging.debug(output_hidden)

            potentials_final = np.dot(self.weights_hidden, output_hidden) + self.bias * self.weights_biases[4:]
            logging.debug("Potentials of output layer:")
            logging.debug(potentials_final)
            output_final = self.activation(potentials_final)
            logging.debug("---> Output of output layer: ")
            logging.debug(output_final)
            
            if label[np.argmax(output_final)] == 1:
                num_correct += 1
            else:
                logging.debug("^\n^\n^\n^\n^\n^\nFailed")
                pass
        print("Accuracy of testing is: ", num_correct / 30 * 100, "%")

    def activation(self, potential):
        """
        Given a potential, calculates an output

        Args
        ----
        potential: numpy array of floats

        Returns
        ---
        Output of the neuron according to the activation function
        """
        return 1/(1 + np.exp(-potential))
    def activation_d(self, potential):
        """
        Given a potential, calculates the derivative of the activation
        function. Used for gradient descent.

        Args
        ----
        potential: numpy array of floats

        Returns
        ---
        Derivative of the activation function
        """
        return self.activation(potential) * (1 - self.activation(potential))
    
    def run(self):
        """
        Runs the trained neural network according to user input

        Args
        ----
        None
        """
        while True:
            user_input = input("Please enter 4 numerics, separated by a space, in the order of:\n1. sepal length in cm\n2. sepal width in cm\n3. petal length in cm\n4. petal width in cm\nTo quit, enter q\n")
            if user_input == "q":
                return
            else:
                custom_input = np.asarray(list(map(float, user_input.split()))) / 10
                potentials_hidden = np.dot(self.weights_input, custom_input) + self.bias * self.weights_biases[:4]
                output_hidden = self.activation(potentials_hidden)
                potentials_final = np.dot(self.weights_hidden, output_hidden) + self.bias * self.weights_biases[4:]
                output_final = self.activation(potentials_final)
                
                if np.argmax(output_final) == 0:   
                    label = "Iris-setosa"
                elif np.argmax(output_final) == 1:   
                    label = "Iris-versicolor"
                elif np.argmax(output_final) == 2:   
                    label = "Iris-virginica"

                print("I am", np.around(np.max(output_final) * 100, decimals=2), "% certain that this is", label)

def main():
    # Setup parser and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", 
                        help="Increase verbosity", 
                        action="store_true")
    parser.add_argument(dest = "filename",
                        help = "Name of file containing training data.")    
    args = parser.parse_args()
    
    # Setup Debugging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    ann = ANN(args.filename)
    ann.train()
    ann.run()

if __name__ == "__main__":
    main()