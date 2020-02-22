import math
import os  # Used to search directory when saving or loading files
import random  # Used to generate random numbers
import statistics
import time  # Used get runtime

from mnist import MNIST  # Used to unpack the MNIST database


class Node:
    def __init__(self, row, layer, is_input=False, is_output=False):

        self.connections = []  # Stores the nodes this node outputs to
        self.inputs = []  # Stores the nodes this node receives input from
        self.row = row  # Defines the node's row
        self.layer = layer  # Defines the node's layer
        self.is_input = is_input  # Defines if the node is an input node
        self.is_output = is_output  # Defines if the node is an output node
        self.weights = []  # Stores the node's weights
        self.old_weights = []  # Stores the node's weights before they're changes, for use in backpropagation.
        self.net = 0  # Stores the node's net value
        self.output_value = 0  # Stores the node's output value
        self.change_in_weight = 0  # Stores the node's change in weight value
        self.delta_rule = 0  # Stores the node's delta rule
        self.old_weight_change = []

    def create_old_changes(self):
        if not self.is_input:
            self.old_weight_change = [0 for x in range(len(self.inputs))]

    def add_connections(self, desired_row, desired_layer, node_list, network__structure):
        if not (desired_row == self.row and desired_layer == self.layer):
            node_id = sum(network__structure[0:desired_layer]) + desired_row
            desired_node = node_list[node_id]

            if desired_node.row == desired_row and desired_node.layer == desired_layer:
                self.connections.append(desired_node)
                desired_node.inputs.append(self)
                return None

            print("add_connections calling a node that doesnt exist or the wrong node")  # Error Chekcing (:
            print(desired_row, desired_layer)

    def show_connections(self):
        temp = []  # Creates a temporary list
        for i in self.connections:  # Iterates through the nodes connections
            temp.append((i.row,
                         i.layer))  # Adds each connection to a temporary list in an easy to read fashion (in vector form).
        return temp  # returns the temporary list

    def show_inputs(self):
        temp = []  # Creates a temporary list
        for i in self.inputs:  # Iterates through the nodes connections
            temp.append((i.row,
                         i.layer))  # Adds each connection to a temporary list in an easy to read fashion (in vector form).
        return temp  # returns the temporary list

    def add_weights(self, weights):
        if (len(weights) - 1 == len(
                self.inputs)) and not self.is_input:  # Checking if the number of inputs the node has equals the number of weights (+1 for the offset weight)
            self.weights = weights  # Sets its weights to the weights provided
            return None  # Then Exits
        print("add_weights has been supplied more weights than the "  # Error Chekcing (:
              "node has connections or its an input node")

    def calculate_net(self, data_input=0):
        self.net = 0
        if not self.is_input:  # Checks if the node is an input node, input nodes don't have weights, if it's not an input node the program continues.
            self.net = (sum([a * b for a, b in zip([x.output_value for x in self.inputs], self.weights[1:])]))
            self.net += self.weights[0]  # Finally the offset value is added
        else:
            self.net = data_input  # If the node is an input node, the net value is simply the input
        self.compute_output()

    def compute_output(self):

        if self.is_input:
            self.output_value = self.net
        elif self.net >= 10:
            self.output_value = 1
        elif self.net <= -10:
            self.output_value = 0
        else:
            self.output_value = 1 / (1 + math.e ** (-self.net))

    def compute_derivative_of_output(self):
        return self.output_value * (1 - self.output_value)

    def calculate_derivative_of_error_function(self, desired_value):
        return desired_value - self.output_value

    def change_weights(self, desired_value, learning_coefficient):
        if self.is_output:
            if self.row == desired_value:
                desired_value = 1
            else:
                desired_value = 0

            self.old_weights = self.weights[:]

            self.delta_rule = self.compute_derivative_of_output() * self.calculate_derivative_of_error_function(
                desired_value)

            change_in_weights = [
                x.output_value * self.delta_rule * learning_coefficient + 0.01 * self.old_weight_change[placeholder] for
                placeholder, x in enumerate(self.inputs)]
            self.old_weight_change = change_in_weights[:]

            self.weights = [self.weights[0]] + [sum(x) for x in zip(self.weights[1:], change_in_weights)]

            self.weights[0] += self.delta_rule * learning_coefficient

        else:
            self.old_weights = self.weights[:]

            propagation_value = sum([x.old_weights[self.row + 1] * x.delta_rule for x in self.connections[1:]])
            self.delta_rule = self.compute_derivative_of_output() * propagation_value

            change_in_weights = [
                x.net * self.delta_rule * learning_coefficient + 0.01 * self.old_weight_change[placeholder] for
                placeholder, x in enumerate(self.inputs)]
            self.weights = [self.weights[0]] + [sum(x) for x in zip(self.weights[1:], change_in_weights)]

            self.weights[0] += self.delta_rule * learning_coefficient

##
def network_creator(node_list, structure_list):
    for layer, number_of_nodes in enumerate(structure_list):
        if layer == 0:
            for row in range(number_of_nodes):
                node_list.append(Node(row, layer, is_input=True))
        elif layer == len(structure_list) - 1:
            for row in range(number_of_nodes):
                node_list.append(Node(row, layer, is_input=False, is_output=True))
        else:
            for row in range(number_of_nodes):
                node_list.append(Node(row, layer))
##

##
def connect_all_nodes(nodes, network__structure):
    for layer_no, layer in enumerate(network__structure[:-1]):
        for output_node in nodes[0:layer]:
            node_up_to = sum(network__structure[0:(layer_no + 1)])
            for input_node in nodes[node_up_to:node_up_to + network__structure[layer_no + 1]]:
                output_node.add_connections(input_node.row, input_node.layer, nodes, network__structure)
##


def generate_weights(nodes, reset=False):
    for node in nodes:
        if node.layer != 0:
            for inputs in range(len(node.inputs) + 1):
                if reset:
                    node.weights[inputs] = random.gauss(0, len(node.inputs) ** -0.5)
                else:
                    node.weights.append(random.gauss(0, len(node.inputs) ** -0.5))
            node.weights[0] = 0


def calculate_total_loss(nodes_list):
    final_layer = nodes_list[len(nodes_list) - 1].layer
    for node in nodes_list:
        if node.layer == final_layer:
            pass


def calculate_loss(expected_output, real_output):
    return math.fabs(expected_output - real_output)


def write_weights_to_file(nodes, data_list, name):
    file = open(name + ".txt", "w")
    file.write(str(data_list) + "\n")
    for node_layer in range(nodes[len(nodes) - 1].layer):
        line = ""
        for x in nodes:
            if x.layer == (node_layer + 1):
                line += str(x.weights) + ":"
        line = line[:-1]
        file.write(line + "\n")


def save_user_defined_file(nodes, data_list):
    user_input = input("Would you like to save this model? Y or N \n --> ")
    if user_input.lower() in ["y", "true", "yes", "sure", "i would", "okay", "es"]:
        name = input("Please enter the desired name of your file. (Without .txt)\n --> ")
        while (name + ".txt") in os.listdir():
            user_input = input(
                "Sorry theres another file with that name saved already. Would you like to try another? Y or N\n --> ")
            if user_input.lower() in ["y", "true", "yes", "sure", "i would", "okay", "es"]:
                name = input("Please enter the desired name of your file. (Without .txt)\n --> ")
            else:
                break
        write_weights_to_file(nodes, data_list, name)


def run_again():
    user_input = input("Would you like to run the code again? T OR N\n" + " --> ")
    if user_input.lower() in ["true", "yes", "sure", "i would", "okay", "es", "", "y"]:
        return True
    else:
        return False


def get_no_of_iterations(iterations):
    print("Default Iterations = " + str(iterations))
    user_input = input("How many Iterations would you like, press enter for default.\n --> ").lower()
    while True:
        try:
            user_input = int(user_input)
            if user_input > 0:
                print("Running " + str(user_input) + " iterations")
                return user_input
            else:
                user_input = input("We must have more than 0 iterations!\n --> ")
        except:
            print("Running " + str(iterations) + " iterations")
            return iterations


def set_weights(nodes, weight_data):
    for node in nodes:
        if node.layer != 0:
            node.weights = weight_data[node.layer - 1][node.row]


def get_file_data(input_file_name):
    file = open(input_file_name + ".txt", "r")
    file_line_list = file.read().split("\n")
    desired_data = eval(file_line_list[0])
    layers = []
    network_structure = [len(desired_data[0][0])]

    for line in file_line_list[1:-1]:
        layers.append(line.split(":"))

    for layer in layers:
        network_structure.append(len(layer))
        for node_no, node in enumerate(layer):
            node = layer[node_no] = eval(node)
            for weight_number, weight in enumerate(node):
                node[weight_number] = float(weight)

    return network_structure, desired_data, layers


def runner(nodes_list, number_of_iterations, data_list, learning_coefficient=0.25):
    iterations_completed = 0
    print("[", end="")
    for i in range(number_of_iterations * len(data_list)):
        if i % len(data_list) == 0:
            print(100*round(i/(number_of_iterations * len(data_list)), 2), "Percent Complete")
            learning_coefficient = learning_coefficient * 0.9
            iterations_completed += 1
        desired_value = data_list[i % len(data_list)][1]
        for node in nodes_list:
            if node.is_input:
                node.calculate_net(data_list[i % len(data_list)][0][node.row])
            else:
                node.calculate_net()

        for node in reversed(nodes_list):
            if not node.is_input:
                node.change_weights(desired_value, learning_coefficient)
            else:
                break
    print("100 Percent Complete")
    print("]", end="\n")


def show(nodes_list, data_list, final_layer):
    total_loss = 0
    good_guesses = 0
    for i in range(len(data_list)):
        desired_value = data_list[i % len(data_list)][1]
        for node in nodes_list:
            if node.is_input:
                node.calculate_net(data_list[i % len(data_list)][0][node.row])
            else:
                node.calculate_net()
        for node in nodes_list:
            last_layer = [x.output_value for x in nodes_list[-10:]]
            if node.layer == final_layer:
                print(node.output_value, desired_value)
                if node.row != desired_value:
                    total_loss += node.output_value
                else:
                    total_loss -= node.output_value
                    if node.output_value == max(last_layer):
                        good_guesses += 1
    total = 0
    for data_point in data_list:
        total += data_point[1]
    print("Total Loss =", total_loss, " Percentage Loss =", (total_loss / total) * 100)
    print("We had", good_guesses, "good guesses or a " + str(good_guesses / (len(data_list)) * 100) + "% success rate")
    if total_loss / total > 0.5:
        return True


def unpack_mnist():
    mndata = MNIST('C:\\Users\\magee\\Desktop\\NEA-20190405T164125Z-001\\Method\\MNIST')

    images, labels = mndata.load_training()

    data_list = []
    number_of_images = 1000

    ##        for i in range(number_of_images):
    ##            print(MNIST.display(images[i]))

    for i in range(number_of_images):
        mean = statistics.mean(images[i])
        std_dev = statistics.stdev(images[i])
        for placeholder, data_point in enumerate(images[i]):
            images[i][placeholder] = (data_point - mean) / std_dev

    for i in range(number_of_images):
        data_list.append([images[i], labels[i]])

    return data_list


def get_user_input(nodes):
    file_name = input("Please enter the desired name of your file. (Without .txt)\n --> ")

    while (file_name + ".txt") not in os.listdir():
        user_input = input("Sorry that file isn't in this directory. Would you like to try another file? Y or N\n --> ")
        if user_input.lower() in ["y", "true", "yes", "sure", "i would", "okay", "es"]:
            file_name = input("Please enter the desired name of your file. (Without .txt)\n --> ")
        else:
            break

    data_file = open(file_name + ".txt", "r")
    file_data = data_file.read().split(",")
    file_data = list(map(float, file_data))
    mean = statistics.mean(file_data)
    std_dev = statistics.stdev(file_data)
    for placeholder, data_point in enumerate(file_data):
        file_data[placeholder] = (data_point - mean) / std_dev

    show_output_from_input(nodes, file_data)


def show_output_from_input(nodes_list, data_input):
    guess = 0
    for i in range(len(data_input)):
        for node in nodes_list:
            if node.is_input:
                node.calculate_net(data_input[node.row])
            else:
                node.calculate_net()
    largest_output = max([x.output_value for x in nodes_list[-10:]])
    for node in nodes_list:
        if node.layer == nodes_list[len(nodes_list) - 1].layer:
            print("Output value:", node.output_value)
            if node.output_value == largest_output:
                guess = node.row
    print("My guess is", guess)


def initialise(network__structure=None, data_list=None, print_time=True, get_network=True):
    if network__structure is None:  # If no network structure is given
        network__structure = [784, 300, 10]  # load a default
    if data_list is None:  # If no data is given
        data_list = unpack_mnist()  # Load the MNIST database

    print(network__structure)
    nodes = []
    network_creator(nodes, network__structure)  # Creats instances of node class
    connect_all_nodes(nodes, network__structure)  # Connects all node layers
    for node in nodes:
        node.create_old_changes()

    if get_network:  # If the user wants to load a already created network
        try:
            network__structure, desired_data, layers = get_file_data(input(
                "Please enter the name of the file to be used as input (Without .txt )\n --> "))  # Try and find it
            set_weights(nodes, layers)  # Try and initialize the weights
        except:
            print("An error has occured")  # Else load default data
            generate_weights(nodes)

    default_learning_coefficient = 0.025  # Default learning coefficient, changes speed of learning
    total_iterations = 0  # Keeps track of the total iterations computed

    iterations = 10  # Default no of iterations
    iterations = get_no_of_iterations(iterations)  # Get user input

    start = time.time()  # Start the timer (After user input)
    runner(nodes, iterations, data_list, default_learning_coefficient)  # Initialises learning
    total_iterations += iterations  # Adds to the total iterations computed

    time_taken = time.time() - start  # Saves the time taken

    show(nodes, data_list, nodes[len(nodes) - 1].layer)  # Show output values for specific inputs

    if print_time:
        print("Time taken:", time_taken)  # Print the time taken

    while input("Would you like to test a value on this model? Y or N\n --> ").lower() in ["y", "true", "yes", "sure",
                                                                                           "i would", "okay", "es"]:
        get_user_input(nodes)

    while run_again():  #

        current_start = time.time()  # Starting a second timer again after user input
        runner(nodes, iterations, data_list, default_learning_coefficient)
        total_iterations += iterations
        print("A total of " + str(total_iterations) + " iterations have been computed.")
        show(nodes, data_list, nodes[len(nodes) - 1].layer)
        if print_time:
            print("Total Time taken:", time.time() - start, "Time for current iteration:", time.time() - current_start)        

    write_weights_to_file(nodes, data_list, "weights")
    file = open("weights", "w")
    file.write(str(default_learning_coefficient))
    file.close()
    save_user_defined_file(nodes, data_list)
    print("Thank You!")


if __name__ == "__main__":
    initialise()
    # cProfile.run('initialise()', sort='cumulative')
