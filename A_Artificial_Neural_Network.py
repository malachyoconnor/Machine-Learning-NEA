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
        if not self.is_input:  # creates a list of weight changes with all zeroes
            self.old_weight_change = [0 for x in range(len(self.inputs))]  # so we have a list of weight changes when the first
            # momentum calculation is done

    def add_connections(self, desired_row, desired_layer, node_list, network__structure):
        if not (desired_row == self.row and desired_layer == self.layer):  # if not searching for itself
            node_id = sum(network__structure[0:desired_layer]) + desired_row  # find the node to connect to
            desired_node = node_list[node_id]  # get the node in the node list

            if desired_node.row == desired_row and desired_node.layer == desired_layer:  # check that was the node we're looking for
                self.connections.append(desired_node)  # add the node to self.connections
                desired_node.inputs.append(self)  # adds itself to the nodes list of inputs
                return None  # exit

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

    def compute_output(self):  # finds the value of our activation function

        if self.is_input:
            self.output_value = self.net  # our input node's output values should be their net
        else:
            self.output_value = max(0, self.net) + math.cos(self.net)

    def compute_derivative_of_output(self):
        if self.net < 0:
            return -math.sin(self.net)
        else:
            return 1 - math.sin(self.net)

    def calculate_derivative_of_error_function(self, desired_value):

        return desired_value - self.output_value

    def change_weights(self, desired_value, learning_coefficient):
        if self.is_output:
            if self.row == desired_value:  # checks if the node is the one we want to fire
                desired_value = 0.98  # if so we want a high output
            else:
                desired_value = -0.98

            self.old_weights = self.weights[:]  # copies old weights list

            self.delta_rule = self.compute_derivative_of_output() * self.calculate_derivative_of_error_function(
                desired_value)  # computes our delta rule

            change_in_weights = [
                x.output_value * self.delta_rule * learning_coefficient + 0.02 * self.old_weight_change[placeholder] for
                placeholder, x in enumerate(self.inputs)]  # finds our delta rule (using momentum)

            self.old_weight_change = change_in_weights[:]  # saves change in weights for use in momentum

            self.weights = [self.weights[0]] + [sum(x) for x in zip(self.weights[1:], change_in_weights)]

            self.weights[0] += self.delta_rule * learning_coefficient

        else:
            self.old_weights = self.weights[:]

            propagation_value = sum([x.old_weights[self.row + 1] * x.delta_rule for x in self.connections[1:]])
            self.delta_rule = self.compute_derivative_of_output() * propagation_value

            change_in_weights = [
                x.net * self.delta_rule * learning_coefficient + 0.02 * self.old_weight_change[placeholder] for
                placeholder, x in enumerate(self.inputs)]  # uses momentum
            self.weights = [self.weights[0]] + [sum(x) for x in zip(self.weights[1:], change_in_weights)]

            self.weights[0] += self.delta_rule * learning_coefficient


def network_creator(node_list, structure_list):
    for layer, number_of_nodes in enumerate(structure_list):  # iterates through our list of nodes
        if layer == 0:  # if an input node
            for row in range(number_of_nodes):
                node_list.append(Node(row, layer, is_input=True))  # we want to initialise with is_input set to True
        elif layer == len(structure_list) - 1:  # if an output node
            for row in range(number_of_nodes):
                node_list.append(Node(row, layer, is_input=False,
                                      is_output=True))  # we want to initialise with is_output set to True
        else:
            for row in range(number_of_nodes):
                node_list.append(Node(row, layer))  # otherwise initialize as normal
    return node_list                


def connect_all_nodes(nodes, network__structure):
    for layer_no, layer in enumerate(network__structure[:-1]):  # iterates through our network structure
        for output_node in nodes[0:layer]:  # iterates through all the nodes in the layer

            node_up_to = sum(network__structure[0:(layer_no + 1)])  # gets the number of nodes before the layer

            for input_node in nodes[node_up_to:node_up_to + network__structure[layer_no + 1]]:
                output_node.add_connections(input_node.row, input_node.layer, nodes, network__structure)  # connects all the nodes
    return nodes    

def generate_weights(nodes, reset=False):
    for node in nodes:
        if node.layer != 0:  # if not an input
            for inputs in range(len(node.inputs) + 1):
                if reset:
                    # generates weights as according to Efficient BackProp, to improve convergence speed
                    node.weights[inputs] = random.gauss(0, len(node.inputs) ** -0.5)
                else:
                    node.weights.append(random.gauss(0, len(node.inputs) ** -0.5))
            node.weights[0] = 0


def write_weights_to_file(nodes, data_list, network_structure, name):
    file = open(name + ".txt", "w")  # opens the specified file
    file.write("NO_DATA\n")  # writes our data list to the file
    for node_layer in range(nodes[len(nodes) - 1].layer):  # gets the number of nodes and iterates that many times
        line = ""
        for x in nodes:
            if x.layer == (node_layer + 1):  # finds nodes of specific layers (dont want input nodes)
                line += str(x.weights) + ":"  # writes adds their weights to a string
        line = line[:-1]  # removes the final : from the string
        file.write(line + "\n")  # writes the string to the file, and adds a new line
    file.write(str(network_structure)[1:-1])


def save_user_defined_file(nodes, data_list, network_structure):
    user_input = input("Would you like to save this model? Y or N \n --> ")
    if user_input.lower() in ["y", "true", "yes", "sure", "i would", "okay", "es"]:
        name = input("Please enter the desired name of your file. (Without .txt)\n --> ")
        while (name + ".txt") in os.listdir():  # searches the directory for the user defined file name
            user_input = input(
                "Sorry theres another file with that name saved already. Would you like to try another? Y or N\n --> ")
            if user_input.lower() in ["y", "true", "yes", "sure", "i would", "okay", "es"]:
                name = input("Please enter the desired name of your file. (Without .txt)\n --> ")
            else:
                break
        write_weights_to_file(nodes, data_list, network_structure, name)


def run_again():
    user_input = input("Would you like to run the code again? Press enter if you would\n" + " --> ")  # get user input
    if user_input.lower() in ["true", "yes", "sure", "i would", "okay", "es", "",
                              "y"]:  # if the user would like to run again
        return True  # return true
    else:
        return False  # otherwise return false


def get_no_of_iterations(iterations, nodes_list, data_list):
    print("Default Iterations = " + str(iterations))
    user_input = input("How many Iterations would you like, press enter for default.\n --> ").lower()
    while True:
        try:  # try to turn user in put to an integer
            user_input = int(user_input)
            if user_input > 0:
                print("Running " + str(user_input) + " iterations")
                return user_input
            elif user_input == 0:
                show(nodes_list, data_list)  # if the user selected zero iterations, show output from the current model
                return 0
            else:
                print("Input must be greater than zero!")
                user_input = input("How many Iterations would you like, press enter for default.\n --> ")
        except:
            print("Running " + str(
                iterations) + " iterations")  # if the suer gave bad input, run default number of iterations
            return iterations


def set_weights(nodes, weight_data):
    for node in nodes:
        if node.layer != 0:  # if the node isn't an input node
            node.weights = weight_data[node.layer - 1][node.row]  # set the weights to the data supplied


def get_file_data(input_file_name):
    file = open(input_file_name + ".txt", "r")  # open the user defined file
    file_line_list = file.read().split("\n")  # read the file
    layers = []
    network_structure = [int(file_line_list[len(file_line_list)-1][0])]  # stores the number of input nodes

    for line in file_line_list[1:-1]:  # removes the brackets at the end of every list
        layers.append(line.split(":"))

    for layer in layers:
        network_structure.append(len(layer))
        for node_no, node in enumerate(layer):
            node = layer[node_no] = eval(node)
            for weight_number, weight in enumerate(node):
                node[weight_number] = float(weight)

    return network_structure, layers


def runner(nodes_list, number_of_iterations, data_list, learning_coefficient=0.25):
    iterations_completed = 0
    print("[", end="")  # Starts the loading bar
    for i in range(number_of_iterations * len(data_list)):  # iterates through all the values in the data list
        if i % len(data_list) == 0:
            print(round(100 * i / (number_of_iterations * len(data_list)), 2),
                  "Percent Complete")  # gives the percentage completed
            learning_coefficient = learning_coefficient * 0.9 # decreases the learning coefficient for faster convergence
            iterations_completed += 1
        desired_value = data_list[i % len(data_list)][
            1]  # loads the desired values (the desired values is always the second
        # value in a data point) i.e: [data, label]
        for node in nodes_list:  # iterates through all the nodes
            if node.is_input:
                node.calculate_net(data_list[i % len(data_list)][0][node.row])  # supplies input nodes with their data
            else:
                node.calculate_net()  # supplies non-input node with no data

        for node in reversed(
                nodes_list):  # propagating backwards through the list of nodes (as defined by backpropagation formula)
            if not node.is_input:
                node.change_weights(desired_value,
                                    learning_coefficient)  # runs change_weights starting with output nodes
            else:
                break  # when we reach an input nodes, we've iterated through all the non-input nodes and can exit the loop
    print("100 Percent Complete")
    print("]", end="\n")  # ends the loading bar


def show(nodes_list, data_list):
    final_layer = nodes_list[len(nodes_list) - 1].layer  # gets the final layer of the neural network
    good_guesses = 0
    for placeholder in range(len(data_list)):  # iterates through all the values in the list
        desired_value = data_list[placeholder % len(data_list)][1]  # gets the desired output for the values in the list
        for node in nodes_list:  # propagates forward to calculate error at output
            if node.is_input:
                node.calculate_net(
                    data_list[placeholder % len(data_list)][0][node.row])  # gives the input nodes their input
            else:
                node.calculate_net()

        for node in nodes_list[-10:]:  # iterates through output nodes
            last_layer_outputs = [node.output_value for node in
                                  nodes_list[-10:]]  # gets the outputs of all the output nodes
            if node.layer == final_layer:  # if the node is an output node
                print(node.output_value, desired_value)  # show the user it's output
                if node.row == desired_value:
                    correct_node_output = node.output_value  # show the output of the node we want to be firing
                    if node.output_value == max(last_layer_outputs):
                        good_guesses += 1  # if the right node fired, add to good_guesses

        highest_output_index = last_layer_outputs.index(
            max(last_layer_outputs))  # the index of the node with the highest output
        print("The correct node ouputted", correct_node_output, "and the model predicted", highest_output_index,
              end=" ")
        if highest_output_index == desired_value:
            print("which was the correct value", end="\n")
        else:
            print("which was the incorrect value", end="\n")

    print("We had", good_guesses, "good guesses or a " + str(good_guesses / (len(data_list)) * 100) + "% success rate")
    if good_guesses / (len(data_list)) > 0.8:
        return True


def unpack_mnist(number_of_images=100):
    mndata = MNIST('.\\dataset')

    images, labels = mndata.load_training()

    data_list = []

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

    while (file_name + ".txt") not in os.listdir():  # searches the directory for the file the user entered
        #   if the file isn't found, ask the user for input again
        user_input = input("Sorry that file isn't in this directory. Would you like to try another file? Y or N\n --> ")
        if user_input.lower() in ["y", "true", "yes", "sure", "i would", "okay", "es"]:  # if they would
            file_name = input("Please enter the desired name of your file. (Without .txt)\n --> ")  # try again
        else:
            return False  # otherwise return false

    try:  # catch any errors
        data_file = open(file_name + ".txt", "r")  # open their file
        file_data = data_file.read().split(",")  # split the file according to commas
        # this is how data from the drawing tool is formatted
        file_data = list(map(float, file_data))  # converts all the data to floats
        mean = statistics.mean(file_data)  # gets the mean of the input data
        std_dev = statistics.stdev(file_data)  # gets the standard deviation of the input data
        for placeholder, data_point in enumerate(file_data):
            file_data[placeholder] = (data_point - mean) / std_dev  # normalizes the data
    except:
        print("That file type isn't supported, data must be seperated by commas")
        return True

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


def initialise(network__structure=None, data_list=None, print_time=True, get_network=False):
    if network__structure is None:  # If no network structure is given
        network__structure = [768, 400, 400, 10]  # load a default
    if data_list is None:  # If no data is given
        pattern_number = 100
        print("Loading",pattern_number, "images")
        data_list = unpack_mnist(pattern_number)  # Load the MNIST database

    print(network__structure)
    nodes = []
    nodes = network_creator(nodes, network__structure)  # Creates instances of node class
    nodes = connect_all_nodes(nodes, network__structure)  # Connects all node layers

    # !!!
    for node in nodes:
        node.create_old_changes()  # creates a list of old weight changes for use in momentum

    if get_network:  # If the user wants to load a already created network (variable defined at runtime)
        while True:
            try:
                network__structure, layers = get_file_data(input("Please enter the name of the file to be used as input (Without .txt )\n --> "))  # Try and find it
                set_weights(nodes, layers)  # Try and initialize the weights
                test_list = unpack_mnist(pattern_number)
                show(nodes, data_list)
                break
            except:
                user_input = input("An error has occured, would you like to reenter the name Y or N \n--> ")  # Else load default data
                if user_input.lower() in ["y", "true", "yes", "sure", "i would", "okay", "es"]:
                    continue
                else:
                    generate_weights(nodes)
                    break
    else:
        generate_weights(nodes)

    default_learning_coefficient = 0.002  # Default learning coefficient, changes speed of learning 0.0033 is a good one
    total_iterations = 0  # Keeps track of the total iterations computed
    # !!!

    iterations = 10  # Default no of iterations
    iterations = get_no_of_iterations(iterations, nodes, data_list)  # Get user input
    
    start = time.time()  # Start the timer (After user input)
    runner(nodes, iterations, data_list, default_learning_coefficient)  # Initialises learning
    total_iterations += iterations  # Adds to the total iterations computed

    time_taken = time.time() - start  # Saves the time taken
    file = open("timings.txt", "a")
    file.write("\n" + str(total_iterations) + " iterations of " + str(pattern_number) + " images, completed in " + str(time_taken) + " seconds")
    file.close()

    show(nodes, data_list)  # Show output values for specific inputs

    if print_time:
        print("Time taken:", time_taken)  # Print the time taken

    while input("Would you like to test a value on this model? Y or N\n --> ").lower() in ["y", "true", "yes", "sure",
                                                                                           "i would", "okay", "es"]:
        if not get_user_input(nodes):  # if the get_user_input function returns false
            break  # exit the loop

    while run_again():

        iterations = get_no_of_iterations(iterations, nodes, data_list)
        current_start = time.time()  # Starting a second timer again after user input
        runner(nodes, iterations, data_list, default_learning_coefficient)
        total_iterations += iterations
        print("A total of " + str(total_iterations) + " iterations have been computed.")
        show(nodes, data_list)
        if print_time:  # print the time taken, print_time defined at run time
            print("Total Time taken:", time.time() - start, "Time for current iteration:", time.time() - current_start)
        while input("Would you like to test a value on this model? Y or N\n --> ").lower() in ["y", "true", "yes", "sure",
                                                                                           "i would", "okay", "es"]:
            if not get_user_input(nodes):  # if the get_user_input function returns false
                break  # exit the loop

    write_weights_to_file(nodes, data_list, network__structure,"weights")  # writes an autosave of the model to a file
    file = open("weights", "w")
    file.close()
    save_user_defined_file(nodes, data_list, network__structure)
    print("Thank You!")


if __name__ == "__main__":
    initialise()
    # cProfile.run('initialise()', sort='cumulative')
