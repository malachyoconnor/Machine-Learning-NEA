import tkinter as tk
from random import randint

def __init__():
    
    layers = []   
    file_length = sum(1 for x in open("weights.txt", "r"))   # get the file length
    file = open("weights.txt", "r")   # open the file containing all the data of our neural network

    for line_number, line in enumerate(file):  
        if line_number != 0 and line_number != file_length-1:   # if we aren't on the first or last line
            print(line)
            print(line.rstrip().split(":"))   
            layers.append(line.rstrip().split(":"))   # rstrip removes any empty space in the string (if the user has entered some)
                                                      # and then splits string of weights into individual lists
        elif line_number == file_length-1:   # if we are on the last line
            arrangement = [int(value) for value in line.split(",")]   # as the network structure is written to the last line of the file
                                                                      # turn all the values in the network stru

    layers.insert(0, [" INPUT " for x in range(arrangement[0])])   # inserts empty strings to be converted to input

    for layer in layers:
        for placeholder, weights in enumerate(layer):
            layer[placeholder] = weights[1:-1].split(",")
            if layer[placeholder] == ["INPUT"]:
                continue
            else:
                layer[placeholder] = [float(x) for x in layer[placeholder]]   # converts all our weights from strings to floats
    

    root = tk.Tk()   
    width = len(layers)*2*90   # gets width of the window

    top = max(map(len, layers))   # gets the largest row
        
    height = top*70+70   # gets height of window
            
    root.geometry(str(width)+"x"+str(height)+"+200+200")   # sets window height and position

    button_window = tk.Toplevel(root)   # creates secondary window for buttons
    button_window.geometry("100x50+78+200")

    button_window.resizable(False,False)   # makes the button window non-resizable in the x or y plane


    y_scrollbar = tk.Scrollbar(root, orient="vertical")
    y_scrollbar.pack(side="right", fill="y")   # adds the scrollbar to the window

    x_scrollbar = tk.Scrollbar(root, orient="horizontal")
    x_scrollbar.pack(side="bottom", fill="x")

    canvas = tk.Canvas(root, width = width, height =height, scrollregion=(0,0,width,height))   # canvas with a scroll region the width & height
                                                                                               # defined earlier

    canvas.pack()

    canvas.config(yscrollcommand=y_scrollbar.set)   # configuring the scrollbars to scroll
    y_scrollbar.config(command=canvas.yview)

    canvas.config(xscrollcommand=x_scrollbar.set)
    x_scrollbar.config(command=canvas.xview)


    class Nodes:

        def __init__(self, row, node_layer, weights=[], is_input=False, is_output=False):
            self.node_layer = node_layer
            self.row   = row
            self.weights = weights
            self.is_input = is_input
            self.is_output = is_output
            node_layers[self.node_layer].append(self)
            self.node_object = None
            self.centre_x = 0
            self.centre_y = 0
            self.lines = []
            self.text = None

            
        def draw_node(self):
            radius = 25
            self.centre_y = 30 + 2*30*self.row +30*(max(arrangement)-len(node_layers[self.node_layer]))   # sets its centre to be in line with other nodes
            self.centre_x = 30 + 2*100*self.node_layer
            colour = "lightgrey"
            if self.is_input:
                colour = "lightblue"
            elif self.is_output:
                colour = "orange"
            self.node_object = canvas.create_oval(self.centre_x-radius, self.centre_y+radius, self.centre_x+radius, self.centre_y-radius, fill=colour)
            if not self.is_input:
                text = str(round(sum(self.weights)/len(self.weights), 5))   # gets the average weight value
            else:
                text = "INPUT"   
            self.text = canvas.create_text(self.centre_x, self.centre_y, text = text, width = 60)   # writes the text on the node
            

        def hide_node(self):
            canvas.delete(self.node_object)
            canvas.delete(self.text)


        def connect_to_next_layer(self):   # draws lines between each node on each layer
            for node in node_layers[self.node_layer + 1]:
                self.lines.append(canvas.create_line(self.centre_x, self.centre_y, node.centre_x, node.centre_y))
                canvas.tag_lower(self.lines[len(self.lines) - 1])   # moves the line behind the circle


        def hide_connections(self):
            for connection in self.lines:
                canvas.delete(connection)


    node_layers = [[] for x in range(len(layers))]

    for layer_no, layer in enumerate(layers):
        for row_no, node in enumerate(layer):
            if layer_no == 0:
                Nodes(row_no, layer_no, node, is_input=True)   # initializes all our nodes
            elif layer_no == len(layers)-1:
                Nodes(row_no, layer_no, node, is_output=True)
            else:
                Nodes(row_no, layer_no, node)

    def show_nodes():
        show_nodes_button.config(text = "Hide Nodes")
        show_nodes_button.config(command = hide_nodes)
        for layer in node_layers:
            for node in layer:
                node.draw_node()
                
    def hide_nodes():
        show_nodes_button.config(text = "Show Nodes")
        show_nodes_button.config(command = show_nodes)
        for layer in node_layers:
            for node in layer:
                node.hide_node()

    def connect_nodes():
        connect_nodes_button.config(text = "Hide Connections")
        connect_nodes_button.config(command = disconnect_nodes)
        for layer in node_layers:
            for node in layer:
                if not node.is_output:
                    node.connect_to_next_layer()

    def disconnect_nodes():
        connect_nodes_button.config(text = "Show Connections")
        connect_nodes_button.config(command = connect_nodes)
        for layer in node_layers:
            for node in layer:
                node.hide_connections()


    
    show_nodes_button = tk.Button(button_window, text = "Hide nodes", command = hide_nodes)
    connect_nodes_button = tk.Button(button_window, text = "Hide Connections", command = disconnect_nodes)
    show_nodes()
    connect_nodes()

    connect_nodes_button.pack()
    show_nodes_button.pack()
    root.mainloop()

if __name__ == "__main__":
    __init__()