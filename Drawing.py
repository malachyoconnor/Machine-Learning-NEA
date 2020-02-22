import tkinter as tk
from tkinter import messagebox
import random
import math
import inspect

class App(tk.Tk):
    
    def __init__(self):
        tk.Tk.__init__(self)   # inherits tkinter class
        self.canvas = tk.Canvas(self, width=280, height=280, borderwidth=0, highlightthickness=0)   # creates a tkinter canvas
        self.canvas.pack()  
        self.rows = 28
        self.columns = 28
        self.cellwidth = 10
        self.cellheight = 10
        
        self.rect = {}   # creates a dictionary of all our squares
   
        def get_square(event):
            if 0 <= event.x < 280 and 0 <= event.y < 280:   #if the cursor has clicked a spot in the grid
                row_clicked = math.floor(event.x/10)   # finds the row of the rectangle clicked
                column_clicked = math.floor(event.y/10)   # finds the column of the rectangle clicked
                item_id = self.rect[column_clicked, row_clicked] # finds the rectangle the row & column refers to
                if event.num == 3 or event.state == 1032:   # if the event is a single right click or a dragged right click 
                    if 0 < row_clicked < 28 and 0 < column_clicked < 28:   # if the cursor is not on the edge, erase in a larger shape
                        paint_square(self.rect[column_clicked+1, row_clicked], colour="white")
                        paint_square(self.rect[column_clicked-1, row_clicked], colour="white")
                        paint_square(self.rect[column_clicked,  row_clicked+1], colour="white")
                        paint_square(self.rect[column_clicked,  row_clicked-1], colour="white")
                    paint_square(item_id, colour="white")
                else:
                    paint_square(item_id)
                    paint_square(self.rect[column_clicked+1, row_clicked], colour="black")
                    paint_square(self.rect[column_clicked-1, row_clicked], colour="black")
                    paint_square(self.rect[column_clicked,  row_clicked+1], colour="black")
                    paint_square(self.rect[column_clicked,  row_clicked-1], colour="black")
                
        def paint_square(item_id, colour="black"):
            self.canvas.itemconfig(item_id, fill=colour)   # Change the colour of whatever square is necessary

        def clear_canvas(event):
            print("Canvas Cleared")
            self.canvas.itemconfig('rect', fill="white")    # Changes the colour of every square on screen
        
        self.canvas.bind("<Button-1>", get_square)
        self.canvas.bind("<B1-Motion>", get_square)
        self.canvas.bind("<Button-3>", get_square)
        self.canvas.bind("<B3-Motion>", get_square)
        self.canvas.bind("<c>", clear_canvas)
        
        for column in range(28):
            for row in range(28):
                x1 = column*self.cellwidth # Defines the x value of the top-left corner of our rectangle
                y1 = row * self.cellheight # Likewise for the y value
                x2 = x1 + self.cellwidth # Finds the x value for the bottom-left corner of our rectangle
                y2 = y1 + self.cellheight # Likewise for the y value
                self.rect[row,column] = self.canvas.create_rectangle(x1,y1,x2,y2, fill="white", tags="rect") # Creates a tkinter rectangle object

if __name__ == "__main__":
    
    def save_and_quit():
        if tk.messagebox.askokcancel("Quit", "Do you really wish to quit? This will save your image"):
            output = []
            for square in app.rect:   # iterates through all rectangles
                if (app.canvas.itemcget(app.rect[(square[1], square[0])], "fill")) == "white":   # checks if the square is white
                    output.append(0)
                else:
                    output.append(1)   # if the square isnt black, append 1
            file = open("drawing.txt", "w")    # opens a file
            file.write(str(output)[1:-1])   # saves our list to that file
            file.close()
            app.destroy()   # closes our tkinter window

    app = App()
    app.canvas.focus_set()   # ensures the canvas detects key presses
    app.protocol("WM_DELETE_WINDOW", save_and_quit)
    app.mainloop()
    




