import tkinter
import time

from decorators import timer


class Grid():

    colors = [
        "white",
        "sky blue",
        "light grey",
        "dark green",
        "light green",
        "brown",
        "red",
        "green",
        "dark grey",
        "salmon",
        "purple"
    ]

    def __init__(self, title, initial_grid=None, speed=0.2, window_width=300, window_height=300):
        self.master = tkinter.Tk() # type: tkinter.Tk
        self.master.title(title)
        self.canvas_height = window_height
        self.canvas_width = window_width
        if initial_grid: self.auto_scale(initial_grid)
        self.speed = speed

    def auto_scale(self, grid):
        """ Sets the scale so that all squares drawn are perfect squares. """
        scale = self.canvas_height / len(grid)
        self.canvas_width = int((len(grid[0]) - 1) * (scale))
        self.master.geometry('{}x{}'.format(self.canvas_width, self.canvas_height)) # Size of window.

    def clear(self):
        for widget in self.master.winfo_children():
            widget.destroy()

    @timer("Grid update")
    def update(self, grid):
        self.clear()
        canvas = tkinter.Canvas(self.master, width=self.canvas_width, height=self.canvas_height)
        canvas.pack()

        # Dimensions for each of the cells:
        cell_height = self.canvas_height / len(grid)
        cell_width  = (self.canvas_width) / len(grid[0])

        for cellx, list in enumerate(grid):
            for celly in range(len(list)):
                # Setting dimensions:
                x = (celly*cell_width)
                y = (cellx*cell_height)

                # Setting color:
                color = self.colors[grid[cellx][celly]+1]

                # Actually drawing:
                canvas.create_rectangle(x, y, x+cell_width, y+cell_height, fill=color, width=0)

        self.canvas = canvas
        self.master.update()

        if self.speed > 0:
            time.sleep(self.speed)
