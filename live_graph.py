import matplotlib.pyplot as PLT

from decorators import timer


class LiveGraph():

    def __init__(self, graph_title, x_title="", y_title="", x_range=None, y_range=None):
        self.figure = PLT.figure(figsize=(10, 7.5), dpi=100)
        # self.figure.suptitle(graph_title)
        self.figure.canvas.set_window_title(graph_title)
        PLT.xlabel(x_title)
        PLT.ylabel(y_title)
        if x_range: PLT.xlim([*x_range])
        if y_range: PLT.ylim([*y_range])
        PLT.grid()
        self.actual_graph = None
        self.target_graph = None
        self.figure.show()

    @timer("Graph update")
    def update(self, actuals, targets, upscale=None):
        if actuals: self.actual_graph = self.plot(actuals, self.actual_graph, marker=".", line_style="-")
        if targets: self.target_graph = self.plot(targets, self.target_graph, marker="X", line_style="None")

        self.figure.canvas.draw()
        PLT.pause(0.0000000000000000000001) # little effect on performance.

    def plot(self, histogram, graph, marker=".", line_style="-", invert=False, upscale=None):
        yl, xl = [], []
        for x, y in histogram:
            xl.append(x if not upscale else x*upscale[0])
            yl.append(y if not upscale else x*upscale[1])
        if graph:
            graph.set_xdata(xl)
            graph.set_ydata(yl)
        else: graph = PLT.plot(xl, yl, marker=marker, linestyle=line_style)[0]
        return graph
