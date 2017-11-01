import matplotlib.pyplot as PLT

class LiveGraph():

    def __init__(self, graph_title, x_title="", y_title="", x_limit=5, y_limit=1):
        self.figure = PLT.figure()
        self.figure.suptitle(graph_title)
        PLT.xlabel(x_title)
        PLT.ylabel(y_title)
        PLT.xlim([0, x_limit])
        if y_limit:
            PLT.ylim([0, y_limit])
        PLT.grid()
        self.actual_graph = None
        self.target_graph = None
        self.figure.show()

    def update(self, actuals, targets):
        if actuals: self.actual_graph = self.plot(actuals, self.actual_graph, marker=".")
        if targets: self.target_graph = self.plot(targets, self.target_graph, marker="X", invert=False)

        self.figure.canvas.draw()
        PLT.pause(0.00025)

    def plot(self, histogram, graph, marker=".", invert=False):
        yl, xl = [], []
        for x, y in histogram:
            xl.append(x)
            yl.append(y if not invert else 1-y)
        if graph:
            graph.set_xdata(xl)
            graph.set_ydata(yl)
        else: graph = PLT.plot(xl, yl, marker=marker)[0]
        return graph
