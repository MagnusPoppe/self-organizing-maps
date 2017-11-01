class Edge():
    def __init__(self, distance=0.0, left=None, right=None):
        self.distance = distance
        self.left = left
        self.right = right

class TSPNode():

    def __init__(self, value, edge_prev=None, edge_next=None):
        self.value = value

        # Edges between this and the next + previous node.
        self.set_egde_prev(edge_prev)
        self.set_egde_next(edge_next)

    def set_egde_next(self, edge_next):
        if edge_next:
            edge_next.right = self
            self.edge_next = edge_next
        else: self.edge_next = Edge(right=self)

    def set_egde_prev(self, edge_prev):
        if edge_prev:
            edge_prev.left = self
            self.edge_prev = edge_prev
        else: self.edge_prev = Edge(left=self)