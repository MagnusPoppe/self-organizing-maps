class Edge():
    def __init__(self, distance=0.0, left=None, right=None):
        self.distance = distance
        self.left = left
        self.right = right

    def __str__(self):
        left = str(self.left.city) if self.left else "?"
        right = str(self.right.city) if self.right else "?"
        return "Egde between %s -> %s. distance %f" %(right, left, self.distance)

class TSPNode():


    def __init__(self, city:int, x:float, y:float, edge_prev=None, edge_next=None):
        self.city = city
        self.x = x
        self.y = y

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

    def __str__(self):
        return "City %d at (%f, %f)" %(self.city, self.x, self.y)
