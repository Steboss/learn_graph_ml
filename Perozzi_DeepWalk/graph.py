# Define the graph class
from six import iterkeys
from collections import defaultdict, Iterable
import random


class Graph(defaultdict):
    """Graph Class inheriting defaultdict"""

    def __init__(self):
        super(Graph, self).__init__(list)


    def nodes(self):
        r""" Graph as a dictionary, so nodes are the keys"""
        return self.keys()


    def adjacency_iter(self):
        r""" Define a graph through adjacency list and iterate"""
        return self.iteritems()


    def subgraph(self, nodes={}):
        r""" Subgraph extraction"""
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph


    def make_undirected(self):
        r""" Make a graph undirected"""

        for v in list(self):
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        self.make_consistent()
        return self

    def make_consistent(self):
        r""" Check all the nodes and edges and remove self loops"""
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):
        r""" Remove self loops from Graphs"""
        removed = 0

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def order(self):
        r""" Return the number of nodes in the graph"""
        return len(self)

    def degree(self, nodes=None):
        r""" Function to retrieve the edges in graph"""
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def number_of_edges(self):
        r""" Return number of edges in the graph"""
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        r"""Returns the number of nodes in the graph"""
        return self.order()

    def random_walk(self, path_length, rand=random.Random(), start=None):
        r""" Main function for a graph: perform a  random walk of a defined length
        starting from a given start node

        Parameters
        ----------
        path_length: length of the random walk
        start: start node of the random walk
        """

        #print(f"Calling random walk with path length {path_length}")
        # define the graph
        G = self
        # uniform sampling of nodes within G
        #print("uniform sampling of the starting node for creating random walk")
        path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                # random
                if rand.random() >= 0.0:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break

        return [str(node) for node in path]