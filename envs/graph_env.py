import csv
import numpy as np

from collections import defaultdict
from .discrete_env import categorical_sample, DiscreteEnv

class Node():
    def __init__(self, coord, graph):
        self.coord = tuple(coord)
        self.graph = graph

    def get_neighbors(self):
        """Returns the neighbors of the current node."""
        return self.graph.node_to_neighbors[self]

    def get_padded_neighbors(self):
        """Returns the neighobrs of the current node, appended at the end with
           self-edges to support consistent neighbor lengths."""
        neighbors = self.get_neighbors()
        num_pads = self.graph.max_degree - len(neighbors)
        return neighbors + [self] * num_pads
   
    def __hash__(self):
        return hash((self.coord, self.graph))

    def __eq__(self, other):
        return self.coord == other.coord and self.graph == other.graph

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return self.coord < other.coord

class Graph():
    def __init__(self, edge_coord_list):
        # All the edge data for the graph. Edges are represented in both directions.
        self.node_to_neighbors = defaultdict(list)
        # The maximum degree of any noded in the graph
        self.max_degree = 0
        # The total number of edges in the graph.
        self.total_edges = 0
        # A mapping of (src_node, edge_node)
        self.closed_road_map = {}
        # Add all the specified edges to create the graph.
        for edge in edge_coord_list:
            self._process_edge(edge)

    def _process_edge(self, edge):
        src_lng, src_lat, end_lng, end_lat = edge
        src_coord = (float(src_lng), float(src_lat))
        end_coord = (float(end_lng), float(end_lat))

        src_node = Node(src_coord, self)
        end_node = Node(end_coord, self)

        if end_node not in self.node_to_neighbors[src_node]:
            self.node_to_neighbors[src_node].append(end_node)
            self.node_to_neighbors[end_node].append(src_node)
            self.total_edges += 1

            self.max_degree = max(
                self.max_degree,
                len(self.node_to_neighbors[src_node]),
                len(self.node_to_neighbors[end_node]),
            )

    def get_nodes(self):
        """Returns a list of Nodes (not coords) in the graph."""
        return sorted(self.node_to_neighbors.keys())

    def get_edges(self):
        """Returns a list of pair-wise coordinates representing edges in the graph."""
        nodes = self.get_nodes()
        seen_nodes = set()
        edges = []
        for node in nodes:
            for neighbor in node.get_neighbors():
                if neighbor not in seen_nodes:
                    edges.append((node.coord, neighbor.coord))
            seen_nodes.add(node)
        return edges

    def close_roads_with_prob(self, p):
        """Randomly deletes edges with some probability"""
        # TODO: Some sort of seeding?
        for edge in self.get_edges():
            if np.random.random() <= p:
                self.close_road(edge[0], edge[1])

    def close_road(self, src_coord, end_coord):
        """Deletes an edge between a specified pair of coordinates if present"""
        src_node = Node(src_coord, self)
        end_node = Node(end_coord, self)

        road_i = self.node_to_neighbors[src_node].index(end_node)
        if road_i >= 0:
            self.closed_road_map[(src_node, end_node)] = road_i
            self.node_to_neighbors[src_node][road_i] = src_node
        road_i = self.node_to_neighbors[end_node].index(src_node)
        if road_i >= 0:
            self.closed_road_map[(end_node, src_node)] = road_i
            self.node_to_neighbors[end_node][road_i] = end_node

    def open_road(self, src_coord, end_coord):
        """Adds an edge back if it has previously been deleted"""
        src_node = Node(src_coord, self)
        end_node = Node(end_coord, self)

        if (src_node, end_node) in self.closed_road_map:
            self.node_to_neighbors[src_node][self.closed_road_map[(src_node, end_node)]] = end_node
            del self.closed_road_map[(src_node, end_node)]
        if (end_node, src_node) in self.closed_road_map:
            self.node_to_neighbors[end_node][self.closed_road_map[(end_node, src_node)]] = src_node
            del self.closed_road_map[(end_node, src_node)]


def graphFromCsv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return Graph(list(reader))

MAPS = {
    'SF': graphFromCsv('sf_map.csv'),
}


# TODO: Properly implement GraphEnv
#
# How do we represent custom goal (and therefore reward) for this environment? The goal will be given for any
# arbitrary tile. Same for initial state...
class GraphEnv(DiscreteEnv):
    """
    Stuff
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, graph, seed=None):
        self.graph = graph

        num_states = len(graph.get_nodes())
        num_actions = graph.max_degree

        transitions = {}
        for node in graph.get_nodes():
            self.transitions[node] = {}
            for i, neighbor in enumerate(node.get_padded_neighbors()):
                self.transitions[node][i] = [(1, neighbor, 0, 0)]

        initial_state_distribution = [1 / num_states] * num_states

        super(GraphEnv, self).__init__(num_states, num_actions, transitions, initial_state_distribution, seed=seed)

    # TODO: Set Reward/Done in initialization and/or in reset
