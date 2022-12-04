'''
Visualize the computational graph of a function.
'''

import networkx as nx
import matplotlib.pyplot as plt
from . import core as anp

class DAGVisualization:
    def __init__(self, end_node: anp.Variable) -> None:
        self.end_node = end_node
        self.graph = nx.DiGraph()
        
    def add_node(self, node: anp.Variable):
        self.graph.add_node(f'Variable({node.value})')
        
    def add_edge(self, parent: anp.Variable, child: anp.Variable):
        self.graph.add_edge(f'Variable({parent.value})', f'Variable({child.value})')
        
    def create_nx_graph(self):
        def construct_graph(start_node: anp.Variable):
            for parent in start_node.parent_nodes:
                self.add_edge(parent, start_node)
                construct_graph(parent)
        construct_graph(self.end_node)
        print(self.graph.nodes)