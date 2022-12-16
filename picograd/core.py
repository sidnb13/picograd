'''Core autodiff functionality of datatypes (variables, tensors) and computational graphs.'''

from __future__ import annotations
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

class Variable:
    def __init__(self, value, requires_grad=True, parent_nodes=[], fn_str=None) -> None:
        '''
        Initialize a Variable superclass, representing a Variable in the computational graph.
        '''
        # Unwrap if needed, we should be dealing with primitives
        self.value = value
        self.parent_nodes = parent_nodes
        self.fn_str=fn_str
        # Gradient parameters
        self.requires_grad = requires_grad
        self.grad = 0
        self.grad_fn = lambda: None


    def toposort(self):
        # Get the number of times a Variable appears as a parent Variable.
        counts = defaultdict(int)
        count_stack = [self]  # correspond to loss fn
        while count_stack:
            node = count_stack.pop()
            # add parent Variables to stack
            # how many times does this Variable appear as a parent Variable
            if node in counts:
                counts[node] += 1
            else:
                counts[node] = 1
                count_stack.extend(node.parent_nodes)

        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            for p in node.parent_nodes:
                # if all children have been visited, add to stack
                if counts[p] == 1:
                    stack.append(p)
                else:
                    counts[p] -= 1

    def backward(self):
        '''Compute and update the gradient of this variable with respect to all other variables.'''

        if not self.requires_grad:
            raise Exception(f'{self} does not track grad')

        # root gradient is 1
        self.grad = 1

        for node in self.toposort():
            node.grad_fn()

    def wrap_primitive(value: int | float):
        return Variable(value, requires_grad=False)
    
    def __add__(self, other: Variable | int | float):
        out = Variable(self.value + other.value, requires_grad=self.requires_grad, parent_nodes=[self, other], fn_str='+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out.grad_fn = _backward
        return out
    
    def __mul__(self, other: Variable | int | float):
        out = Variable(self.value * other.value, requires_grad=self.requires_grad, parent_nodes=[self, other], fn_str='*')
        
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        
        out.grad_fn = _backward
        return out

    def __pow__(self, other: int | float):
        assert isinstance(other, (int, float)), 'Exponent must be an integer or float'
        out = Variable(self.value ** other, requires_grad=self.requires_grad, parent_nodes=[self], fn_str='**')
        
        def _backward():
            self.grad += other * self.value ** (other - 1) * out.grad
            
        out.grad_fn = _backward
        return out
    
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    
    def __repr__(self) -> str:
        return f'Variable({self.value}, leaf={len(self.parent_nodes) == 0}, requires_grad={self.requires_grad})'


class DAGVisualization:
    def __init__(self, end_node: Variable) -> None:
        self.end_node = end_node
        self.graph = nx.DiGraph()

    def add_node(self, node: Variable):
        self.graph.add_node(f'Variable({node.value}, op={node.fn_str})')

    def add_edge(self, parent: Variable, child: Variable):
        self.graph.add_edge(
            f'Variable({parent.value})', f'Variable({child.value})')

    def create_nx_graph(self):
        def construct_graph(start_node: Variable):
            for parent in start_node.parent_nodes:
                self.add_edge(parent, start_node)
                construct_graph(parent)
        construct_graph(self.end_node)
