'''Core autodiff functionality of datatypes (variables, tensors) and computational graphs.'''

from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from functools import wraps
from typing import Callable
from types import FunctionType, BuiltinFunctionType
import numpy as _np


class Variable:
    def __init__(self, value, leaf=False, requires_grad=True, parent_nodes=[], fn=None) -> None:
        '''
        Initialize a Variable superclass, representing a Variable in the computational graph.
        '''

        # Unwrap if needed, we should be dealing with np arrays or primitives
        self.value = value.value if isinstance(value, Variable) else value
        self.leaf = leaf
        self.fn = fn
        self.requires_grad = requires_grad
        self.parent_nodes = parent_nodes
        self.grad = None

    def _grad(self):
        raise NotImplementedError('grad not implemented for this Variable')

    def toposort(self):
        def get_node_counts(end_node: Variable):
            '''Get the number of times a Variable appears as a parent Variable.'''
            counts = defaultdict(int)
            stack = [end_node]  # correspond to loss fn
            while stack:
                node = stack.pop()
                # add parent Variables to stack
                stack.extend(node.parent_nodes)
                # how many times does this Variable appear as a parent Variable
                counts[node] += 1
            return counts

        counts = get_node_counts(self)
        stack = [self]
        while stack:
            node = stack.pop()
            if counts[node] == 0:
                yield node
            for p in node.parent_nodes:
                counts[p] -= 1
                # if all children have been visited, add to stack
                if counts[p] == 0:
                    stack.append(p)

    def backward(self):
        '''Compute and update the gradient of this variable with respect to all other variables.'''

        if not leaf and not self.requires_grad:
            raise Exception(f'{self} does not track grad')

        # TODO perform backpropagation on nodes and implement the jvp and vjp functions for numpy ops
        for node in self.toposort():
            node._grad()
            node.backward()

    def __repr__(self) -> str:
        return f'Variable({self.value}, leaf={self.leaf}, requires_grad={self.requires_grad})'


class DAGVisualization:
    def __init__(self, end_node: Variable) -> None:
        self.end_node = end_node
        self.graph = nx.DiGraph()

    def add_node(self, node: Variable):
        self.graph.add_node(f'Variable({node.value})')

    def add_edge(self, parent: Variable, child: Variable):
        self.graph.add_edge(
            f'Variable({parent.value})', f'Variable({child.value})')

    def create_nx_graph(self):
        def construct_graph(start_node: Variable):
            for parent in start_node.parent_nodes:
                self.add_edge(parent, start_node)
                construct_graph(parent)
        construct_graph(self.end_node)
        print(self.graph.nodes)


def wrap_namespace(nns, ons):
    '''
    Wrap numpy namespace to add autodiff functionality to numpy functions.
    Filters out functions that are not differentiable.
    '''
    nograd_functions = [
        _np.ndim, _np.shape, _np.iscomplexobj, _np.result_type, _np.zeros_like,
        _np.ones_like, _np.floor, _np.ceil, _np.round, _np.rint, _np.around,
        _np.fix, _np.trunc, _np.all, _np.any, _np.argmax, _np.argmin,
        _np.argpartition, _np.argsort, _np.argwhere, _np.nonzero, _np.flatnonzero,
        _np.count_nonzero, _np.searchsorted, _np.sign, _np.ndim, _np.shape,
        _np.floor_divide, _np.logical_and, _np.logical_or, _np.logical_not,
        _np.logical_xor, _np.isfinite, _np.isinf, _np.isnan, _np.isneginf,
        _np.isposinf, _np.allclose, _np.isclose, _np.array_equal, _np.array_equiv,
        _np.greater, _np.greater_equal, _np.less, _np.less_equal, _np.equal,
        _np.not_equal, _np.iscomplexobj, _np.iscomplex, _np.size, _np.isscalar,
        _np.isreal, _np.zeros_like, _np.ones_like, _np.result_type
    ]
    function_types = {_np.ufunc, FunctionType, BuiltinFunctionType}

    # construct new namespace with primitive wrapped functions
    for name, obj in ons.items():
        if obj in nograd_functions:
            nns[name] = primitive(obj, usegrad=False)
        elif type(obj) in function_types:
            nns[name] = primitive(obj)
        else:
            nns[name] = obj


def copy_doc(copy_func: Callable):
    '''Decorator to copy docstring from another function.'''
    def wrapper(func: Callable) -> Callable:
        func.__doc__ = copy_func.__doc__
        return func
    return wrapper


def primitive(fn, usegrad=True):
    '''Wraps operations to add them to the computational graph.'''
    @copy_doc(fn)
    @wraps(fn)
    def inner(*args, **kwargs):
        def getval(x):
            return x.value if isinstance(x, Variable) else x

        argvals = [getval(x) for x in args] if len(args) else args
        kwargvals = [dict(k, getval(v))
                     for k, v in kwargs.items()] if len(kwargs) else kwargs

        # get parents
        parents = [x for x in (
            list(args) + list(kwargs.values())) if isinstance(x, Variable)]

        value = fn(*argvals, **kwargvals)
        print(fn)
        return Variable(value, leaf=False, requires_grad=usegrad, parent_nodes=parents, fn=fn)

    return inner


# wrap numpy namespace
numpy_wrapper = globals()
wrap_namespace(numpy_wrapper, _np.__dict__)

setattr(Variable, 'ndim', property(lambda self: self.value.ndim))
setattr(Variable, 'size', property(lambda self: self.value.size))
setattr(Variable, 'dtype', property(lambda self: self.value.dtype))
setattr(Variable, 'T', property(lambda self: numpy_wrapper['transpose'](self)))
setattr(Variable, 'shape', property(lambda self: self.value.shape))

setattr(Variable, '__len__', lambda self, other: len(self._value))
setattr(Variable, 'astype', lambda self, *args, **
        kwargs: numpy_wrapper['_astype'](self, *args, **kwargs))
setattr(Variable, '__neg__', lambda self: numpy_wrapper['negative'](self))
setattr(Variable, '__add__', lambda self,
        other: numpy_wrapper['add'](self, other))
setattr(Variable, '__sub__', lambda self,
        other: numpy_wrapper['subtract'](self, other))
setattr(Variable, '__mul__', lambda self,
        other: numpy_wrapper['multiply'](self, other))
setattr(Variable, '__pow__', lambda self,
        other: numpy_wrapper['power'](self, other))
setattr(Variable, '__div__', lambda self,
        other: numpy_wrapper['divide'](self, other))
setattr(Variable, '__mod__', lambda self,
        other: numpy_wrapper['mod'](self, other))
setattr(Variable, '__truediv__', lambda self,
        other: numpy_wrapper['true_divide'](self, other))
setattr(Variable, '__matmul__', lambda self,
        other: numpy_wrapper['matmul'](self, other))
setattr(Variable, '__radd__', lambda self,
        other: numpy_wrapper['add'](other, self))
setattr(Variable, '__rsub__', lambda self,
        other: numpy_wrapper['subtract'](other, self))
setattr(Variable, '__rmul__', lambda self,
        other: numpy_wrapper['multiply'](other, self))
setattr(Variable, '__rpow__', lambda self,
        other: numpy_wrapper['power'](other, self))
setattr(Variable, '__rdiv__', lambda self,
        other: numpy_wrapper['divide'](other, self))
setattr(Variable, '__rmod__', lambda self,
        other: numpy_wrapper['mod'](other, self))
setattr(Variable, '__rtruediv__', lambda self,
        other: numpy_wrapper['true_divide'](other, self))
setattr(Variable, '__rmatmul__', lambda self,
        other: numpy_wrapper['matmul'](other, self))
setattr(Variable, '__eq__', lambda self,
        other: numpy_wrapper['equal'](self, other))
setattr(Variable, '__ne__', lambda self,
        other: numpy_wrapper['not_equal'](self, other))
setattr(Variable, '__gt__', lambda self,
        other: numpy_wrapper['greater'](self, other))
setattr(Variable, '__ge__', lambda self,
        other: numpy_wrapper['greater_equal'](self, other))
setattr(Variable, '__lt__', lambda self,
        other: numpy_wrapper['less'](self, other))
setattr(Variable, '__le__', lambda self,
        other: numpy_wrapper['less_equal'](self, other))
setattr(Variable, '__abs__', lambda self: numpy_wrapper['abs'](self))
setattr(Variable, '__hash__', lambda self: id(self))
