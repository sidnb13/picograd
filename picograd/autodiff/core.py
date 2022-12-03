'''Core autodiff functionality of variables and computational graphs.'''


class DAGNode:
    '''Simple node wrapper for computational graph.'''

    def __init__(self, weight, child, traced=False) -> None:
        self.weight = weight
        self.child = child

    def __repr__(self) -> str:
        return f'DAGNode({self.weight}, {self.child})'

    def backflow(self):
        '''Propagate the value through the node gate.'''
        return self.weight * (self.child.grad if self.child.grad is not None else 1.)


class Variable:
    def __init__(self, value, leaf=False, requires_grad=True) -> None:
        # The value of the variable
        self.value = value
        self.leaf = leaf
        self.requires_grad = requires_grad
        # values which depend on this variable in format (grad_z_wrt_self, z)
        self.child_nodes = [DAGNode(1., self)] if not leaf else []
        self.grad = None
        # dependencies for this variable (Nodes)
        self.deps = None if leaf else []

    def backward(self):
        '''Compute and update the gradient of this variable with respect to all other variables.'''

        # update the gradient wrt child nodes (beneficiaries) and avoid erroneous recomputing
        if self.grad is None:
            self.grad = sum([child.backflow() for child in self.child_nodes])

        # trigger recursive backpropagation
        if not self.leaf:
            for dep in self.deps:
                dep.backward()

    def __repr__(self) -> str:
        return f'Variable({self.value})'

    def __mul__(self, other):
        z = Variable(self.value * other.value, leaf=False,
                     requires_grad=self.requires_grad)

        if self.requires_grad:
            self.child_nodes.append(DAGNode(other.value, z))
            other.child_nodes.append(DAGNode(self.value, z))
            # add dependency
            z.deps.append(self)
            z.deps.append(other)
        return z

    def __add__(self, other):
        z = Variable(self.value + other.value, leaf=False,
                     requires_grad=self.requires_grad)
        if self.requires_grad:
            self.child_nodes.append(DAGNode(1., z))
            other.child_nodes.append(DAGNode(1., z))
            # add dependency
            z.deps.append(self)
            z.deps.append(other)
        return z

    def __div__(self, other):
        z = Variable(self.value / other.value, leaf=False,
                     requires_grad=self.requires_grad)
        if self.requires_grad:
            self.child_nodes.append(DAGNode(1 / other.value, z))
            other.child_nodes.append(DAGNode(-self.value / other.value**2, z))
            # add dependency
            z.deps.append(self)
            z.deps.append(other)
        return z
