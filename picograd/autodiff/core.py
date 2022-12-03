'''Core autodiff functionality of variables and computational graphs.'''

class Variable:
    def __init__(self, value: int) -> None:
        # The value of the variable
        self.value = value
        # values which depend on this variable in format (grad_z_wrt_self, z)
        self.child_nodes = []
        self.gradient = None
    
    def __mul__(self, other: Variable) -> Variable:
        z = Variable(self.value * other.value)
        self.child_nodes.append((other.value, z))
        other.child_nodes.append((self.value, z))
        
    def __add__(self, other: int) -> Variable:
        z = Variable(self.value + other.value)
        self.child_nodes.append((other.value, z))
        other.child_nodes.append((self.value, z))
        return z
    
    def __div__(self, other: Variable) -> Variable:
        z = Variable(self.value / other.value)
        self.child_nodes.append((1 / other.value, z))
        other.child_nodes.append((-self.value / other.value**2, z))
        return z