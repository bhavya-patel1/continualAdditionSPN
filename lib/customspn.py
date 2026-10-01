import numpy as np
from typing import Set, List, Optional
from scipy.special import logsumexp

class Node:
    '''Base class for all SPN nodes'''
    def __init__(self):
        self.scope: Set[int] = set()
        self.id: Optional[int] = None

    def eval(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LeafNode(Node):
    '''Leaf distribution over single variable'''
    def __init__(self, scope: int, params: np.ndarray):
        super().__init__()
        self.scope = {scope}
        self.params = params  # e.g., categorical probabilities

    def eval(self, data: np.ndarray) -> np.ndarray:
        # For drvs
        values = data[:, list(self.scope)[0]].astype(int)
        # return np.log(self.params[values])
        safe_values = np.clip(values, 0, len(self.params) - 1)
        probs = self.params[safe_values]
        probs = np.where(values >= len(self.params), 1e-6, probs)
        
        return np.log(probs)

class ProductNode(Node):
    '''Product of independent child distributions'''
    def __init__(self, children: List[Node]):
        super().__init__()
        self.children = children
        self._rebuild_scope()

    def _rebuild_scope(self):
        self.scope = set().union(*[c.scope for c in self.children])

    def eval(self, data: np.ndarray) -> np.ndarray:
        # Sum of log-probabilities
        return sum(child.eval(data) for child in self.children)

class SumNode(Node):
    '''Weighted mixture of child distributions'''
    def __init__(self, children: List[Node], weights: np.ndarray):
        super().__init__()
        self.children = children
        self.weights = weights / np.sum(weights)  # normalize
        self._rebuild_scope()

    def _rebuild_scope(self):
        self.scope = set().union(*[c.scope for c in self.children])

    def eval(self, data: np.ndarray) -> np.ndarray:
        # Log-sum-exp for numerical stability
        child_vals = np.array([child.eval(data) for child in self.children])
        log_weights = np.log(self.weights)[:, np.newaxis]
        return logsumexp(child_vals + log_weights, axis=0)

class SPN:
    '''Top-level SPN container'''
    def __init__(self, root: Node):
        self.root = root

    def eval(self, data: np.ndarray) -> np.ndarray:
        return self.root.eval(data)

    def is_valid(self) -> bool:
        return self._check_valid(self.root)

    def _check_valid(self, node: Node) -> bool:
        if isinstance(node, LeafNode):
            return True

        if isinstance(node, SumNode):
            # Completeness: all children same scope
            scopes = [child.scope for child in node.children]
            if not all(s == scopes[0] for s in scopes):
                print(f"Invalid SumNode: children have different scopes")
                return False

        if isinstance(node, ProductNode):
            # Consistency: children have disjoint scopes
            all_vars = []
            for child in node.children:
                all_vars.extend(child.scope)
            if len(all_vars) != len(set(all_vars)):
                print(f"Invalid ProductNode: children have overlapping scopes")
                return False

        return all(self._check_valid(child) for child in node.children)
    