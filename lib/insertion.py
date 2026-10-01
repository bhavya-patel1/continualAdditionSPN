import numpy as np
from scipy.stats import chi2_contingency

from lib.customspn import Node, LeafNode, ProductNode, SumNode
import pandas as pd
import copy

def pairwise_g_test(x_data: np.ndarray, y_data: np.ndarray):
    """Computes the G-Test score between two discrete variables."""
    # # Create a contingency table 
    # x_max, y_max = int(np.max(x_data)), int(np.max(y_data))
    # table = np.zeros((x_max + 1, y_max + 1))
    # np.add.at(table, (x_data.astype(int), y_data.astype(int)), 1)

    # # If a variable is a constant, they are entirely independent
    # if table.shape[0] < 2 or table.shape[1] < 2:
    #     return 0.0, 1.0  # g stat = 0, p-value = 1.0
    if len(np.unique(x_data)) < 2 or len(np.unique(y_data)) < 2:
        return 0.0, 1.0  # g stat = 0, p-value = 1.0
        
    table = pd.crosstab(x_data, y_data).values

    safe_table = table + 1e-8
    
    try:
        stat, p_val, dof, expected = chi2_contingency(safe_table, lambda_="log-likelihood")
        return stat, p_val
    except ValueError:
        return 0.0, 1.0

    stat, p_val, dof, expected = chi2_contingency(table, lambda_= "log-likelihood")
    return stat, p_val

def test_independence(scope: set, new_var_id: int, data: np.ndarray, p_threshold: float = 0.05) -> bool:
    """
    Returns False if new_var_id is independent of the scope and True if dependent
    """
    y_data = data[:, new_var_id]
    
    for x_id in scope:
        x_data = data[:, x_id]
        _, p_val = pairwise_g_test(x_data, y_data)
        
        if p_val < p_threshold:
            return True 
            
    # independent
    return False

def score_dependence(scope: set, new_var_id: int, data: np.ndarray) -> float:
    """
    Returns the maximum dependency score between the new variable and the scope.
    Ranks children to determine best branch to drill into
    """
    y_data = data[:, new_var_id]
    max_stat = -1.0
    
    for x_id in scope:
        x_data = data[:, x_id]
        stat, _ = pairwise_g_test(x_data, y_data)
        if stat > max_stat:
            max_stat = stat
            
    return max_stat 

def create_marginal_leaf(new_var_id: int, data: np.ndarray) -> LeafNode:
    """Calculates the marginal distribution P(Y) and returns a LeafNode."""
    y_data = data[:, new_var_id].astype(int)
    counts = np.bincount(y_data)

    smoothed_counts = counts + 0.1
    probs = smoothed_counts / smoothed_counts.sum()
    
    return LeafNode(scope=new_var_id, params=probs)

def create_bivariate_gadget(node: LeafNode, new_var_id: int, data: np.ndarray) -> SumNode:
    """
    Replaces a univariate leaf with a small Sum-Product gadget that models 
    the joint distribution P(X, Y) for binary variables.
    """
    #Identify existing variable X
    x_id = list(node.scope)[0]
    x_data = data[:, x_id].astype(int)
    y_data = data[:, new_var_id].astype(int)
    
    x_max = int(np.max(x_data))
    #Calculate P(X) for SumNode weights
    x_counts = np.bincount(x_data, minlength=x_max + 1) + 0.1
    x_weights = x_counts / x_counts.sum()
    
    children = []
    

    # #Calculate P(Y | X=0)
    # y_given_x0_counts = np.bincount(y_data[x_data == 0], minlength=2) + 0.1
    # y_probs_x0 = y_given_x0_counts / y_given_x0_counts.sum()
    
    # #Create product node for (X=0 AND Y)
    # prod_0 = ProductNode([
    #     LeafNode(scope=x_id, params=np.array([1.0 - 1e-6, 1e-6])), 
    #     LeafNode(scope=new_var_id, params=y_probs_x0)              # Y probs given X=0
    # ])
    # children.append(prod_0)
    
    # #Calculate P(Y | X=1)
    # y_given_x1_counts = np.bincount(y_data[x_data == 1], minlength=2) + 0.1
    # y_probs_x1 = y_given_x1_counts / y_given_x1_counts.sum()
    
    # prod_1 = ProductNode([
    #     LeafNode(scope=x_id, params=np.array([1e-6, 1.0 - 1e-6])), 
    #     LeafNode(scope=new_var_id, params=y_probs_x1)              # Y probs given X=1
    # ])
    # children.append(prod_1)
    for x_val in range(x_max + 1):
        # Calculate P(Y | X=x_val)
        mask = (x_data == x_val)
        if np.any(mask):
            y_given_x_counts = np.bincount(y_data[mask]) + 0.1
        else:
            # Fallback if a specific category doesn't appear in this data slice
            y_given_x_counts = np.bincount(y_data) + 0.1 
            
        y_probs_x = y_given_x_counts / y_given_x_counts.sum()
        
        # Create a "one-hot" probability array for X = x_val (e.g., [1e-6, 0.999998, 1e-6])
        x_params = np.full(x_max + 1, 1e-6)
        x_params[x_val] = 1.0 - (1e-6 * x_max)
        
        # Create product node for (X=x_val AND Y)
        prod = ProductNode([
            LeafNode(scope=x_id, params=x_params), 
            LeafNode(scope=new_var_id, params=y_probs_x)
        ])
        children.append(prod)
    
    #Combine with a SumNode
    return SumNode(children=children, weights=x_weights)

# def insert_variable(node: Node, new_var_id: int, data: np.ndarray) -> Node:
#     """
#     Recursively traverses the SPN to insert a new variable.
#     Returns the modified node.
#     """
    
#     #If new variable is independent of current scope
#     if not test_independence(node.scope, new_var_id, data):
#         marginal_leaf = create_marginal_leaf(new_var_id, data)
#         return ProductNode(children=[node, marginal_leaf])

#     if isinstance(node, ProductNode):
#         #Child with most dependence
#         scores = [score_dependence(child.scope, new_var_id, data) for child in node.children]
#         best_child_idx = np.argmax(scores)

#         node.children[best_child_idx] = insert_variable(node.children[best_child_idx], new_var_id, data)
        
#         #Update scope 
#         node._rebuild_scope()
#         return node
        
#     elif isinstance(node, SumNode):
#         # Branch with best corelation
#         scores = [score_dependence(child.scope, new_var_id, data) for child in node.children]
#         best_child_idx = np.argmax(scores)
#         node.children[best_child_idx] = insert_variable(node.children[best_child_idx], new_var_id, data)
        
#         # Padding sibling branches of best branch 
#         for i in range(len(node.children)):
#             if i != best_child_idx:
#                 padding_leaf = create_marginal_leaf(new_var_id, data)
#                 node.children[i] = ProductNode(children=[node.children[i], padding_leaf])
                
#         # Update scope
#         node._rebuild_scope()
#         return node
        
#     elif isinstance(node, LeafNode):
#         # Leaf node but still dependent
#         return create_bivariate_gadget(node, new_var_id, data)


def insert_variable(node: Node, new_var_id: int, data: np.ndarray) -> Node:
    """
    Recursively traverses the SPN to insert a new variable using Greedy Likelihood Optimization.
    Returns the modified node.
    """
    # Evaluating independent baseline
    marginal_leaf = create_marginal_leaf(new_var_id, data)
    indep_node = ProductNode(children=[copy.deepcopy(node), marginal_leaf])
    indep_node._rebuild_scope()
    indep_ll = np.sum(indep_node.eval(data))
    
    dep_node = None
    
    if isinstance(node, ProductNode):
        best_ll = -np.inf
        best_child_idx = -1
        best_updated_child = None

        for i, child in enumerate(node.children):
            temp_child = copy.deepcopy(child)
            updated_temp = insert_variable(temp_child, new_var_id, data)
            
            # Temporarily build the product node to test LL
            temp_prod = copy.deepcopy(node)
            temp_prod.children[i] = updated_temp
            temp_prod._rebuild_scope()
            
            ll = np.sum(temp_prod.eval(data))
            if ll > best_ll:
                best_ll = ll
                best_child_idx = i
                best_updated_child = updated_temp

        dep_node = copy.deepcopy(node)
        dep_node.children[best_child_idx] = best_updated_child
        dep_node._rebuild_scope()

    # elif isinstance(node, SumNode):
    #     dep_node = copy.deepcopy(node)
        
    #     # Get log-likelihoods per sample for each child branch
    #     ll_matrix = np.column_stack([child.eval(data) for child in node.children])
    #     log_weights = np.log(np.array(node.weights) + 1e-8)
    #     weighted_ll = ll_matrix + log_weights
        
    #     #EM cluster assignment
    #     assignments = np.argmax(weighted_ll, axis=1)
    #     min_samples = 100
        
        
    #     for i, child in enumerate(node.children):
    #         child_data = data[assignments == i]
            
    #         # In case of too few samples
    #         if len(child_data) < min_samples:
    #             padding_leaf = create_marginal_leaf(new_var_id, data)
    #             dep_node.children[i] = ProductNode(children=[copy.deepcopy(child), padding_leaf])
    #         else:
    #             dep_node.children[i] = insert_variable(copy.deepcopy(child), new_var_id, child_data)
                
    #     dep_node._rebuild_scope()
    
    
    
    elif isinstance(node, SumNode):
        dep_node = copy.deepcopy(node)
        
        # Hard EM cluster assignment 
        ll_matrix = np.column_stack([child.eval(data) for child in node.children])
        log_weights = np.log(np.array(node.weights) + 1e-8)
        weighted_ll = ll_matrix + log_weights
        assignments = np.argmax(weighted_ll, axis=1)

        for i, child in enumerate(node.children):
            # Slice the data for this specific cluster
            child_data = data[assignments == i]
            
            # 
            if len(child_data) > 25 and test_independence(child.scope, new_var_id, child_data):
                # DEPENDENT: Route the variable deeply into this cluster
                dep_node.children[i] = insert_variable(copy.deepcopy(child), new_var_id, child_data)
            else:
                # INDEPENDENT: Prune it to save nodes. Use padding 
                padding_leaf = create_marginal_leaf(new_var_id, child_data if len(child_data) > 0 else data) # Use parent data for a stable marginal
                dep_node.children[i] = ProductNode(children=[copy.deepcopy(child), padding_leaf])

        dep_node._rebuild_scope()

    elif isinstance(node, LeafNode):
        from lib.learnSPN import learn_spn
        x_id = node.scope if isinstance(node.scope, int) else list(node.scope)[0]
        local_scope = [x_id, new_var_id]
        dep_node = learn_spn(
            data,
            scope=local_scope,
            p_threshold=0.05,
            min_instances=10,
        )
    # Evaluate routing LL
    dep_ll = np.sum(dep_node.eval(data))

    # If dependent ll is not better than independent
    if dep_ll > indep_ll:
        return dep_node
    else:
        return indep_node
