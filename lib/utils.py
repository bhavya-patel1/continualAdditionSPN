import numpy as np
from lib.customspn import Node, LeafNode, ProductNode, SumNode
import uuid
import json
import os

def print_spn(node, depth=0):
    """Prints the SPN structure in the terminal."""
    indent = "    " * depth
    
    if isinstance(node, LeafNode):
        # Print leaf node with scope and probabilities
        params_str = np.round(node.params, 3)
        print(f"{indent} LeafNode  | Scope: {list(node.scope)} | P: {params_str}")
        
    elif isinstance(node, ProductNode):
        # Print product node
        print(f"{indent} ProductNode | Scope: {list(node.scope)}")
        for child in node.children:
            print_spn(child, depth + 1)
            
    elif isinstance(node, SumNode):
        # Print sum node with branch weights
        weights_str = np.round(node.weights, 3)
        print(f"{indent} SumNode   | Scope: {list(node.scope)} | Weights: {weights_str}")
        for child in node.children:
            print_spn(child, depth + 1)

def count_nodes(node: Node) -> int:
    """
    Recursively counts the total number of structural nodes in the SPN.
    """
    if isinstance(node, LeafNode):
        return 1
        
    count = 1  
    for child in node.children:
        count += count_nodes(child)
        
    return count

def count_nodes_by_depth(node, depth=0, depth_counts=None):
    """Recursively counts nodes at each depth level of the SPN."""
    if depth_counts is None:
        depth_counts = {}
    
    if depth not in depth_counts:
        depth_counts[depth] = 0
    depth_counts[depth] += 1
    
    if hasattr(node, 'children'):
        for child in node.children:
            count_nodes_by_depth(child, depth + 1, depth_counts)
            
    return depth_counts

def get_depth_differences(old_root, updated_root):
    """Compares two SPNs to output exactly how many nodes were added at each depth."""
    old_depths = count_nodes_by_depth(old_root)
    new_depths = count_nodes_by_depth(updated_root)
    
    added_by_depth = {}
    max_depth = max(list(old_depths.keys()) + list(new_depths.keys()))
    
    total_added = 0
    for d in range(max_depth + 1):
        old_count = old_depths.get(d, 0)
        new_count = new_depths.get(d, 0)
        added = new_count - old_count
        
        if added != 0:
            added_by_depth[d] = added
            total_added += added
            
    return added_by_depth, total_added