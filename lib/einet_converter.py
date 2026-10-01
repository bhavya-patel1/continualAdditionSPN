
import copy
import numpy as np
import torch
import torch.nn.functional as F
from lib.customspn import LeafNode, ProductNode, SumNode, Node
from lib.einet_wrapper import build_einet, train_einet
from simple_einet.einet import Einet


def einet_to_spn(einet: Einet, scope: list) -> Node:
    """
    Convert a trained Einet to a custom SPN tree exactly.
    """
    leaf = einet.leaf
    base_leaf = leaf.base_leaf
    logits = base_leaf.logits.detach().cpu()
    # shape: (1, 1, num_features, num_leaves, num_repetitions, num_bins)
    probs = torch.softmax(logits, dim=-1).numpy()[0, 0]
    
    D, I, R, bins = probs.shape
    num_features_out = leaf.num_features_out
    
    # 1. Base Leaves
    base_nodes = []
    for feat in range(D):
        actual_var = scope[feat]
        i_nodes = []
        for i in range(I):
            rep_nodes = []
            for rep in range(R):
                p = probs[feat, i, rep, :]
                rep_nodes.append(LeafNode(scope=actual_var, params=p))
            i_nodes.append(rep_nodes)
        base_nodes.append(i_nodes)
        
    # 2. FactorizedLeaf mapping
    current_nodes = []
    if hasattr(leaf, "permutation") and hasattr(leaf, "cardinality"):
        cardinality = leaf.cardinality
        perm = leaf.permutation.detach().cpu().numpy()
        for out_feat in range(num_features_out):
            feat_nodes = []
            for i in range(I):
                rep_nodes = []
                for rep in range(R):
                    prod_children = []
                    for c in range(cardinality):
                        idx = out_feat * cardinality + c
                        if idx < len(perm):
                            orig_feat = perm[idx]
                            if orig_feat < D:
                                prod_children.append(base_nodes[orig_feat][i][rep])
                    if len(prod_children) > 1:
                        rep_nodes.append(ProductNode(copy.deepcopy(prod_children)))
                    elif len(prod_children) == 1:
                        rep_nodes.append(copy.deepcopy(prod_children[0]))
                    else:
                        rep_nodes.append(None)
                feat_nodes.append(rep_nodes)
            current_nodes.append(feat_nodes)
            
    elif hasattr(leaf, "scopes"):
        scopes_tensor = leaf.scopes.detach().cpu().numpy() # [D, num_features_out, R]
        for out_feat in range(num_features_out):
            feat_nodes = []
            for i in range(I):
                rep_nodes = []
                for rep in range(R):
                    prod_children = []
                    for orig_feat in range(D):
                        if scopes_tensor[orig_feat, out_feat, rep] == 1:
                            prod_children.append(base_nodes[orig_feat][i][rep])
                    if len(prod_children) > 1:
                        rep_nodes.append(ProductNode(copy.deepcopy(prod_children)))
                    elif len(prod_children) == 1:
                        rep_nodes.append(copy.deepcopy(prod_children[0]))
                    else:
                        rep_nodes.append(None)
                feat_nodes.append(rep_nodes)
            current_nodes.append(feat_nodes)
    else:
        current_nodes = copy.deepcopy(base_nodes)

    # 3. Inner Linsum Layers (processed bottom-to-top)
    for layer in einet.layers:
        S_in = layer.num_sums_in
        S_out = layer.num_sums_out
        num_features_out = layer.num_features_out
        
        # log_weights via F.log_softmax (dim=1 normalises over S_in)
        log_weights = F.log_softmax(layer.logits.detach().cpu(), dim=1)
        w_norm = torch.exp(log_weights).numpy()
        
        new_nodes = []
        for out_feat in range(num_features_out):
            feat_nodes = []
            left_idx = 2 * out_feat
            right_idx = 2 * out_feat + 1
            
            for s_out in range(S_out):
                s_out_nodes = []
                for rep in range(R):
                    sum_children = []
                    sum_weights = []
                    for s_in in range(S_in):
                        left_node = None
                        if left_idx < len(current_nodes) and s_in < len(current_nodes[left_idx]) and rep < len(current_nodes[left_idx][s_in]):
                            left_node = current_nodes[left_idx][s_in][rep]
                            
                        right_node = None
                        if right_idx < len(current_nodes) and s_in < len(current_nodes[right_idx]) and rep < len(current_nodes[right_idx][s_in]):
                            right_node = current_nodes[right_idx][s_in][rep]
                            
                        if left_node is not None and right_node is not None:
                            prod = ProductNode([copy.deepcopy(left_node), copy.deepcopy(right_node)])
                            sum_children.append(prod)
                            sum_weights.append(w_norm[out_feat, s_in, s_out, rep])
                        elif left_node is not None:
                            sum_children.append(copy.deepcopy(left_node))
                            sum_weights.append(w_norm[out_feat, s_in, s_out, rep])
                        elif right_node is not None:
                            sum_children.append(copy.deepcopy(right_node))
                            sum_weights.append(w_norm[out_feat, s_in, s_out, rep])
                            
                    if len(sum_children) > 1:
                        sum_weights = np.array(sum_weights)
                        if sum_weights.sum() > 0:
                            sum_weights = sum_weights / sum_weights.sum()
                        else:
                            sum_weights = np.ones(len(sum_children)) / len(sum_children)
                        s_out_nodes.append(SumNode(sum_children, sum_weights))
                    elif len(sum_children) == 1:
                        s_out_nodes.append(sum_children[0])
                    else:
                        s_out_nodes.append(None)
                feat_nodes.append(s_out_nodes)
            new_nodes.append(feat_nodes)
        current_nodes = new_nodes
        
    # 4. Mixing Layer
    root_reps = current_nodes[0][0]
    
    if R > 1:
        mixing = einet.mixing
        # mixing.logits shape: (1, S_out, S_in) -> S_out=1, S_in=R
        log_w = F.log_softmax(mixing.logits.detach().cpu(), dim=2)
        w_norm = torch.exp(log_w).numpy()[0, 0, :] # shape (R,)
        w_norm = w_norm / w_norm.sum()
        
        root = SumNode(root_reps, list(w_norm))
    else:
        root = root_reps[0]
        
    return root


def train_and_convert_einet(
    data: np.ndarray,
    scope: list,
    depth: int = 3,
    num_sums: int = 8,
    num_leaves: int = 4,
    num_repetitions: int = 4,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 256,
    device: str = "cpu",
    matched: bool = True,
) -> Node:
    """
    Build, train, and convert an Einet to a custom SPN tree.
    """
    import math
    num_features = len(scope)
    num_bins = int(data[:, scope].max()) + 1

    max_depth = max(1, int(math.floor(math.log2(max(1, num_features)))))
    if depth is None:
        depth = max(1, max_depth - 1) if matched else max_depth   
    else:
        depth = min(depth, max_depth)

    print(f"  [Converter] Building Einet | features={num_features} | bins={num_bins} | depth={depth} | S={num_sums} | I={num_leaves} | R={num_repetitions}")
    einet = build_einet(num_features, num_bins, depth, num_sums, num_leaves, num_repetitions)
    einet = train_einet(einet, data, scope, epochs=epochs, lr=lr, batch_size=batch_size, device=device)

    print("  Converting Einet to custom SPN tree")
    root = einet_to_spn(einet, scope)
    return root