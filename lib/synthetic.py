import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import numpy as np

def generate_synthetic_bn(num_nodes=15, edge_prob=0.3, max_states=4, seed=42):
    """Generates a random Bayesian Network DAG and multi-class CPTs."""
    rng = np.random.default_rng(seed)
    
    # Generate random Acyclic Graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                G.add_edge(i, j)
                
    model = BayesianNetwork(G.edges())
    model.add_nodes_from(G.nodes()) 
    
    cards = {}
    for node in model.nodes():
        cards[node] = rng.integers(2, max_states + 1)
        
    #Generate conditional probability tables
    for node in model.nodes():
        cardinality = cards[node]
        parents = list(model.predecessors(node))
        
        if not parents:
            # Root node prior probabilities
            probs = rng.dirichlet(np.ones(cardinality))
            cpd = TabularCPD(
                variable=node, 
                variable_card=cardinality, 
                values=probs.reshape(-1, 1) 
            )
        else:
            # Look up how many states each parent has
            parent_cards = [cards[p] for p in parents]
            num_parent_states = np.prod(parent_cards)
            
            # Generate random columns that sum to 1 for every parent combination
            probs = rng.dirichlet(np.ones(cardinality), size=num_parent_states).T
            cpd = TabularCPD(
                variable=node, 
                variable_card=cardinality, 
                values=probs,
                evidence=parents,
                evidence_card=parent_cards
            )
        model.add_cpds(cpd)
        
    assert model.check_model(), "Invalid Bayesian Network generated!"
    return model

def sample_synthetic_data(model, num_samples=10000, seed=42):
    """Samples data from the generated Bayesian Network."""
    sampler = BayesianModelSampling(model)
    df = sampler.forward_sample(size=num_samples, seed=seed, show_progress=False)
    return df
