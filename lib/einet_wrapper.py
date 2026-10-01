
import numpy as np
import torch
import torch.optim as optim
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.categorical import Categorical


def build_einet(
    num_features: int,
    num_bins: int,
    depth: int = 3,
    num_sums: int = 8,
    num_leaves: int = 4,
    num_repetitions: int = 4,
) -> Einet:
    """Build an Einet (RAT-SPN) with Categorical leaves for discrete data."""
    cfg = EinetConfig(
        num_features=num_features,
        num_channels=1,
        num_sums=num_sums,
        num_leaves=num_leaves,
        num_repetitions=num_repetitions,
        depth=depth,
        num_classes=1,  # generative mode
        leaf_type=Categorical,
        leaf_kwargs={"num_bins": num_bins},
    )
    return Einet(cfg)


def train_einet(
    einet: Einet,
    data: np.ndarray,
    scope: list,
    marginalized_scopes: list = None,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 256,
    device: str = "cpu",
) -> Einet:
    """
    Train an Einet on data[:, scope] via NLL minimisation.

    Args:
        einet:               Built Einet (from build_einet).
        data:                Full data array (N, total_vars). Integer-valued.
        scope:               Column indices to use (einet was built for len(scope) features).
        marginalized_scopes: Local feature indices (0..len(scope)-1) to marginalise
                             during training. Used in Phase 1 of native insertion to
                             keep the new variable's leaf untrained.
        epochs:              Number of training epochs.
        lr:                  Adam learning rate.
        batch_size:          Mini-batch size.
        device:              'cpu' or 'cuda'.
    """
    einet = einet.to(device)
    einet.train()

    # Re-index data to local 0..len(scope)-1 indices
    local_data = data[:, scope].astype(np.int64)
    n = local_data.shape[0]

    optimizer = optim.Adam(einet.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            # Einet.forward() handles 2D input [N, D] by adding channel dim internally
            batch = torch.tensor(local_data[idx], dtype=torch.long, device=device)

            optimizer.zero_grad()
            log_prob = einet(batch, marginalized_scopes=marginalized_scopes)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | NLL: {total_loss/n_batches:.4f}")

    einet.eval()
    return einet


def einet_log_prob(einet: Einet, data: np.ndarray, scope: list, device: str = "cpu") -> np.ndarray:
    """
    Evaluate per-sample log-probabilities of an Einet.

    Returns:
        np.ndarray of shape (N,) — per-sample log-likelihoods.
    """
    einet = einet.to(device)
    einet.eval()
    local_data = data[:, scope].astype(np.int64)
    tensor = torch.tensor(local_data, dtype=torch.long, device=device)
    with torch.no_grad():
        log_prob = einet(tensor)
    return log_prob.squeeze(-1).cpu().numpy()