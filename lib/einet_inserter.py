
import math
import numpy as np
import torch
import torch.optim as optim

from lib.einet_wrapper import build_einet, train_einet
from simple_einet.einet import Einet


def _inject_noise_to_sum_layers(einet: Einet, noise_std: float):
    """Add Gaussian noise to sum-layer logits only (not leaf parameters)."""
    from simple_einet.layers.linsum import LinsumLayer
    from simple_einet.layers.mixing import MixingLayer
    with torch.no_grad():
        for module in einet.modules():
            if isinstance(module, (LinsumLayer, MixingLayer)):
                if hasattr(module, "logits") and module.logits is not None:
                    module.logits.data.add_(
                        torch.randn_like(module.logits) * noise_std
                    )


def insert_variable_einet_native(
    data: np.ndarray,
    base_scope: list,
    new_var_id: int,
    depth: int = 3,
    num_sums: int = 8,
    num_leaves: int = 4,
    num_repetitions: int = 4,
    epochs_phase1: int = 100,
    epochs_phase2: int = 50,
    lr: float = 0.01,
    batch_size: int = 256,
    device: str = "cpu",
    freeze_base: bool = True,
    noise_std: float = 0.0,
    matched: bool = True,
) -> tuple:
    """
    Trains a RAT-SPN on base_scope + new_var_id using the marginalisation strategy.

    Args:
        freeze_base:  If True (default), freeze all base weights in Phase 2 and only
                      fine-tune the new variable's leaf logits.
                      If False, fine-tune ALL parameters on the full dataset,
                      optionally after injecting noise into sum-layer logits.
        noise_std:    Standard deviation of Gaussian noise added to sum-layer logits
                      at the start of Phase 2 (only when freeze_base=False).
                      0.0 = no noise. Typical values: 0.01, 0.1.

    Returns:
        (einet, full_scope) where full_scope = base_scope + [new_var_id].
    """
    full_scope = base_scope + [new_var_id]
    num_features = len(full_scope)
    new_var_local_idx = num_features - 1

    num_bins = int(data[:, full_scope].max()) + 1

    max_depth = max(1, int(math.floor(math.log2(max(1, num_features)))))
    if depth is None:
        depth = max(1, max_depth - 1) if matched else max_depth
    else:
        depth = min(depth, max_depth)

    mode_str = "frozen" if freeze_base else f"unfrozen (noise={noise_std})"
    print(f"  [Native/{mode_str}] Building Einet | features={num_features} | bins={num_bins} "
          f"| depth={depth} | S={num_sums} | I={num_leaves} | R={num_repetitions}")

    einet = build_einet(num_features, num_bins, depth, num_sums, num_leaves, num_repetitions)

    # Phase 1: train on N features, marginalise the new variable
    print("  Phase 1: Training on base scope (new variable marginalised)...")
    einet = train_einet(
        einet, data, full_scope,
        marginalized_scopes=[new_var_local_idx],
        epochs=epochs_phase1, lr=lr, batch_size=batch_size, device=device,
    )

    #Phase 2: incorporate the new variable
    local_data = data[:, full_scope].astype(np.int64)
    n = local_data.shape[0]
    einet = einet.to(device)
    einet.train()

    if freeze_base:
        # Freeze everything, unfreeze only the new variable's leaf logits
        print("  Phase 2: Fine-tuning new variable leaf (all other weights frozen)...")
        for param in einet.parameters():
            param.requires_grad = False

        leaf_logits = einet.leaf.base_leaf.logits
        leaf_logits.requires_grad = True

        def mask_grad(grad):
            mask = torch.zeros_like(grad)
            # logits shape: (1, num_channels, num_features, num_leaves, num_repetitions, num_bins)
            mask[:, :, new_var_local_idx, :, :, :] = 1.0
            return grad * mask

        hook_handle = leaf_logits.register_hook(mask_grad)
        phase2_lr = lr
        optimizer = optim.Adam([leaf_logits], lr=phase2_lr)

    else:
        # Optionally inject noise into sum-layer logits, then fine-tune everything
        if noise_std > 0:
            print(f"  Injecting Gaussian noise (std={noise_std}) into sum-layer logits...")
            _inject_noise_to_sum_layers(einet, noise_std)

        print("  Phase 2: Fine-tuning ALL parameters on full N-variable data...")
        for param in einet.parameters():
            param.requires_grad = True
        phase2_lr = lr * 0.5   # lower lr to avoid catastrophic forgetting
        optimizer = optim.Adam(einet.parameters(), lr=phase2_lr)
        hook_handle = None

    for epoch in range(epochs_phase2):
        perm = np.random.permutation(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = torch.tensor(local_data[idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            log_prob = einet(batch)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs_phase2} | NLL: {total_loss/n_batches:.4f}")

    if hook_handle is not None:
        hook_handle.remove()
    einet.eval()
    return einet, full_scope