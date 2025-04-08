from torch import Tensor, zeros, arange, randint, randperm, float32, manual_seed

def generate_sparse_mask_torch(
        in_features: int,
        out_features: int,
        density: float = 0.1,
        device: str = 'cuda',
        seed: int = None
) -> Tensor:

    if seed is not None:
        manual_seed(seed)

    mask: Tensor = zeros((in_features, out_features), dtype=float32, device=device)

    # Ensure one per row
    row_idx = arange(in_features)
    col_idx = randint(0, out_features, (in_features,))
    mask[row_idx, col_idx] = 1

    # Ensure one per column
    col_idx = arange(out_features)
    row_idx = randint(0, in_features, (out_features,))
    mask[row_idx, col_idx] = 1

    # Fill remaining based on density
    total = in_features * out_features
    target = int(total * density)
    current = mask.sum().int().item()
    needed = max(target - current, 0)

    if needed > 0:
        zero_indices = (mask == 0).nonzero(as_tuple=False)
        selected = zero_indices[randperm(zero_indices.size(0))[:needed]]
        mask[selected[:, 0], selected[:, 1]] = 1

    return mask
