import torch
import torch.nn as nn
import numpy as np

def get_angles(pos, k, d: int):
    """
    Calculate angles for positional encoding.
    This is a helper function and remains largely the same.
    """
    i = k // 2
    angles = pos / (10000 ** (2 * i / d))
    return angles

def positional_encoding(positions: int, d_model: int, device: torch.device):
    """
    Precomputes a matrix with all the positional encodings.

    Args:
        positions (int): Maximum number of positions to be encoded.
        d_model (int): The dimension of the model.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to place the tensor on.

    Returns:
        torch.Tensor: A tensor of shape (1, positions, d_model) with positional encodings.
    """
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # Apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = torch.from_numpy(angle_rads).float().unsqueeze(0).to(device)
    return pos_encoding

def create_causal_mask(size: int, device: torch.device):
    """
    Creates a causal mask for the decoder's self-attention.
    This prevents the model from looking at future tokens during training.

    Args:
        size (int): The sequence length.
        device (torch.device): The device to place the tensor on.

    Returns:
        torch.Tensor: A square tensor of shape (size, size) with the upper triangle masked.
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
