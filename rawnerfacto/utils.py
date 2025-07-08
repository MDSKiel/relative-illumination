import torch
from jaxtyping import Float
from torch import Tensor


def get_transmittance(
    deltas: Float[Tensor, "*batch num_samples 1"], densities: Float[Tensor, "*batch num_samples 1"]
) -> Float[Tensor, "*batch num_samples 1"]:
    """Return transmittance based on predicted densities
    Args:
        densities: Predicted densities for samples along ray
    Returns:
        transmittance for each sample
    """
    delta_density = deltas * densities
    transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance = torch.cat(
        [torch.zeros((*transmittance.shape[:1], 1, transmittance.shape[-1]), device=densities.device), transmittance],
        dim=-2,
    )
    transmittance = torch.exp(-transmittance)  # [..., "num_samples"]
    transmittance = torch.nan_to_num(transmittance)
    return transmittance


def acc_loss(transmittance, acc_loss_factor):
    P = torch.exp(-torch.abs(transmittance) / 0.1) + acc_loss_factor * torch.exp(-torch.abs(1 - transmittance) / 0.1)
    loss = torch.mean(-torch.log(P))
    return loss
