"""
Gradient analysis for emulator evaluation.

Computes Jacobian norms and saliency maps to assess emulator
sensitivity and receptive field quality.
"""

import numpy as np
import torch
from tqdm import tqdm


def compute_jacobian_norms(
    model,
    loader,
    device,
    n_samples: int = 100,
) -> dict:
    """Compute gradient norms for each functional at the median threshold.

    The gradient of output[:, k, mid_q].sum() w.r.t. input gives the
    sensitivity of the k-th functional to each input pixel.

    NOTE: This computes d(sum_b output_b)/d(input_b). Because the model
    uses GroupNorm (not BatchNorm), this equals d(output_b)/d(input_b)
    for each sample independently.

    Parameters
    ----------
    model : nn.Module (emulator, eval mode)
    loader : DataLoader
    device : str
    n_samples : int

    Returns
    -------
    dict with keys "Area", "Perimeter", "Topology", each a list of floats
    """
    model.eval()
    names = ["Area", "Perimeter", "Topology"]
    norms = {n: [] for n in names}
    count = 0

    for batch in loader:
        if count >= n_samples:
            break
        inputs = batch[0].to(device)
        inputs.requires_grad_(True)

        output = model(inputs)
        n_q = output.shape[2]
        mid = n_q // 2

        for k, name in enumerate(names):
            target = output[:, k, mid].sum()
            retain = (k < len(names) - 1)
            grad = torch.autograd.grad(
                target, inputs, retain_graph=retain, create_graph=False,
            )[0]
            batch_norms = grad.view(grad.size(0), -1).norm(p=2, dim=1)
            norms[name].extend(batch_norms.detach().cpu().numpy().tolist())

        count += inputs.shape[0]

    return norms


def generate_saliency_maps(
    model,
    loader,
    device,
    scaler_val: float,
    n_examples: int = 10,
) -> list:
    """Generate input saliency maps for selected examples.

    Selects examples across the precipitation intensity spectrum
    (dry, moderate, intense) for interpretability.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : str
    scaler_val : float
        For denormalization to identify wet/dry samples.
    n_examples : int

    Returns
    -------
    list of (input_image, gradient_map, label) tuples
    """
    model.eval()
    samples = []
    found_dry = False
    found_intense = False

    for batch in loader:
        if len(samples) >= n_examples:
            break

        inputs = batch[0].to(device)
        inputs.requires_grad_(True)

        output = model(inputs)
        target = output[:, 0, :].sum()  # area across all thresholds
        grad = torch.autograd.grad(target, inputs, create_graph=False)[0]

        inp_np = inputs.detach().cpu().numpy()
        grad_np = grad.detach().cpu().numpy()

        for b in range(inputs.shape[0]):
            if len(samples) >= n_examples:
                break

            img = inp_np[b, 0]
            g = grad_np[b, 0]
            # Denormalize to physical for classification
            phys_total = float(np.expm1(img * scaler_val).sum())

            if phys_total < 1.0 and not found_dry:
                label = "Dry"
                found_dry = True
            elif phys_total > 5000.0 and not found_intense:
                label = "Intense storm"
                found_intense = True
            else:
                label = f"Sample {len(samples)}"

            samples.append((img, np.abs(g), label))

    return samples
