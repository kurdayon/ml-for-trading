import numpy as np
from typing import Tuple, Union, Optional

def get_synth_data(
    n: int,
    a: float = 0.683 / 2.0,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (x, y) pairs.

    x ~ Uniform[-1, 1]
    mean  = a * x
    median = -a * x
    y = median + ε, where ε has mean ≈ (mean - median) and std ≈ 1 via a
        piecewise-scaled normal: ε = p*z if z>0 else n*z, z~N(0,1).

    Args
    ----
    n   : number of samples
    a   : slope parameter (default 0.683/2)
    rng : seed (int) or numpy Generator for reproducibility

    Returns
    -------
    xs, ys : np.ndarray, np.ndarray
    """
    # Normalize rng input
    rng = np.random.default_rng(rng)

    # Draw x
    xs = rng.uniform(-1.0, 1.0, size=n)

    # Targets
    mean   = a * xs
    median = -a * xs
    target = mean - median  # == 2*a*xs

    # Piecewise scaling parameters (vectorized)
    mu = np.sqrt(2.0 / np.pi)
    max_mean = np.sqrt(2.0 / (np.pi - 2.0))

    if np.any(np.abs(target) > max_mean + 1e-12):
        raise ValueError(
            f"Some target means are infeasible with std=1. "
            f"Max |mean| is {max_mean:.6f}. "
            f"Try reducing |a| (currently {a})."
        )

    d = 2.0 * target / mu
    s = np.sqrt(4.0 + 2.0 * (2.0 - np.pi) * target * target)

    pos_scale = (s + d) / 2.0
    neg_scale = (s - d) / 2.0

    # Draw z and apply piecewise scaling
    z = rng.standard_normal(n)
    eps = np.where(z > 0.0, pos_scale * z, neg_scale * z)

    ys = median + eps
    return xs, ys
