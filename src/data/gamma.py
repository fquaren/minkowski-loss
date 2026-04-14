"""
Computation of Minkowski functionals (area, perimeter, topology) via
persistent homology and excursion set analysis.

This module is the single source of truth for all topological target
computation in the Mink-DDPM project. It is used by:
  - Preprocessing (offline gamma target generation)
  - Evaluation (exact metric computation on predictions)
  - Testing (mechanistic emulator verification)

References:
  - GUDHI library: Maria et al., 2014
  - Persistent homology for excursion sets: Edelsbrunner & Harer, 2010
  - Minkowski functionals: Schneider & Weil, 2008
"""

import warnings
import numpy as np
import gudhi as gd
from skimage import measure


# ---------------------------------------------------------------------------
# 1. Low-level TDA primitives
# ---------------------------------------------------------------------------

def compute_persistence_diagram(field_2d: np.ndarray) -> list:
    """Compute the persistence diagram of a 2D field via superlevel-set filtration.

    We negate the field so that GUDHI's sublevel-set filtration recovers
    superlevel-set features (precipitation peaks → connected components).

    Parameters
    ----------
    field_2d : np.ndarray, shape (H, W)
        Physical precipitation field. NaN values are replaced with 0.0
        (background / no precipitation).

    Returns
    -------
    list of (dim, (birth, death))
        Raw persistence pairs from GUDHI, in the *negated* domain.
    """
    clean = np.nan_to_num(field_2d, nan=0.0).astype(np.float64)
    neg_field = -clean
    cc = gd.CubicalComplex(
        dimensions=neg_field.shape,
        top_dimensional_cells=neg_field.flatten(),
    )
    return cc.persistence()


def _extract_pairs(persistence_pairs: list, dim: int) -> np.ndarray:
    """Extract (birth, death) pairs for a given homological dimension.

    Returns an empty (0, 2) array if no pairs exist for that dimension.
    """
    pairs = [p[1] for p in persistence_pairs if p[0] == dim]
    if not pairs:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(pairs, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. Betti number counting at given thresholds
# ---------------------------------------------------------------------------

def _count_b0(
    persistence_pairs: list,
    thresholds: np.ndarray,
    thresh_b0: float,
) -> np.ndarray:
    """Count persistence-filtered connected components (B0) at each threshold.

    In superlevel-set persistence (obtained by negating the field for
    GUDHI's sublevel-set algorithm):

    - Components are **born** at local maxima of the precipitation field.
    - Components **die** at saddle points where they merge into an older
      (higher-born) component.
    - The **essential** feature is the global maximum: it never merges
      and is alive at every threshold t < birth.

    A finite component is alive at threshold t if:
        birth >= t  AND  death < t  AND  persistence > thresh_b0

    The essential component is alive at threshold t if:
        birth >= t  (it never dies, so no death condition)

    Parameters
    ----------
    persistence_pairs : list
        Raw GUDHI output from `compute_persistence_diagram`.
    thresholds : np.ndarray, shape (Q,)
        Physical intensity thresholds in mm/h.
    thresh_b0 : float
        Minimum persistence to consider a feature significant.

    Returns
    -------
    np.ndarray, shape (Q,)
        Filtered B0 counts at each threshold.
    """
    pairs_d0 = _extract_pairs(persistence_pairs, dim=0)
    counts = np.zeros(len(thresholds), dtype=np.float32)

    if pairs_d0.shape[0] == 0:
        return counts

    # Convert from negated domain back to original coordinates.
    # GUDHI sublevel: birth_neg < death_neg.
    # Original superlevel: birth_orig = -birth_neg > -death_neg = death_orig.
    births = -pairs_d0[:, 0]   # peak intensity where component appears
    deaths = -pairs_d0[:, 1]   # saddle intensity where component merges

    # Essential feature: GUDHI returns death_neg = +inf for the component
    # that never merges.  After negation: death_orig = -inf.
    is_essential = ~np.isfinite(deaths)

    # Persistence: birth - death for finite pairs, inf for essential.
    persistence = np.where(is_essential, np.inf, births - deaths)
    is_significant = persistence > thresh_b0

    # Broadcast: (n_pairs, 1) vs (1, Q)
    thresh_1d = thresholds[np.newaxis, :]

    # Finite features: alive at threshold t if born above t and dead below t
    alive_finite = (
        is_significant[:, np.newaxis]
        & (~is_essential[:, np.newaxis])
        & (births[:, np.newaxis] >= thresh_1d)
        & (deaths[:, np.newaxis] < thresh_1d)
    )

    # Essential feature: alive at every threshold below its birth value
    alive_essential = (
        is_essential[:, np.newaxis]
        & (births[:, np.newaxis] >= thresh_1d)
    )

    counts = np.sum(alive_finite, axis=0) + np.sum(alive_essential, axis=0)
    return counts.astype(np.float32)


def _count_b1(
    persistence_pairs: list,
    thresholds: np.ndarray,
    thresh_b1: float,
) -> np.ndarray:
    """Count persistence-filtered holes (B1) at each threshold.

    A hole born at `birth_orig` and dying at `death_orig` is alive at
    threshold t if: birth_orig >= t AND death_orig < t.

    Parameters
    ----------
    persistence_pairs : list
        Raw GUDHI output from `compute_persistence_diagram`.
    thresholds : np.ndarray, shape (Q,)
        Physical intensity thresholds in mm/h.
    thresh_b1 : float
        Minimum persistence to consider a hole significant.

    Returns
    -------
    np.ndarray, shape (Q,)
        Filtered B1 counts at each threshold.
    """
    pairs_d1 = _extract_pairs(persistence_pairs, dim=1)
    counts = np.zeros(len(thresholds), dtype=np.float32)

    if pairs_d1.shape[0] == 0:
        return counts

    births = -pairs_d1[:, 0]
    deaths = -pairs_d1[:, 1]
    persistence = births - deaths

    is_significant = persistence > thresh_b1
    thresh_1d = thresholds[np.newaxis, :]

    alive = (
        is_significant[:, np.newaxis]
        & (births[:, np.newaxis] >= thresh_1d)
        & (deaths[:, np.newaxis] < thresh_1d)
    )

    counts = np.sum(alive, axis=0)
    return counts.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Area and perimeter via excursion sets
# ---------------------------------------------------------------------------

def _compute_area(masks_3d: np.ndarray, pixel_area_km2: float) -> np.ndarray:
    """Compute excursion set area at each threshold.

    Parameters
    ----------
    masks_3d : np.ndarray, shape (H, W, Q)
        Boolean excursion set masks.
    pixel_area_km2 : float
        Physical area of a single pixel in km².

    Returns
    -------
    np.ndarray, shape (Q,)
    """
    return np.sum(masks_3d, axis=(0, 1)).astype(np.float32) * pixel_area_km2


def _compute_perimeter(masks_3d: np.ndarray, pixel_size_km: float) -> np.ndarray:
    """Compute excursion set perimeter at each threshold via marching squares.

    Uses sub-pixel contour extraction at the 0.5 level of the binary mask,
    scaled by the physical pixel resolution.

    Parameters
    ----------
    masks_3d : np.ndarray, shape (H, W, Q)
        Boolean excursion set masks.
    pixel_size_km : float
        Physical pixel edge length in km.

    Returns
    -------
    np.ndarray, shape (Q,)
    """
    n_thresholds = masks_3d.shape[2]
    perimeters = np.zeros(n_thresholds, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for i in range(n_thresholds):
            mask = masks_3d[:, :, i]
            if not np.any(mask):
                continue
            contours = measure.find_contours(mask.astype(float), 0.5)
            total_px = sum(
                np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
                for c in contours
            )
            perimeters[i] = total_px * pixel_size_km

    return perimeters


# ---------------------------------------------------------------------------
# 4. Full gamma matrix computation
# ---------------------------------------------------------------------------

def compute_gamma_matrix(
    field_2d: np.ndarray,
    thresholds: np.ndarray,
    pixel_size_km: float,
    thresh_b0: float,
    thresh_b1: float,
) -> np.ndarray:
    """Compute the 4-channel gamma matrix [A, P, B0, B1] for a precipitation field.

    This is the lossless representation storing all four Minkowski-related
    quantities. Downstream code selects the desired 3-channel subset
    (A, P, B0) or (A, P, chi=B0-B1) via `select_topology_target`.

    Parameters
    ----------
    field_2d : np.ndarray, shape (H, W)
        Physical precipitation field in mm/h.
    thresholds : np.ndarray, shape (Q,)
        Physical intensity thresholds in mm/h.
    pixel_size_km : float
        Pixel edge length in km.
    thresh_b0 : float
        Persistence threshold for connected components.
    thresh_b1 : float
        Persistence threshold for holes.

    Returns
    -------
    np.ndarray, shape (4, Q), dtype float32
        Row 0: area (km²)
        Row 1: perimeter (km)
        Row 2: B0 (connected components)
        Row 3: B1 (holes)
    """
    thresholds = np.asarray(thresholds, dtype=np.float32)
    n_q = len(thresholds)
    gamma = np.zeros((4, n_q), dtype=np.float32)

    clean = np.nan_to_num(field_2d, nan=0.0)
    pixel_area_km2 = pixel_size_km ** 2

    # Excursion set masks: (H, W, Q)
    masks_3d = clean[..., np.newaxis] >= thresholds[np.newaxis, np.newaxis, :]

    # Area and perimeter (geometric, no TDA)
    gamma[0, :] = _compute_area(masks_3d, pixel_area_km2)
    gamma[1, :] = _compute_perimeter(masks_3d, pixel_size_km)

    # Topological features via persistent homology
    persistence_pairs = compute_persistence_diagram(clean)
    gamma[2, :] = _count_b0(persistence_pairs, thresholds, thresh_b0)
    gamma[3, :] = _count_b1(persistence_pairs, thresholds, thresh_b1)

    return gamma


def select_topology_target(
    gamma_4ch: np.ndarray,
    mode: str = "euler",
) -> np.ndarray:
    """Convert 4-channel gamma to 3-channel with the chosen topology target.

    Parameters
    ----------
    gamma_4ch : np.ndarray, shape (..., 4, Q)
        Full gamma matrix with [A, P, B0, B1].
    mode : str
        "euler" → third channel is chi = B0 - B1 (Euler characteristic).
        "b0"   → third channel is B0 (connected components only).

    Returns
    -------
    np.ndarray, shape (..., 3, Q)
        Gamma matrix with [A, P, topology_target].
    """
    if mode == "euler":
        topo = gamma_4ch[..., 2, :] - gamma_4ch[..., 3, :]
    elif mode == "b0":
        topo = gamma_4ch[..., 2, :]
    else:
        raise ValueError(f"Unknown topology mode: {mode!r}. Use 'euler' or 'b0'.")

    return np.stack([gamma_4ch[..., 0, :], gamma_4ch[..., 1, :], topo], axis=-2)


# ---------------------------------------------------------------------------
# 5. Dataset-level statistics (used during preprocessing)
# ---------------------------------------------------------------------------

def compute_climatological_thresholds(
    zarr_path: str,
    quantiles: np.ndarray,
    drizzle_threshold: float = 0.1,
    max_pixels: int = 50_000_000,
    chunk_size: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """Compute physical precipitation thresholds from the training CDF.

    Samples wet pixels (above drizzle threshold) from the training split
    and returns the physical intensity values corresponding to the
    requested quantile levels.

    Parameters
    ----------
    zarr_path : str
        Path to preprocessed_dataset.zarr.
    quantiles : np.ndarray
        Quantile levels in [0, 1], e.g. [0.01, 0.05, ..., 0.99].
    drizzle_threshold : float
        Minimum intensity to consider a pixel "wet".
    max_pixels : int
        Maximum number of wet pixels to sample (memory bound).
    chunk_size : int
        Number of samples to read per Zarr chunk.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    np.ndarray, shape (len(quantiles),), dtype float32
        Physical thresholds in mm/h.
    """
    import zarr
    from tqdm import tqdm

    store = zarr.open(zarr_path, mode="r")
    if "train" not in store:
        raise ValueError("Train group not found in Zarr store.")

    train_data = store["train/original_precip"]
    rng = np.random.default_rng(seed=seed)

    indices = np.arange(train_data.shape[0])
    rng.shuffle(indices)

    sampled_wet = []
    pixel_count = 0

    for i in tqdm(range(0, len(indices), chunk_size), desc="Sampling wet pixels"):
        if pixel_count >= max_pixels:
            break
        chunk_idx = np.sort(indices[i : i + chunk_size])
        chunk = train_data.oindex[chunk_idx]
        wet = chunk[chunk > drizzle_threshold]
        sampled_wet.append(wet)
        pixel_count += len(wet)

    all_wet = np.concatenate(sampled_wet)
    thresholds = np.quantile(all_wet, quantiles).astype(np.float32)

    print(f"Sampled {len(all_wet)} wet pixels.")
    for q, t in zip(quantiles, thresholds):
        print(f"  q={q:.2f} → {t:.4f} mm/h")

    return thresholds


def _extract_persistences(img: np.ndarray) -> tuple:
    """Worker function to compute persistence diagram metrics for a single image.
    
    Extracted from compute_persistence_thresholds to allow Python's pickle 
    module to serialize the function across process boundaries.
    """
    pairs = compute_persistence_diagram(img)
    p0, p1 = [], []
    for dim, (b, d) in pairs:
        if not np.isfinite(b) or not np.isfinite(d):
            continue
        pers = abs(b - d)
        if pers <= 1e-6:
            continue
        if dim == 0:
            p0.append(pers)
        elif dim == 1:
            p1.append(pers)
    return p0, p1


def compute_persistence_thresholds(
    zarr_path: str,
    num_samples: int = 2000,
    target_percentile: float = 95.0,
    seed: int = 42,
    max_workers: int = 4,
) -> dict:
    """Estimate empirical persistence noise floors from the training data.

    Computes persistence diagrams on a random subset of training images
    and returns the requested percentile of the persistence distribution
    for B0 and B1 independently.

    Parameters
    ----------
    zarr_path : str
        Path to preprocessed_dataset.zarr.
    num_samples : int
        Number of images to sample.
    target_percentile : float
        Percentile of the persistence distribution defining the noise floor.
    seed : int
        Random seed.
    max_workers : int
        Number of parallel workers for TDA computation.

    Returns
    -------
    dict with keys:
        "thresh_b0": float
        "thresh_b1": float
        "unified": float (max of both)
    """
    import zarr
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed

    store = zarr.open(zarr_path, mode="r")
    if "train" not in store:
        raise ValueError("Train group not found in Zarr store.")

    train_data = store["train/original_precip"]
    total = train_data.shape[0]

    rng = np.random.default_rng(seed=seed)
    sample_idx = np.sort(
        rng.choice(total, size=min(num_samples, total), replace=False)
    )
    images = train_data.oindex[sample_idx]

    all_p_b0 = []
    all_p_b1 = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_persistences, img) for img in images]
        for future in tqdm(as_completed(futures), total=len(futures), desc="TDA"):
            b0_vals, b1_vals = future.result()
            all_p_b0.extend(b0_vals)
            all_p_b1.extend(b1_vals)

    all_p_b0 = np.array(all_p_b0)
    all_p_b1 = np.array(all_p_b1)

    thresh_b0 = float(np.percentile(all_p_b0, target_percentile)) if len(all_p_b0) > 0 else 0.0
    thresh_b1 = float(np.percentile(all_p_b1, target_percentile)) if len(all_p_b1) > 0 else 0.0
    unified = max(thresh_b0, thresh_b1)

    print(f"Persistence thresholds at {target_percentile}th percentile:")
    print(f"  B0: {thresh_b0:.4f}")
    print(f"  B1: {thresh_b1:.4f}")
    print(f"  Unified (max): {unified:.4f}")

    return {
        "thresh_b0": thresh_b0,
        "thresh_b1": thresh_b1,
        "unified": unified,
    }