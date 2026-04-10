"""
Integral-geometric target computation via persistent homology.

This module is the single source of truth for computing the gamma-vector
(Area, Perimeter, B0/Euler) from 2D precipitation fields. It consolidates
logic previously duplicated across compute_gamma_targets.py,
compute_persistence_thresholds.py, evaluate_baselines.py, and
test_emulator_suite.py.

References:
    Schneider & Weil (2008), Stochastic and Integral Geometry, Springer.
    Maria et al. (2014), The GUDHI library, INRIA.
"""

import warnings
import numpy as np
import gudhi as gd
from skimage import measure


# ---------------------------------------------------------------------------
# Low-level TDA
# ---------------------------------------------------------------------------

def compute_persistence_diagram(field_2d: np.ndarray) -> list:
    """
    Compute the persistence diagram of a 2D precipitation field using
    superlevel-set filtration (implemented via negation + sublevel-set).

    Parameters
    ----------
    field_2d : np.ndarray, shape (H, W)
        Physical precipitation field. NaNs are replaced with 0.0 prior
        to filtration so they map to the background level after negation.

    Returns
    -------
    list of (dim, (birth, death)) tuples in the *negated* domain.
    """
    clean = np.nan_to_num(field_2d, nan=0.0).astype(np.float64)
    neg = -clean
    cc = gd.CubicalComplex(
        dimensions=neg.shape, top_dimensional_cells=neg.flatten()
    )
    return cc.persistence()


def extract_persistence_pairs(diagram: list, dim: int) -> np.ndarray:
    """
    Extract birth-death pairs for a given homological dimension,
    converted back to the original (non-negated) domain.

    Parameters
    ----------
    diagram : list
        Output of compute_persistence_diagram.
    dim : int
        Homological dimension (0 for components, 1 for holes).

    Returns
    -------
    np.ndarray, shape (n_pairs, 4)
        Columns: [birth_orig, death_orig, persistence, is_essential]
        where is_essential=1 for infinite-persistence features.
    """
    raw = np.array(
        [p[1] for p in diagram if p[0] == dim], dtype=np.float64
    )
    if raw.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)

    # Convert from negated sublevel-set to original superlevel-set coords
    births = -raw[:, 0]
    deaths = -raw[:, 1]

    # Identify essential features (infinite death in negated domain
    # becomes -inf in original; these are the global background)
    is_essential = np.isinf(deaths) & (deaths < 0)
    deaths[is_essential] = np.inf
    persistence = births - deaths
    persistence[is_essential] = np.inf

    return np.column_stack([births, deaths, persistence,
                            is_essential.astype(np.float64)])


# ---------------------------------------------------------------------------
# Betti number counting at given thresholds
# ---------------------------------------------------------------------------

def count_betti_at_thresholds(
    pairs: np.ndarray,
    thresholds: np.ndarray,
    persistence_threshold: float,
    include_background_at_low: bool = True,
    background_threshold: float = 0.01,
) -> np.ndarray:
    """
    Count the number of significant topological features alive at each
    physical threshold, implementing Proposition 1 from the paper.

    A feature born at b and dying at d is alive at threshold u iff:
        b >= u  AND  d < u  (for finite features)
    The essential (background) component is counted only when
    u <= background_threshold.

    Parameters
    ----------
    pairs : np.ndarray, shape (n, 4)
        Output of extract_persistence_pairs.
    thresholds : np.ndarray, shape (Q,)
        Physical intensity thresholds.
    persistence_threshold : float
        Minimum persistence to count a feature as significant.
    include_background_at_low : bool
        Whether to include the essential feature at low thresholds.
    background_threshold : float
        Maximum threshold value at which the essential feature is counted.

    Returns
    -------
    np.ndarray, shape (Q,)
        Feature count at each threshold.
    """
    if pairs.shape[0] == 0:
        return np.zeros(len(thresholds), dtype=np.float32)

    births = pairs[:, 0]
    deaths = pairs[:, 1]
    persistence = pairs[:, 2]
    is_essential = pairs[:, 3].astype(bool)

    is_significant = persistence > persistence_threshold
    thresh_bcast = thresholds[np.newaxis, :]  # (1, Q)

    # Finite features: born >= u AND dead < u AND significant
    finite_mask = (
        is_significant[:, np.newaxis]
        & (births[:, np.newaxis] >= thresh_bcast)
        & (deaths[:, np.newaxis] < thresh_bcast)
        & (~is_essential[:, np.newaxis])
    )
    counts = np.sum(finite_mask, axis=0).astype(np.float32)

    # Essential (background) feature at low thresholds
    if include_background_at_low:
        bg_mask = (
            is_significant[:, np.newaxis]
            & (births[:, np.newaxis] >= thresh_bcast)
            & is_essential[:, np.newaxis]
            & (thresh_bcast <= background_threshold)
        )
        counts += np.sum(bg_mask, axis=0).astype(np.float32)

    return counts


# ---------------------------------------------------------------------------
# Area and perimeter computation
# ---------------------------------------------------------------------------

def compute_area_perimeter(
    field_2d: np.ndarray,
    thresholds: np.ndarray,
    pixel_size_km: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute excursion-set area (km^2) and perimeter (km) at each threshold.

    Area is computed as the count of pixels above threshold times pixel area.
    Perimeter uses the marching-squares algorithm at the 0.5 contour level
    of the binary mask, scaled by pixel resolution.

    Parameters
    ----------
    field_2d : np.ndarray, shape (H, W)
        Physical precipitation field (NaN-cleaned).
    thresholds : np.ndarray, shape (Q,)
    pixel_size_km : float

    Returns
    -------
    area : np.ndarray, shape (Q,)
        Excursion-set area in km^2.
    perimeter : np.ndarray, shape (Q,)
        Excursion-set perimeter in km.
    """
    pixel_area = pixel_size_km ** 2
    clean = np.nan_to_num(field_2d, nan=0.0)
    Q = len(thresholds)

    # Vectorized area via broadcasting: (H, W, 1) >= (1, 1, Q)
    masks = clean[..., np.newaxis] >= thresholds[np.newaxis, np.newaxis, :]
    area = np.sum(masks, axis=(0, 1)).astype(np.float32) * pixel_area

    # Perimeter via marching squares (not vectorizable)
    perimeter = np.zeros(Q, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for i in range(Q):
            if not np.any(masks[:, :, i]):
                continue
            contours = measure.find_contours(
                masks[:, :, i].astype(np.float64), 0.5
            )
            perimeter[i] = sum(
                np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
                for c in contours
            ) * pixel_size_km

    return area, perimeter


# ---------------------------------------------------------------------------
# Full gamma-matrix computation
# ---------------------------------------------------------------------------

def compute_gamma_matrix(
    field_2d: np.ndarray,
    thresholds: np.ndarray,
    pixel_size_km: float = 2.0,
    thresh_b0: float = 0.05,
    thresh_b1: float = 0.05,
    topology_target: str = "euler",
) -> np.ndarray:
    """
    Compute the full gamma-matrix for a single precipitation field.

    Parameters
    ----------
    field_2d : np.ndarray, shape (H, W)
        Physical precipitation field.
    thresholds : np.ndarray, shape (Q,)
        Physical intensity thresholds.
    pixel_size_km : float
        Spatial resolution of a single pixel.
    thresh_b0 : float
        Persistence noise floor for 0-dimensional features (components).
    thresh_b1 : float
        Persistence noise floor for 1-dimensional features (holes).
    topology_target : str, one of {"euler", "b0"}
        If "euler": third row is chi = B0 - B1 (Euler characteristic).
        If "b0": third row is B0 only (connected component count).

    Returns
    -------
    gamma : np.ndarray, shape (3, Q) if topology_target in {"euler", "b0"}
            or shape (4, Q) if you need all four channels.
        Row 0: Area (km^2)
        Row 1: Perimeter (km)
        Row 2: B0 or chi depending on topology_target
    """
    if topology_target not in ("euler", "b0"):
        raise ValueError(
            f"topology_target must be 'euler' or 'b0', got '{topology_target}'"
        )

    Q = len(thresholds)
    thresholds = np.asarray(thresholds, dtype=np.float32)

    # 1. Geometry
    area, perimeter = compute_area_perimeter(field_2d, thresholds, pixel_size_km)

    # 2. Topology
    diagram = compute_persistence_diagram(field_2d)

    pairs_b0 = extract_persistence_pairs(diagram, dim=0)
    b0 = count_betti_at_thresholds(
        pairs_b0, thresholds, thresh_b0,
        include_background_at_low=True
    )

    if topology_target == "euler":
        pairs_b1 = extract_persistence_pairs(diagram, dim=1)
        b1 = count_betti_at_thresholds(
            pairs_b1, thresholds, thresh_b1,
            include_background_at_low=False
        )
        topo_row = b0 - b1  # Euler characteristic (can be negative)
    else:
        topo_row = b0

    gamma = np.stack([area, perimeter, topo_row], axis=0)
    return gamma


def compute_gamma_matrix_4ch(
    field_2d: np.ndarray,
    thresholds: np.ndarray,
    pixel_size_km: float = 2.0,
    thresh_b0: float = 0.05,
    thresh_b1: float = 0.05,
) -> np.ndarray:
    """
    Compute the 4-channel gamma matrix [A, P, B0, B1] for storage.

    This is used during preprocessing to store all four channels,
    allowing the choice of B0 vs Euler to be made at training time
    without rerunning the TDA pipeline.

    Returns
    -------
    gamma : np.ndarray, shape (4, Q)
    """
    thresholds = np.asarray(thresholds, dtype=np.float32)

    area, perimeter = compute_area_perimeter(field_2d, thresholds, pixel_size_km)

    diagram = compute_persistence_diagram(field_2d)

    pairs_b0 = extract_persistence_pairs(diagram, dim=0)
    b0 = count_betti_at_thresholds(
        pairs_b0, thresholds, thresh_b0,
        include_background_at_low=True
    )

    pairs_b1 = extract_persistence_pairs(diagram, dim=1)
    b1 = count_betti_at_thresholds(
        pairs_b1, thresholds, thresh_b1,
        include_background_at_low=False
    )

    return np.stack([area, perimeter, b0, b1], axis=0)


# ---------------------------------------------------------------------------
# Persistence threshold estimation
# ---------------------------------------------------------------------------

def estimate_persistence_thresholds(
    fields: np.ndarray,
    percentile: float = 95.0,
) -> dict:
    """
    Estimate empirical persistence noise floors from a collection of fields.

    Collects all finite persistence values across the sample and returns
    the specified percentile as the threshold for each homological dimension.

    Parameters
    ----------
    fields : np.ndarray, shape (N, H, W)
        Sample of precipitation fields (physical units).
    percentile : float
        Percentile of the persistence distribution to use as noise floor.

    Returns
    -------
    dict with keys "thresh_b0", "thresh_b1", "thresh_unified".
    """
    all_p_b0 = []
    all_p_b1 = []

    for i in range(fields.shape[0]):
        diagram = compute_persistence_diagram(fields[i])

        for dim, (b, d) in diagram:
            if not np.isfinite(b) or not np.isfinite(d):
                continue
            pers = abs(b - d)
            if pers <= 1e-6:
                continue
            if dim == 0:
                all_p_b0.append(pers)
            elif dim == 1:
                all_p_b1.append(pers)

    all_p_b0 = np.array(all_p_b0)
    all_p_b1 = np.array(all_p_b1)

    thresh_b0 = float(np.percentile(all_p_b0, percentile)) if len(all_p_b0) > 0 else 0.0
    thresh_b1 = float(np.percentile(all_p_b1, percentile)) if len(all_p_b1) > 0 else 0.0

    return {
        "thresh_b0": thresh_b0,
        "thresh_b1": thresh_b1,
        "thresh_unified": max(thresh_b0, thresh_b1),
    }


# ---------------------------------------------------------------------------
# Climatological threshold computation
# ---------------------------------------------------------------------------

def compute_climatological_thresholds(
    wet_pixels: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """
    Map probability quantiles to physical intensity thresholds using
    the empirical CDF of precipitating (wet) pixels.

    Parameters
    ----------
    wet_pixels : np.ndarray, shape (N_pixels,)
        All pixel values exceeding the drizzle threshold from the
        training set.
    quantiles : np.ndarray, shape (Q,)
        Probability levels in [0, 1].

    Returns
    -------
    np.ndarray, shape (Q,)
        Physical thresholds in mm/h.
    """
    return np.quantile(wet_pixels, quantiles).astype(np.float32)
