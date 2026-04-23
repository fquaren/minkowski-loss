"""
Evaluation metrics for Minkowski functional emulation and super-resolution.

Computes per-sample and grouped metrics, per-feature matrices, and
isoperimetric violation rates.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm


def compute_isoperimetric_violation(pred_phys: np.ndarray) -> float:
    """Fraction of (sample, threshold) pairs violating P² < 4πA.

    Parameters
    ----------
    pred_phys : np.ndarray, shape (N, 3, Q)
        Predicted gamma in physical space [A, P, topology].

    Returns
    -------
    float
        Percentage of violations.
    """
    A = pred_phys[:, 0, :]
    P = pred_phys[:, 1, :]
    P_min_sq = 4.0 * np.pi * A
    violations = P**2 < P_min_sq - 1e-6
    total = violations.size
    return float(np.sum(violations) / total * 100) if total > 0 else 0.0


def _per_sample_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """R², MSE, and target variance per sample per component.

    Parameters
    ----------
    preds, targets : np.ndarray, shape (N, 3, Q)

    Returns
    -------
    dict of arrays, each shape (N,)
    """
    N = preds.shape[0]
    components = ["A", "P", "T"]
    metrics = {}
    for m in ["R2", "MSE", "Var"]:
        for c in components:
            metrics[f"{m}_{c}"] = np.full(N, np.nan, dtype=np.float64)

    for i in tqdm(range(N), desc="Per-sample metrics", leave=False):
        for j, comp in enumerate(components):
            p = preds[i, j, :]
            t = targets[i, j, :]
            mask = np.isfinite(p) & np.isfinite(t)
            if np.sum(mask) < 2:
                continue
            tc, pc = t[mask], p[mask]
            var = np.var(tc)
            metrics[f"Var_{comp}"][i] = var
            metrics[f"MSE_{comp}"][i] = mean_squared_error(tc, pc)
            metrics[f"R2_{comp}"][i] = r2_score(tc, pc) if var > 1e-9 else np.nan

    return metrics


def _precipitation_groups(mean_precip: np.ndarray) -> pd.Series:
    """Assign samples to Zero / Low / Mid / High precipitation groups."""
    groups = pd.Series("Zero", index=range(len(mean_precip)), dtype=object)
    nonzero = mean_precip > 0
    if np.any(nonzero):
        nz_vals = mean_precip[nonzero]
        p33, p67 = np.quantile(nz_vals, [0.33, 0.67])
        for idx in np.where(nonzero)[0]:
            v = mean_precip[idx]
            if v <= p33:
                groups[idx] = "Low"
            elif v <= p67:
                groups[idx] = "Mid"
            else:
                groups[idx] = "High"
    return groups


def create_metrics_dataframe(
    preds_phys: np.ndarray,
    targets_phys: np.ndarray,
    original_images: np.ndarray,
    total_losses: np.ndarray,
    geom_losses: np.ndarray,
) -> pd.DataFrame:
    """Build comprehensive metrics DataFrame.

    Parameters
    ----------
    preds_phys : (N, 3, Q) predicted gamma in physical space
    targets_phys : (N, 3, Q) target gamma in physical space
    original_images : (N, H, W) input precipitation fields
    total_losses : (N,) total Minkowski loss per sample
    geom_losses : (N, 3) per-functional Minkowski loss

    Returns
    -------
    pd.DataFrame with per-sample metrics, group labels, and loss columns.
    """
    sample_metrics = _per_sample_metrics(preds_phys, targets_phys)
    mean_precip = np.mean(original_images, axis=(1, 2))

    data = {
        "total_loss": total_losses,
        "geom_loss_A": geom_losses[:, 0],
        "geom_loss_P": geom_losses[:, 1],
        "geom_loss_T": geom_losses[:, 2],
        "mean_precip": mean_precip,
    }
    data.update(sample_metrics)

    df = pd.DataFrame(data)
    df["precip_group"] = _precipitation_groups(mean_precip)
    return df


def calculate_grouped_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of all metrics grouped by precipitation intensity.

    Returns a DataFrame with groups [Zero, Low, Mid, High, All].
    """
    metric_cols = [
        c
        for c in df.columns
        if c.startswith(("total_loss", "geom_loss", "R2_", "MSE_", "Var_"))
    ]

    grouped = df.groupby("precip_group")[metric_cols].mean()
    counts = df.groupby("precip_group").size().to_frame("n_samples")
    result = pd.concat([grouped, counts], axis=1)

    all_row = df[metric_cols].mean().to_frame("All").T
    all_row["n_samples"] = len(df)
    result = pd.concat([result, all_row])

    order = ["Zero", "Low", "Mid", "High", "All"]
    return result.reindex(order).dropna(how="all")


def calculate_per_feature_metrics(
    preds_phys: np.ndarray,
    targets_phys: np.ndarray,
    quantiles: np.ndarray,
) -> dict:
    """R², MSE, and target variance for each (component, threshold) pair.

    Parameters
    ----------
    preds_phys, targets_phys : (N, 3, Q)
    quantiles : (Q,) threshold values

    Returns
    -------
    dict with DataFrames: r2_matrix, mse_matrix, var_matrix, mean_by_component
    """
    N, n_comp, n_q = preds_phys.shape
    idx = pd.Index(["Area", "Perimeter", "Topology"], name="Component")
    cols = pd.Index(quantiles, name="Threshold (mm/h)")

    p_flat = preds_phys.reshape(N, -1)
    t_flat = targets_phys.reshape(N, -1)
    valid = np.isfinite(p_flat).all(axis=1) & np.isfinite(t_flat).all(axis=1)

    if np.sum(valid) < 2:
        nan_df = pd.DataFrame(np.nan, index=idx, columns=cols)
        return {
            "r2_matrix": nan_df,
            "mse_matrix": nan_df.copy(),
            "var_matrix": nan_df.copy(),
            "mean_by_component": pd.DataFrame(
                {"Avg_R2": np.nan, "Avg_MSE": np.nan, "Avg_Var": np.nan},
                index=idx,
            ),
            "quantiles": quantiles,
        }

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = r2_score(t_flat[valid], p_flat[valid], multioutput="raw_values")
    mse = mean_squared_error(t_flat[valid], p_flat[valid], multioutput="raw_values")
    var = np.var(t_flat[valid], axis=0)

    r2_df = pd.DataFrame(r2.reshape(n_comp, n_q), index=idx, columns=cols)
    mse_df = pd.DataFrame(mse.reshape(n_comp, n_q), index=idx, columns=cols)
    var_df = pd.DataFrame(var.reshape(n_comp, n_q), index=idx, columns=cols)

    mean_df = pd.DataFrame(
        {
            "Avg_R2": r2_df.mean(axis=1),
            "Avg_MSE": mse_df.mean(axis=1),
            "Avg_Var": var_df.mean(axis=1),
        }
    )

    return {
        "r2_matrix": r2_df,
        "mse_matrix": mse_df,
        "var_matrix": var_df,
        "mean_by_component": mean_df,
        "quantiles": quantiles,
    }


def evaluate_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    quantiles: np.ndarray,
    feature_names: list,
) -> dict:
    """Aggregate evaluation metrics.

    Parameters
    ----------
    y_true_log, y_pred_log : (N, C, Q) log-transformed
    quantiles : (Q,)
    feature_names : list of str, length C

    Returns
    -------
    dict with MSE, R², Minkowski distance, isoperimetric violation.
    """
    y_t = y_true_log.reshape(y_true_log.shape[0], -1)
    y_p = y_pred_log.reshape(y_pred_log.shape[0], -1)

    metrics = {
        "MSE_Total": float(mean_squared_error(y_t, y_p)),
        "R2_Total": float(r2_score(y_t, y_p)),
        "Isoperimetric_Violation_Pct": compute_isoperimetric_violation(
            np.sign(y_pred_log) * np.expm1(np.abs(y_pred_log))
        ),
    }

    abs_diff = np.abs(y_pred_log - y_true_log.unsqueeze(-1))
    dist = np.trapezoid(abs_diff, x=quantiles, axis=2)
    metrics["Minkowski_Total"] = float(dist.sum(axis=1).mean())

    for i, name in enumerate(feature_names):
        metrics[f"R2_{name}"] = float(
            r2_score(y_true_log[:, i, :].flatten(), y_pred_log[:, i, :].flatten())
        )
        metrics[f"Minkowski_{name}"] = float(dist[:, i].mean())

    return metrics
