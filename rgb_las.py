"""
rgb_las.py

LAS (Locally-Adaptive Sorting) implementation for 2D grid arrangement.

Reference
---------
Visual Computing Group, HPI — LAS / FLAS repository:
https://github.com/Visual-Computing/LAS_FLAS/tree/main

The algorithm iteratively:
  1. Low-pass filters the current grid to produce smooth "prototype" vectors.
  2. Solves a linear assignment problem between input vectors and prototype vectors.
  3. Rearranges the grid according to the optimal assignment.
  4. Shrinks the filter radius and repeats until radius < 1.

Public API
----------
sort_with_las(X, radius_factor, wrap)   → (sorted_grid, permutation_indices)
distance_preservation_quality(sorted_X) → float
"""

import time
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment


# ── low-level distance / filter helpers ───────────────────────────────────────

def squared_l2_distance(q, p):
    """
    Compute the squared L2 (Euclidean) distance matrix between rows of *q* and *p*.

    Parameters
    ----------
    q, p : np.ndarray
        2-D arrays of shape (M, D) and (N, D) respectively.

    Returns
    -------
    np.ndarray
        Shape (N, M) distance matrix, clipped to [0, ∞).
    """
    ps = np.sum(p * p, axis=-1, keepdims=True)
    qs = np.sum(q * q, axis=-1, keepdims=True)
    return np.clip(ps - 2 * np.matmul(p, q.T) + qs.T, 0, np.inf)


def low_pass_filter(image, filter_size_x, filter_size_y, wrap=False):
    """
    Apply a uniform (box) low-pass filter to *image* along both spatial axes.

    Parameters
    ----------
    image : np.ndarray
        3-D array (H, W, D).
    filter_size_x : int
        Filter width along the W axis.
    filter_size_y : int
        Filter height along the H axis.
    wrap : bool
        If True, use circular ('wrap') boundary conditions; otherwise 'reflect'.

    Returns
    -------
    np.ndarray
        Filtered array, same shape as *image*.
    """
    mode = "wrap" if wrap else "reflect"
    im2 = uniform_filter1d(image, filter_size_y, axis=0, mode=mode)
    return uniform_filter1d(im2, filter_size_x, axis=1, mode=mode)


# ── spatial distance helpers (used by DPQ) ────────────────────────────────────

def compute_spatial_distances_for_grid(grid_shape, wrap):
    """Return the matrix of squared spatial distances for a 2-D grid."""
    if wrap:
        return _compute_spatial_distances_wrapped(grid_shape)
    return _compute_spatial_distances_non_wrapped(grid_shape)


def _compute_spatial_distances_wrapped(grid_shape):
    """Squared spatial distances with toroidal (wrapped) boundaries."""
    n_x, n_y = grid_shape
    wrap1 = [[0, 0],   [0, 0],   [0, 0],     [0, n_y],   [0, n_y],   [n_x, 0], [n_x, 0],   [n_x, n_y]]
    wrap2 = [[0, n_y], [n_x, 0], [n_x, n_y], [0, 0],     [n_x, 0],   [0, 0],   [0, n_y],   [0, 0]]

    a, b = np.indices(grid_shape)
    mat_flat = np.concatenate(
        [np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1
    ).reshape(-1, 2)

    d = squared_l2_distance(mat_flat, mat_flat)
    for w1, w2 in zip(wrap1, wrap2):
        d = np.minimum(d, squared_l2_distance(mat_flat + w1, mat_flat + w2))
    return d


def _compute_spatial_distances_non_wrapped(grid_shape):
    """Squared spatial distances without boundary wrapping."""
    a, b = np.indices(grid_shape)
    mat_flat = np.concatenate(
        [np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1
    ).reshape(-1, 2)
    return squared_l2_distance(mat_flat, mat_flat)


def _sort_hddists_by_2d_dists(hd_dists, ld_dists):
    """
    Sort each row of *hd_dists* in order of ascending *ld_dists*, breaking
    ties by the HD distance value itself.
    """
    max_hd = np.max(hd_dists) * 1.0001
    combined = hd_dists / max_hd + ld_dists
    combined_sorted = np.sort(combined)
    return np.fmod(combined_sorted, 1) * max_hd


def _get_distance_preservation_gain(sorted_d_mat, d_mean):
    """
    Compute the per-k Distance Preservation Gain Δ DP_k(S).

    Parameters
    ----------
    sorted_d_mat : np.ndarray
        Shape (N, N); each row is sorted in ascending order.
    d_mean : float
        Expected value of the pairwise distances.

    Returns
    -------
    np.ndarray
        Shape (N-1,) gain values, clipped to [0, ∞).
    """
    nums   = np.arange(1, len(sorted_d_mat))
    cumsum = np.cumsum(sorted_d_mat[:, 1:], axis=1)
    d_k    = (cumsum / nums).mean(axis=0)
    return np.clip((d_mean - d_k) / d_mean, 0, np.inf)


# ── public API ─────────────────────────────────────────────────────────────────

def sort_with_las(X, radius_factor=0.9, wrap=False):
    """
    Sort *X* onto a 2-D grid using the LAS (Locally-Adaptive Sorting) algorithm.

    Parameters
    ----------
    X : np.ndarray
        Shape (H, W, D) — H×W elements each with D-dimensional feature vector.
    radius_factor : float
        Multiplicative decay applied to the filter radius each iteration
        (0 < radius_factor < 1; smaller = faster convergence, lower quality).
    wrap : bool
        If True, use toroidal boundary conditions.

    Returns
    -------
    grid : np.ndarray
        Sorted grid of shape (H, W, D).
    final_indices : np.ndarray
        1-D permutation array of length H×W mapping original element indices
        to their grid positions.
    """
    np.random.seed(7)  # reproducible results

    N          = np.prod(X.shape[:-1])
    H, W       = X.shape[:-1]
    start_time = time.time()

    # Randomly initialise the grid
    grid   = np.random.permutation(X.reshape(N, -1)).reshape(X.shape).astype(float)
    flat_X = X.reshape(N, -1)

    radius_f      = max(H, W) / 2 - 1
    final_indices = np.arange(N)

    while True:
        print(".", end="", flush=True)
        radius       = int(np.round(radius_f))
        filter_size_x = min(W - 1, 2 * radius + 1)
        filter_size_y = min(H - 1, 2 * radius + 1)

        # Smooth the grid to obtain prototype vectors
        grid      = low_pass_filter(grid, filter_size_x, filter_size_y, wrap=wrap)
        flat_grid = grid.reshape(N, -1)

        # Solve the assignment problem
        C = squared_l2_distance(flat_X, flat_grid)
        C = (C / C.max() * 2048).astype(int)  # quantise for speed
        _, best_perm = linear_sum_assignment(C)

        # Apply permutation
        final_indices = final_indices[best_perm]
        flat_X        = flat_X[best_perm]
        grid          = flat_X.reshape(X.shape)

        radius_f *= radius_factor
        if radius_f < 1:
            break

    elapsed = time.time() - start_time
    print(f"\nSorted with LAS in {elapsed:.3f} seconds")
    return grid, final_indices


def distance_preservation_quality(sorted_X, p=2, wrap=False):
    """
    Compute the Distance Preservation Quality DPQ_p(S) of a sorted grid.

    DPQ measures how well the 2-D spatial neighbourhood structure reflects
    the high-dimensional distance structure of the data. A value of 1.0 is
    optimal; 0.0 means no improvement over random.

    Parameters
    ----------
    sorted_X : np.ndarray
        Sorted grid of shape (H, W, D).
    p : int or float
        Norm order used to aggregate the per-k gains (default 2).
    wrap : bool
        If True, assume toroidal grid boundaries.

    Returns
    -------
    float
        DPQ score in [0, 1].
    """
    grid_shape = sorted_X.shape[:-1]
    N          = np.prod(grid_shape)
    flat_X     = sorted_X.reshape(N, -1)

    dists_HD   = np.sqrt(squared_l2_distance(flat_X, flat_X))
    sorted_D   = np.sort(dists_HD, axis=1)
    mean_D     = sorted_D[:, 1:].mean()

    dists_spatial    = compute_spatial_distances_for_grid(grid_shape, wrap)
    sorted_HD_by_2D  = _sort_hddists_by_2d_dists(dists_HD, dists_spatial)

    delta_DP_k_2D = _get_distance_preservation_gain(sorted_HD_by_2D, mean_D)
    delta_DP_k_HD = _get_distance_preservation_gain(sorted_D, mean_D)

    return (np.linalg.norm(delta_DP_k_2D, ord=p)
            / np.linalg.norm(delta_DP_k_HD, ord=p))
