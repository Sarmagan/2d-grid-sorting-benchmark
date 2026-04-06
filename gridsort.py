"""
gridsort.py

Performance comparison of algorithms that map a set of elements to a 2D grid
based on their similarity. The benchmark uses a 32x32 random RGB image where
each pixel is a single element (1024 elements total).

Algorithms compared
-------------------
- LAS        : Locally-Adaptive Sorting
- GradSort   : Gradient-based differentiable sorting via Gumbel-Sinkhorn networks
- RasterFairy: t-SNE projection followed by grid assignment + swap optimisation
- SOM        : Self-Organising Maps
- Random     : Random baseline
- KS         : Kernelized Sorting
- TSNEH     : t-SNE + Hungarian + swap refinement

Usage
-----
    python gridsort.py --method [LAS|Gradsort|RF|SOM|Random|KS|IsoMatch|TSNEH]
"""

from PIL import Image
import numpy as np
import random
from scipy.optimize import linear_sum_assignment

import argparse
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

from minisom import MiniSom
from scipy.spatial.distance import cdist
from rgb_las import distance_preservation_quality, sort_with_las

from sklearn.manifold import TSNE
import rasterfairy
from rasterfairy import rfoptimizer

from KernelizedSorting_master.kernelized_sorting_color import KS
from isomatch import isomatch_algorithm

# ── device ────────────────────────────────────────────────────────────────────
device = torch.device("cpu")


# ── data generation ────────────────────────────────────────────────────────────

def generate_random_colors(nx=32, ny=32):
    """Return a (nx, ny, 3) float32 RGB tensor normalised to [0, 1]."""
    np.random.seed(3)
    X = np.random.uniform(0, 255, size=(nx, ny, 3)).astype(int)
    return torch.from_numpy(X).float() / 255


def generate_random_colors_numpy(nx=32, ny=32):
    """Return a (nx, ny, 3) uint8-range integer numpy array."""
    np.random.seed(3)
    return np.random.uniform(0, 255, size=(nx, ny, 3)).astype(int)


# ── reproducibility ────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """Set random seeds for PyTorch, NumPy and Python's random module."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# ── GradSort network building blocks ──────────────────────────────────────────

def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 1.0,
                    n_iter: int = 20, noise_factor: float = 0) -> torch.Tensor:
    """Apply Gumbel noise and Sinkhorn normalisation to obtain a soft permutation."""
    uniform_noise = torch.rand_like(log_alpha)
    gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
    log_alpha = log_alpha + noise_factor * gumbel_noise
    return log_sinkhorn_norm(log_alpha / tau, n_iter)


def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
    """Iterative Sinkhorn normalisation in log-space."""
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
    return log_alpha.exp()


class MultiheadAttention(nn.Module):
    """Thin wrapper around nn.MultiheadAttention for self-attention."""

    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x, mask=None):
        return self.attention(x, x, x, attn_mask=mask)[0]


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer (self-attention + feed-forward)."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(x, mask))
        x = self.norm1(x)
        x = x + self.dropout(self.feedforward(x))
        x = self.norm2(x)
        return x


class Transformer(nn.Module):
    """
    Transformer encoder that maps N input vectors to a flat N²-dimensional
    output (later reshaped into an N×N assignment matrix).
    """

    def __init__(self, input_size, n, d_model=512, n_heads=1, num_layers=4):
        super().__init__()
        scale = d_model ** -0.5
        self.embedding = nn.Linear(input_size, d_model)
        self.positional_embedding = nn.Parameter(scale * torch.randn(n + 1, d_model))
        self.class_embedding = nn.Parameter(scale * torch.randn(d_model))
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads) for _ in range(num_layers)]
        )
        self.out_Layer = nn.Linear(d_model, n**2)

    def forward(self, src):
        src = self.embedding(src)
        src = torch.cat([self.class_embedding.unsqueeze(0).to(src.dtype), src], dim=0)
        src = src + self.positional_embedding.to(src.dtype)
        for layer in self.encoder_layers:
            src = layer(src)
        return self.out_Layer(src[0, :])  # use class token only


class Gumbel_Sinkhorn_Transformer_Network(nn.Module):
    """
    Full GradSort network: Transformer → Gumbel-Sinkhorn → soft/hard permutation.
    """

    def __init__(self, input_dim, n, d_model, n_iter, num_layers, n_heads):
        super().__init__()
        self.n = n
        self.n_iter = n_iter
        self.transformer = Transformer(input_dim, n, d_model,
                                       num_layers=num_layers, n_heads=n_heads)

    def forward(self, in_x, tau=1, noise_factor=0):
        x = self.transformer(in_x).reshape(self.n, self.n)
        P_hat = gumbel_sinkhorn(x, tau, self.n_iter, noise_factor)
        P = torch.zeros_like(P_hat)
        P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
        P_hat_hard = (P - P_hat).detach() + P_hat
        return P_hat, P_hat_hard


class Gumbel_Sinkhorn_Network(nn.Module):
    """
    Lightweight GradSort network: a single learnable N²-vector passed through
    Gumbel-Sinkhorn (no transformer backbone).
    """

    def __init__(self, n, n_iter, device):
        super().__init__()
        self.n = n
        self.n_iter = n_iter
        self.W = nn.Parameter(torch.rand(n * n), requires_grad=True)

    def forward(self, tau=1, noise_factor=0):
        x = self.W.reshape(self.n, self.n)
        P_hat = gumbel_sinkhorn(x, tau, self.n_iter, noise_factor)
        P = torch.zeros_like(P_hat)
        P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
        P_hat_hard = (P - P_hat).detach() + P_hat
        return P_hat, P_hat_hard


# ── GradSort loss functions ────────────────────────────────────────────────────

def dist_matrix_loss_func(x_shuffled, x_sorted, p_dist):
    """Distance-matrix loss L_p(P): encourages global distance structure preservation."""
    dist_x_sorted = torch.cdist(x_sorted, x_sorted) ** p_dist
    dist_x_shuffled = torch.cdist(x_shuffled, x_shuffled) ** p_dist
    dist_x_shuffled_sort = torch.sort(torch.sort(dist_x_shuffled, 0)[0], 1)[0]
    dist_x_sorted_sort   = torch.sort(torch.sort(dist_x_sorted,   0)[0], 1)[0]
    mean_dist_x_shuffled = torch.mean(dist_x_shuffled_sort)
    return torch.mean(torch.abs(dist_x_shuffled_sort - dist_x_sorted_sort)) / mean_dist_x_shuffled


def neighbor_loss_func(x_shuffled, x_sorted, k, p_dist):
    """Neighbourhood loss L_nbr(P): penalises large distances between grid neighbours."""
    n = k * k
    dist_x_shuffled = torch.cdist(x_shuffled, x_shuffled, p=2) ** p_dist
    dist_x_sorted   = torch.cdist(x_sorted,   x_sorted,   p=2) ** p_dist
    mean_dist_x_shuffled = dist_x_shuffled.mean()

    dist_hor = torch.diag(dist_x_sorted, diagonal=1)
    ind = torch.arange(n - 1)
    dist_hor = dist_hor[(ind + 1) % k != 0]
    dist_hor_mean = torch.mean(dist_hor)

    dist_ver = torch.diag(dist_x_sorted, diagonal=k)
    dist_ver_mean = torch.mean(dist_ver)

    return (0.5 * dist_hor_mean + 0.5 * dist_ver_mean) / mean_dist_x_shuffled


def constraint_loss(perm):
    """Stochastic constraint loss L_s(P): penalises doubly-stochastic deviations."""
    return (torch.mean((torch.sum(perm, dim=0) - 1) ** 2)
          + torch.mean((torch.sum(perm, dim=1) - 1) ** 2))


# ── helper: printing ───────────────────────────────────────────────────────────

def pretty_print(**args):
    """Print key=value pairs in a compact aligned format."""
    for key, val in args.items():
        print(key, end=': ')
        if isinstance(val, int):
            print('%4d' % val, end=' ')
        elif isinstance(val, torch.Tensor):
            print('%5.3f' % round(float(val.detach().cpu()), 3), end=' ')
        elif isinstance(val, float):
            print('%5.3f' % val, end=' ')
        elif isinstance(val, str):
            print('%s' % val, end=' ')
    print()


def build_grid_coordinates(grid_size):
    """Return (N, 2) integer coordinates for a square grid."""
    yy, xx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
    return np.column_stack((yy.ravel(), xx.ravel())).astype(np.float64)


def build_grid_neighbor_lists(grid_size, radius=1):
    """
    Return weighted local neighbourhood lists for each grid cell.

    All cells within the given Euclidean radius are included and weighted by
    inverse distance, so closer grid relationships contribute more strongly.
    """
    coordinates = build_grid_coordinates(grid_size)
    neighbors = []
    for src_idx, src in enumerate(coordinates):
        cell_neighbors = []
        for dst_idx, dst in enumerate(coordinates):
            if src_idx == dst_idx:
                continue
            dist = np.linalg.norm(src - dst)
            if dist == 0 or dist > radius:
                continue
            cell_neighbors.append((dst_idx, 1.0 / dist))
        neighbors.append(cell_neighbors)
    return neighbors


def normalised_pairwise_distances(points):
    """Compute pairwise Euclidean distances and scale them to [0, 1]."""
    distances = cdist(points, points, metric='euclidean').astype(np.float64)
    max_dist = distances.max()
    if max_dist > 0:
        distances /= max_dist
    return distances


def normalise_points(points):
    """Min-max normalise each coordinate dimension to [0, 1]."""
    points = points.astype(np.float64)
    mins = points.min(axis=0, keepdims=True)
    maxs = points.max(axis=0, keepdims=True)
    return (points - mins) / (maxs - mins + 1e-12)


def resolve_tsne_perplexity(n_items, perplexity=None):
    """
    Pick a valid t-SNE perplexity for the current problem size.

    The default heuristic keeps the 32x32 behaviour close to the tuned value of
    24 while shrinking automatically for smaller grids where a large perplexity
    would be invalid or overly global.
    """
    if perplexity is None:
        target = min(24.0, max(5.0, np.sqrt(n_items)))
    else:
        target = float(perplexity)

    max_valid = max(1.0, n_items - 1.0 - 1e-3)
    return min(target, max_valid)


def build_tsne_assignment_cost(data, grid_size, perplexity=None, tsne_iter=2000):
    """
    Build the assignment cost for t-SNE + Hungarian grid placement.

    We first embed the RGB vectors into 2D with t-SNE over their pairwise
    distances, then measure the cost of assigning each embedded point to each
    grid cell. Hungarian assignment then converts this continuous embedding into
    a one-to-one placement on the discrete grid.
    """
    data_norm = data.astype(np.float64)
    if data_norm.max() > 1.0:
        data_norm /= 255.0

    resolved_perplexity = resolve_tsne_perplexity(data.shape[0], perplexity)
    distance_matrix = cdist(data_norm, data_norm, metric='euclidean')
    source_points = TSNE(
        n_components=2,
        perplexity=resolved_perplexity,
        metric='precomputed',
        init='random',
        learning_rate='auto',
        max_iter=tsne_iter,
        random_state=0,
        verbose=False,
    ).fit_transform(distance_matrix)
    source_points = normalise_points(source_points)

    target_points = build_grid_coordinates(grid_size)
    target_points = normalise_points(target_points)

    feature_cost = cdist(source_points, target_points, metric='euclidean')
    max_cost = feature_cost.max()
    if max_cost > 0:
        feature_cost /= max_cost
    return feature_cost


def _swap_delta_neighbor_energy(perm, pos_a, pos_b, feature_distances, neighbors):
    """
    Return the change in local neighbour energy if two grid positions are swapped.

    The energy is the sum of feature distances along grid edges. Negative deltas
    improve the layout by making adjacent cells more similar in feature space.
    """
    if pos_a == pos_b:
        return 0.0

    item_a = perm[pos_a]
    item_b = perm[pos_b]

    affected_edges = set()
    edge_weights = {}
    for src in (pos_a, pos_b):
        for dst, weight in neighbors[src]:
            edge = (src, int(dst)) if src < dst else (int(dst), src)
            affected_edges.add(edge)
            edge_weights[edge] = weight

    old_energy = 0.0
    new_energy = 0.0
    for u, v in affected_edges:
        weight = edge_weights[(u, v)]
        old_u = item_a if u == pos_a else item_b if u == pos_b else perm[u]
        old_v = item_a if v == pos_a else item_b if v == pos_b else perm[v]
        new_u = item_b if u == pos_a else item_a if u == pos_b else perm[u]
        new_v = item_b if v == pos_a else item_a if v == pos_b else perm[v]
        old_energy += weight * feature_distances[old_u, old_v]
        new_energy += weight * feature_distances[new_u, new_v]

    return new_energy - old_energy


def _compute_cell_neighbor_energies(perm, feature_distances, neighbors):
    """Return the per-cell weighted neighbour energy for the current layout."""
    energies = np.zeros(perm.shape[0], dtype=np.float64)
    for pos in range(perm.shape[0]):
        item = perm[pos]
        energies[pos] = sum(
            weight * feature_distances[item, perm[neighbor_pos]]
            for neighbor_pos, weight in neighbors[pos]
        )
    return energies


def _compute_focus_edge_energy(perm, focus_positions, feature_distances, neighbors):
    """Return the weighted neighbour energy over all edges touching focus cells."""
    seen_edges = set()
    energy = 0.0
    for src in focus_positions:
        for dst, weight in neighbors[src]:
            dst = int(dst)
            edge = (src, dst) if src < dst else (dst, src)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            energy += weight * feature_distances[perm[edge[0]], perm[edge[1]]]
    return energy


def iter_window_positions(grid_size, window_size, stride):
    """Yield flattened grid indices for sliding square windows."""
    max_start = grid_size - window_size
    if max_start < 0:
        return

    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)

    for row_start in starts:
        for col_start in starts:
            positions = []
            for row in range(row_start, row_start + window_size):
                base = row * grid_size
                for col in range(col_start, col_start + window_size):
                    positions.append(base + col)
            yield np.array(positions, dtype=np.int32)


def refine_permutation_with_window_reassignments(
    perm,
    data,
    grid_size,
    p_dist=2,
    window_sizes=(4, 3),
    radius=2,
):
    """
    Improve local regions with small Hungarian reassignments.

    Each window proposes a multi-item reassignment using only its fixed external
    neighbourhood as context, then accepts it only if the exact local energy
    over all touched edges decreases.
    """
    data_norm = data.astype(np.float64)
    if data_norm.max() > 1.0:
        data_norm /= 255.0

    feature_distances = cdist(data_norm, data_norm, metric='euclidean') ** p_dist
    neighbors = build_grid_neighbor_lists(grid_size, radius=radius)
    refined_perm = perm.copy()
    accepted = 0

    for window_size in window_sizes:
        if window_size > grid_size:
            continue
        stride = max(1, window_size // 2)
        for window_positions in iter_window_positions(grid_size, window_size, stride):
            window_set = {int(pos) for pos in window_positions}
            window_items = refined_perm[window_positions]
            cost_matrix = np.zeros((window_positions.shape[0], window_positions.shape[0]), dtype=np.float64)

            for target_col, target_pos in enumerate(window_positions):
                external_neighbors = [
                    (int(neighbor_pos), weight)
                    for neighbor_pos, weight in neighbors[int(target_pos)]
                    if int(neighbor_pos) not in window_set
                ]
                if not external_neighbors:
                    continue

                for item_row, item in enumerate(window_items):
                    cost_matrix[item_row, target_col] = sum(
                        weight * feature_distances[item, refined_perm[neighbor_pos]]
                        for neighbor_pos, weight in external_neighbors
                    )

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            reassigned_items = window_items[row_ind[np.argsort(col_ind)]]
            if np.array_equal(reassigned_items, window_items):
                continue

            old_energy = _compute_focus_edge_energy(
                refined_perm,
                window_positions,
                feature_distances,
                neighbors,
            )
            candidate_perm = refined_perm.copy()
            candidate_perm[window_positions] = reassigned_items
            new_energy = _compute_focus_edge_energy(
                candidate_perm,
                window_positions,
                feature_distances,
                neighbors,
            )
            if new_energy < old_energy - 1e-12:
                refined_perm = candidate_perm
                accepted += 1

    return refined_perm, accepted


def refine_permutation_with_neighbor_swaps(perm, data, grid_size, p_dist=2,
                                           num_attempts=250_000,
                                           candidate_pool=128,
                                           exploration_rate=0.10,
                                           early_stop_patience=50_000,
                                           neighborhood_radius=3,
                                           best_of_candidates=1,
                                           proposal_strategy="baseline",
                                           random_seed=0):
    """
    Greedy swap refinement using a local neighbour-similarity objective.

    Starting from an existing permutation, repeatedly propose swaps and accept
    them only when they reduce a weighted local feature-distance energy on the
    grid. Most proposals come from nearest-neighbour items in feature space,
    with a small amount of random exploration. The optional energy-mix proposal
    biases the first swap position toward currently expensive cells, and the
    best-of-k search can choose the strongest improvement among several sampled
    swap candidates.
    """
    n_items = perm.shape[0]
    data_norm = data.astype(np.float64)
    if data_norm.max() > 1.0:
        data_norm /= 255.0

    feature_distances = cdist(data_norm, data_norm, metric='euclidean') ** p_dist
    neighbors = build_grid_neighbor_lists(grid_size, radius=neighborhood_radius)

    k = min(candidate_pool + 1, n_items)
    nearest_items = np.argsort(feature_distances, axis=1)[:, :k]

    refined_perm = perm.copy()
    positions_of_items = np.empty(n_items, dtype=np.int32)
    positions_of_items[refined_perm] = np.arange(n_items, dtype=np.int32)

    rng = np.random.default_rng(random_seed)
    accepted = 0
    stale_steps = 0
    energy_probs = None

    for step in range(num_attempts):
        if proposal_strategy == "energy_mix":
            if step % 512 == 0 or energy_probs is None:
                energies = _compute_cell_neighbor_energies(
                    refined_perm,
                    feature_distances,
                    neighbors,
                )
                energy_probs = energies + 1e-9
                energy_probs /= energy_probs.sum()
            if rng.random() < 0.75:
                pos_a = int(rng.choice(n_items, p=energy_probs))
            else:
                pos_a = int(rng.integers(n_items))
        else:
            pos_a = int(rng.integers(n_items))
        item_a = refined_perm[pos_a]

        best_delta = 0.0
        best_pos_b = pos_a
        for _ in range(best_of_candidates):
            if rng.random() < exploration_rate:
                pos_b = int(rng.integers(n_items))
            else:
                candidate_idx = int(rng.integers(1, k))
                item_b = int(nearest_items[item_a, candidate_idx])
                pos_b = int(positions_of_items[item_b])

            delta = _swap_delta_neighbor_energy(
                refined_perm,
                pos_a,
                pos_b,
                feature_distances,
                neighbors,
            )
            if delta < best_delta:
                best_delta = delta
                best_pos_b = pos_b

        if best_delta < -1e-12:
            pos_b = best_pos_b
            item_b = refined_perm[pos_b]
            refined_perm[pos_a], refined_perm[pos_b] = item_b, item_a
            positions_of_items[item_a], positions_of_items[item_b] = pos_b, pos_a
            accepted += 1
            stale_steps = 0
        else:
            stale_steps += 1
            if stale_steps >= early_stop_patience:
                break

    return refined_perm, accepted


def build_size_aware_swap_schedule(grid_size):
    """
    Scale the swap-refinement schedule with the grid dimensions.

    Radii are chosen as fractions of the grid width, while the search budget is
    scaled roughly linearly with the number of cells. This preserves the tuned
    32x32 behaviour but adapts naturally to smaller or larger square grids.
    """
    n_items = grid_size * grid_size

    raw_radii = [
        max(1.0, grid_size / 6.0),
        max(1.0, grid_size / 10.0),
        max(1.0, grid_size / 20.0),
        1.0,
    ]
    radii = []
    for radius in raw_radii:
        if not radii or abs(radius - radii[-1]) > 0.25:
            radii.append(radius)

    attempt_coeffs = [98.0, 147.0, 98.0, 59.0]
    exploration_rates = [0.10, 0.10, 0.08, 0.05]
    best_of_values = [4, 4, 4, 8]
    pool_multipliers = [5, 4, 3, 3]
    patience_ratios = [0.25, 0.233, 0.25, 0.25]

    schedule = []
    for idx, radius in enumerate(radii):
        attempts = int(max(8_000, round(attempt_coeffs[idx] * n_items)))
        candidate_pool = min(n_items - 1, max(16, pool_multipliers[idx] * grid_size))
        patience = int(max(2_000, round(patience_ratios[idx] * attempts)))
        schedule.append(
            {
                "radius": radius,
                "attempts": attempts,
                "pool": candidate_pool,
                "exploration": exploration_rates[idx],
                "patience": patience,
                "best_of": best_of_values[idx],
            }
        )
    return schedule


def refine_permutation_with_multistage_swaps(perm, data, grid_size, p_dist=2):
    """
    Apply coarse-to-fine local swap refinement.

    Larger-radius stages first fix broader layout defects, then narrower stages
    clean up local inconsistencies.
    """
    stages = build_size_aware_swap_schedule(grid_size)
    refined_perm = perm.copy()
    total_accepted = 0
    for stage_idx, stage in enumerate(stages):
        refined_perm, accepted = refine_permutation_with_neighbor_swaps(
            refined_perm,
            data,
            grid_size,
            p_dist=p_dist,
            num_attempts=stage["attempts"],
            candidate_pool=stage["pool"],
            exploration_rate=stage["exploration"],
            early_stop_patience=stage["patience"],
            neighborhood_radius=stage["radius"],
            best_of_candidates=stage["best_of"],
            proposal_strategy="energy_mix",
            random_seed=stage_idx,
        )
        total_accepted += accepted
    return refined_perm, total_accepted


def solve_tsne_hungarian_permutation(data, grid_size, perplexity=None,
                                     tsne_iter=2000,
                                     use_window_refinement=True):
    """
    Build a grid permutation from a t-SNE embedding plus Hungarian assignment.

    The t-SNE embedding supplies a 2D geometry for the items, Hungarian maps
    those embedded points to discrete grid cells, and the result is then passed
    through the repo's non-cheating local swap refinement plus an optional
    windowed reassignment pass for stronger local repairs.
    """
    assignment_cost = build_tsne_assignment_cost(
        data,
        grid_size,
        perplexity=perplexity,
        tsne_iter=tsne_iter,
    )
    row_ind, col_ind = linear_sum_assignment(assignment_cost)
    permutation = row_ind[np.argsort(col_ind)]
    permutation, _ = refine_permutation_with_multistage_swaps(
        permutation,
        data,
        grid_size,
        p_dist=2,
    )
    if use_window_refinement:
        permutation, _ = refine_permutation_with_window_reassignments(
            permutation,
            data,
            grid_size,
            p_dist=2,
        )
    return permutation


# ── main algorithm functions ───────────────────────────────────────────────────

def las_rgb():
    """
    Sort a random 32×32 RGB grid using the LAS (Locally-Adaptive Sorting) algorithm.
    The result is saved as 'las_rgb.png'.
    """
    grid_size = 32
    X = generate_random_colors_numpy(grid_size, grid_size)  # (32, 32, 3)

    flat_img = X.reshape(-1, 3)
    X_grid = flat_img.reshape(grid_size, grid_size, -1)

    sorted_las, permutation_indices = sort_with_las(X_grid.copy(),
                                                    radius_factor=0.95,
                                                    wrap=False)
    dpq = distance_preservation_quality(sorted_las, p=16)
    print(f"Distance Preservation Quality: {dpq}")

    x_reordered = flat_img[permutation_indices].reshape(grid_size, grid_size, -1)
    Image.fromarray(x_reordered.astype(np.uint8)).save('las_rgb.png')




def gradsort_rgb():
    """
    Sort a random 32×32 RGB grid using GradSort.

    GradSort learns a permutation matrix end-to-end via a transformer network
    and a differentiable Gumbel-Sinkhorn operator. The result is saved as
    'gradsort_rgb.png'.
    """
    # Hyperparameters
    seed          = 0
    iterations    = 5000
    learning_rate = 0.04
    n_iter        = 15    # Sinkhorn iterations
    tau           = 1     # Sinkhorn temperature
    noise_factor  = 0.1   # Gumbel noise scale
    p_dist        = 2     # Distance exponent for loss functions
    dpq_p         = 16    # DPQ norm order

    x_orig = generate_random_colors(32, 32)
    ny, nx = x_orig.shape[0:2]
    n = nx * ny

    set_global_seed(seed)

    x_shuffled = x_orig.reshape(-1, 3).to(device)

    net = Gumbel_Sinkhorn_Transformer_Network(
        input_dim=x_shuffled.shape[-1], n=n, d_model=16,
        num_layers=3, n_iter=n_iter, n_heads=4
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for iteration in range(iterations + 1):
        optimizer.zero_grad()

        alpha = np.clip(iteration / (iterations + 1), 0, 1)
        perm_soft, perm_hard = net(x_shuffled, tau, noise_factor)

        x_sorted_soft = torch.matmul(perm_soft, x_shuffled)
        x_sorted_hard = torch.matmul(perm_hard, x_shuffled)

        loss  = neighbor_loss_func(x_shuffled, x_sorted_soft, nx, p_dist)
        loss += 5 * alpha * dist_matrix_loss_func(x_shuffled, x_sorted_soft, p_dist)
        loss += 100 * constraint_loss(perm_soft)

        ind = torch.argmax(perm_hard, -1)
        num_duplicates = int(n - torch.unique(ind).shape[0])
        end = (iteration == iterations) or (num_duplicates == 0)

        if iteration % 200 == 0 or end:
            dist = torch.cdist(x_sorted_soft, x_shuffled, p=p_dist)
            _, ind = linear_sum_assignment(dist.detach().cpu().numpy())
            pretty_print(
                it=iteration,
                L=float(loss),
                Dups=num_duplicates,
            )

        if end:
            break

        loss.backward()
        optimizer.step()

    # Save result in original RGB colours
    grid_size = 32
    data = generate_random_colors_numpy(grid_size, grid_size).reshape(-1, 3).astype(np.float32)
    x_reordered = data[ind].reshape(grid_size, grid_size, -1)
    Image.fromarray(x_reordered.astype(np.uint8)).save('gradsort_rgb.png')

    dpq = distance_preservation_quality(x_reordered, p=16)
    print(f"Distance Preservation Quality: {dpq}")

def raster_fairy_rgb():
    """
    Sort a random 32×32 RGB grid using RasterFairy.

    Elements are first embedded into 2D via t-SNE on pairwise Euclidean
    distances, then assigned to grid cells by RasterFairy's point-cloud
    transformer followed by a swap optimiser. Result saved as 'rf_rgb.png'.
    """
    grid_size = 32
    x_orig = generate_random_colors_numpy(grid_size, grid_size)
    w, h, size = x_orig.shape
    data = x_orig.reshape(-1, 3)

    # Embed to 2D
    distance_matrix = cdist(data, data, metric='euclidean')
    tsne = TSNE(perplexity=20, metric="precomputed", verbose=True, max_iter=5000, init="random")
    points_2d = tsne.fit_transform(distance_matrix)

    # Assign 2D points to grid cells
    grid_xy_tuple = rasterfairy.transformPointCloud2D(points_2d,
                                                      target=(grid_size, grid_size))
    grid_xy    = grid_xy_tuple[0]
    grid_shape = grid_xy_tuple[1]

    # Refine assignment via swap optimisation
    optimizer  = rfoptimizer.SwapOptimizer()
    swapTable  = optimizer.optimize(points_2d, grid_xy, grid_size, grid_size, 1_000_000)

    grid_xy      = grid_xy[swapTable]
    grid_indices = grid_xy[:, 0] * grid_shape[1] + grid_xy[:, 1]
    combined     = list(zip(range(grid_size * grid_size), grid_indices.tolist()))
    permutation_indices = [elem for elem, _ in sorted(combined, key=lambda x: x[1])]

    x_reordered = data[permutation_indices].reshape(h, w, size)
    Image.fromarray(x_reordered.astype(np.uint8)).save('rf_rgb.png')

    dpq = distance_preservation_quality(x_reordered, p=16)
    print(f"Distance Preservation Quality: {dpq}")


def self_organizing_maps_rgb():
    """
    Sort a random 32×32 RGB grid using a Self-Organising Map (SOM).

    A MiniSom network is trained on raw RGB pixel values. Each pixel
    is then assigned to its best-matching unit (BMU); collisions are resolved
    by the Hungarian algorithm. Result saved as 'som_rgb.png'.
    """
    x_orig = generate_random_colors_numpy(32, 32)
    w, h, size = x_orig.shape
    data = x_orig.reshape(-1, 3)

    # Train SOM on raw RGB values
    som = MiniSom(h, w, size,
                  learning_rate=0.4, sigma=5,
                  neighborhood_function='gaussian')
    som.train_random(data, 100_000, verbose=True)
    win_map = som.win_map(data, return_indices=True)

    new_order = np.full(w * h, -1, dtype=int)
    collided  = []

    # First pass: one index per BMU, collect extras
    for i in range(h):
        for j in range(w):
            position = (i, j)
            if position in win_map:
                indices = win_map[position]
                new_order[i * w + j] = indices[0]
                collided.extend(indices[1:])

    empty_positions = np.where(new_order == -1)[0]
    print(f"Empty positions: {len(empty_positions)}")
    print(f"Collided indices: {len(collided)}")

    # Resolve collisions via Hungarian assignment
    if len(collided) > 0 and len(empty_positions) > 0:
        empty_coords    = np.array([np.unravel_index(p, (h, w)) for p in empty_positions])
        collided_coords = np.array([som.winner(data[idx]) for idx in collided])
        cost_matrix     = cdist(collided_coords, empty_coords, metric='euclidean')
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_ind, col_ind):
            new_order[empty_positions[col]] = collided[row]

    x_reordered = data[new_order].reshape(h, w, size)
    Image.fromarray(x_reordered.astype(np.uint8)).save('som_rgb.png')

    dpq = distance_preservation_quality(x_reordered, p=16)
    print(f"Distance Preservation Quality: {dpq}")


def random_rgb():
    """
    Baseline: randomly shuffle the elements of a 32×32 RGB grid and report DPQ.
    """
    grid_size = 32
    x_orig = generate_random_colors_numpy(grid_size, grid_size)
    data   = x_orig.reshape(-1, 3)

    nums = list(range(grid_size * grid_size))
    random.shuffle(nums)
    permutation_indices = nums

    x_reordered = data[permutation_indices].reshape(grid_size, grid_size, -1)
    Image.fromarray(x_reordered.astype(np.uint8)).save('random_rgb.png')
    dpq = distance_preservation_quality(x_reordered, p=16)
    print(f"Distance Preservation Quality: {dpq}")

def KS_rgb():
    """
    Sort a random 32×32 RGB grid using Kernelized Sorting (KS).

    KS solves a joint kernel alignment problem to find the permutation matrix
    that best aligns the feature-space kernel with a spatial grid kernel.
    Result saved as 'ks_rgb.png'.
    """
    grid_size = 32
    x_orig = generate_random_colors_numpy(grid_size, grid_size)
    w, h, size = x_orig.shape
    data = x_orig.reshape(-1, 3)

    # Build target grid coordinates
    ny, nx = grid_size, grid_size
    griddata = np.zeros((2, ny * nx))
    griddata[0, ] = np.kron(range(1, ny + 1), np.ones((1, nx)))
    griddata[1, ] = np.tile(range(1, nx + 1), (1, ny))

    PI = KS(data, griddata.T)
    sorted_indices = PI.argmax(axis=1)

    x_reordered = data[sorted_indices].reshape(h, w, size)
    Image.fromarray(x_reordered.astype(np.uint8)).save('ks_rgb.png')

    dpq = distance_preservation_quality(x_reordered, p=16)
    print(f"Distance Preservation Quality: {dpq}")

def isomatch_rgb():
    """
    Sort a random 32×32 RGB grid using IsoMatch.

    IsoMatch uses an Isomap embedding to find a structure-preserving mapping
    from elements to grid positions via bipartite matching. Result saved as
    'isomatch_rgb.png'.
    """
    from scipy.spatial.distance import pdist, squareform

    grid_size = 32
    x_orig = generate_random_colors_numpy(grid_size, grid_size)
    data   = x_orig.reshape(-1, 3)

    data_norm = data.astype(np.float64) / 255.0
    d_matrix  = squareform(pdist(data_norm))

    perm, _, _ = isomatch_algorithm(d_matrix, grid_size=(grid_size, grid_size), num_swaps=50000)

    x_reordered = data[perm].reshape(grid_size, grid_size, -1)
    Image.fromarray(x_reordered.astype(np.uint8)).save('isomatch_rgb.png')

    dpq = distance_preservation_quality(x_reordered, p=16)
    print(f"Distance Preservation Quality: {dpq}")


def tsne_hungarian_rgb():
    """
    Sort a random 32×32 RGB grid using t-SNE + Hungarian assignment.

    The RGB vectors are embedded into 2D with t-SNE, assigned to grid cells via
    the Hungarian algorithm, and then refined with the repo's weighted local
    swap search. Result saved as 'tsneh_rgb.png'.
    """
    grid_size = 32
    x_orig = generate_random_colors_numpy(grid_size, grid_size)
    data = x_orig.reshape(-1, 3)

    permutation_indices = solve_tsne_hungarian_permutation(data, grid_size, tsne_iter=2000)

    x_reordered = data[permutation_indices].reshape(grid_size, grid_size, -1)
    Image.fromarray(x_reordered.astype(np.uint8)).save('tsneh_rgb.png')

    dpq = distance_preservation_quality(x_reordered, p=16)
    print(f"Distance Preservation Quality: {dpq}")

# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a 2D grid-sorting algorithm on a random 32×32 RGB image."
    )
    parser.add_argument(
        "--method",
        default="LAS",
        choices=["LAS", "Gradsort", "RF", "SOM", "Random", "KS", "IsoMatch", "TSNEH"],
        help="Algorithm to run."
    )
    args = parser.parse_args()

    dispatch = {
        "LAS":      las_rgb,
        "Gradsort": gradsort_rgb,
        "RF":       raster_fairy_rgb,
        "SOM":      self_organizing_maps_rgb,
        "Random":   random_rgb,
        "KS":       KS_rgb,
        "IsoMatch": isomatch_rgb,
        "TSNEH":    tsne_hungarian_rgb,
    }
    dispatch[args.method]()