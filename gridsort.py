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

Usage
-----
    python gridsort.py --method [LAS|Gradsort|RF|SOM|Random|KS]
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


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a 2D grid-sorting algorithm on a random 32×32 RGB image."
    )
    parser.add_argument(
        "--method",
        default="LAS",
        choices=["LAS", "Gradsort", "RF", "SOM", "Random", "KS"],
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
    }
    dispatch[args.method]()