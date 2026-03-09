"""
benchmark.py

Runs all grid-sorting algorithms on the same 32×32 random RGB image,
saves each result to ./results/, and prints a comparison table of
run times and DPQ scores.

Usage
-----
    python benchmark.py
"""

import os
import time
import random
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from minisom import MiniSom
from sklearn.manifold import TSNE
import rasterfairy
from rasterfairy import rfoptimizer
from KernelizedSorting_master.kernelized_sorting_color import KS

from rgb_las import distance_preservation_quality, sort_with_las
from gridsort import (
    generate_random_colors,
    generate_random_colors_numpy,
    set_global_seed,
    Gumbel_Sinkhorn_Transformer_Network,
    neighbor_loss_func,
    dist_matrix_loss_func,
    constraint_loss,
    device,
)

from isomatch import isomatch_algorithm


GRID_SIZE  = 32
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save(name, array):
    """Save a (H, W, 3) uint8 array as a PNG into the results folder."""
    Image.fromarray(array.astype(np.uint8)).save(
        os.path.join(RESULTS_DIR, f"{name}.png")
    )


def run_las(data, grid_size):
    X_grid = data.reshape(grid_size, grid_size, -1)
    sorted_grid, perm = sort_with_las(X_grid.copy(), radius_factor=0.95, wrap=False)
    result = data[perm].reshape(grid_size, grid_size, -1)
    dpq = distance_preservation_quality(result, p=16)
    return result, dpq


def run_gradsort(data, grid_size):
    seed          = 0
    iterations    = 5000
    learning_rate = 0.04
    n_iter        = 15
    tau           = 1
    noise_factor  = 0.1
    p_dist        = 2

    x_orig = torch.from_numpy(data).float() / 255
    ny, nx = grid_size, grid_size
    n = nx * ny

    set_global_seed(seed)
    x_shuffled = x_orig.reshape(-1, 3).to(device)

    net = Gumbel_Sinkhorn_Transformer_Network(
        input_dim=3, n=n, d_model=16, num_layers=3, n_iter=n_iter, n_heads=4
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for iteration in range(iterations + 1):
        optimizer.zero_grad()
        alpha = np.clip(iteration / (iterations + 1), 0, 1)
        perm_soft, perm_hard = net(x_shuffled, tau, noise_factor)
        x_sorted_soft = torch.matmul(perm_soft, x_shuffled)

        loss  = neighbor_loss_func(x_shuffled, x_sorted_soft, nx, p_dist)
        loss += 5 * alpha * dist_matrix_loss_func(x_shuffled, x_sorted_soft, p_dist)
        loss += 100 * constraint_loss(perm_soft)

        ind = torch.argmax(perm_hard, -1)
        num_duplicates = int(n - torch.unique(ind).shape[0])
        end = (iteration == iterations) or (num_duplicates == 0)
        if end:
            dist = torch.cdist(x_sorted_soft, x_shuffled, p=p_dist)
            _, ind = linear_sum_assignment(dist.detach().cpu().numpy())
            break

        loss.backward()
        optimizer.step()

    result = data[ind].reshape(grid_size, grid_size, -1)
    dpq = distance_preservation_quality(result, p=16)
    return result, dpq


def run_raster_fairy(data, grid_size):
    distance_matrix = cdist(data, data, metric='euclidean')
    tsne = TSNE(perplexity=20, metric="precomputed", verbose=False,
                max_iter=5000, init="random")
    points_2d = tsne.fit_transform(distance_matrix)

    grid_xy_tuple = rasterfairy.transformPointCloud2D(points_2d, target=(grid_size, grid_size))
    grid_xy    = grid_xy_tuple[0]
    grid_shape = grid_xy_tuple[1]

    optimizer  = rfoptimizer.SwapOptimizer()
    swapTable  = optimizer.optimize(points_2d, grid_xy, grid_size, grid_size, 1_000_000)

    grid_xy      = grid_xy[swapTable]
    grid_indices = grid_xy[:, 0] * grid_shape[1] + grid_xy[:, 1]
    combined     = list(zip(range(grid_size * grid_size), grid_indices.tolist()))
    perm         = [e for e, _ in sorted(combined, key=lambda x: x[1])]

    result = data[perm].reshape(grid_size, grid_size, -1)
    dpq = distance_preservation_quality(result, p=16)
    return result, dpq


def run_som(data, grid_size):
    w, h, size = grid_size, grid_size, 3
    som = MiniSom(h, w, size, learning_rate=0.4, sigma=5,
                  neighborhood_function='gaussian')
    som.train_random(data, 100_000, verbose=False)
    win_map = som.win_map(data, return_indices=True)

    new_order = np.full(w * h, -1, dtype=int)
    collided  = []
    for i in range(h):
        for j in range(w):
            pos = (i, j)
            if pos in win_map:
                indices = win_map[pos]
                new_order[i * w + j] = indices[0]
                collided.extend(indices[1:])

    empty_positions = np.where(new_order == -1)[0]
    if len(collided) > 0 and len(empty_positions) > 0:
        empty_coords    = np.array([np.unravel_index(p, (h, w)) for p in empty_positions])
        collided_coords = np.array([som.winner(data[idx]) for idx in collided])
        cost_matrix     = cdist(collided_coords, empty_coords, metric='euclidean')
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_ind, col_ind):
            new_order[empty_positions[col]] = collided[row]

    result = data[new_order].reshape(h, w, size)
    dpq = distance_preservation_quality(result, p=16)
    return result, dpq


def run_random(data, grid_size):
    perm = list(range(grid_size * grid_size))
    random.shuffle(perm)
    result = data[perm].reshape(grid_size, grid_size, -1)
    dpq = distance_preservation_quality(result, p=16)
    return result, dpq


def run_ks(data, grid_size):
    ny, nx = grid_size, grid_size
    griddata = np.zeros((2, ny * nx))
    griddata[0, ] = np.kron(range(1, ny + 1), np.ones((1, nx)))
    griddata[1, ] = np.tile(range(1, nx + 1), (1, ny))

    PI = KS(data, griddata.T)
    perm = PI.argmax(axis=1)

    result = data[perm].reshape(grid_size, grid_size, -1)
    dpq = distance_preservation_quality(result, p=16)
    return result, dpq


def run_isomatch(data, grid_size):
    from scipy.spatial.distance import pdist, squareform
    # IsoMatch expects float features in [0,1]
    data_norm = data.astype(np.float64) / 255.0
    d_matrix  = squareform(pdist(data_norm))
    perm, _, _ = isomatch_algorithm(d_matrix, grid_size=(grid_size, grid_size), num_swaps=10000)
    result = data[perm].reshape(grid_size, grid_size, -1)
    dpq = distance_preservation_quality(result, p=16)
    return result, dpq



ALGORITHMS = [
    ("Random",     run_random),
    ("LAS",        run_las),
    ("GradSort",   run_gradsort),
    ("RasterFairy",run_raster_fairy),
    ("SOM",        run_som),
    ("KS",         run_ks),
    ("IsoMatch",   run_isomatch),
]

def main():
    # Generate the shared input — same seed for all algorithms
    data = generate_random_colors_numpy(GRID_SIZE, GRID_SIZE).reshape(-1, 3)

    results = []
    for name, fn in ALGORITHMS:
        print(f"Running {name}...", flush=True)
        t0 = time.perf_counter()
        grid, dpq = fn(data.copy(), GRID_SIZE)
        elapsed = time.perf_counter() - t0
        save(name.lower().replace(" ", "_"), grid)
        results.append((name, elapsed, dpq))
        print(f"  Done in {elapsed:.1f}s  |  DPQ: {dpq:.4f}")

    # Print comparison table
    col_w = 14
    print("\n" + "─" * (col_w * 3 + 2))
    print(f"{'Algorithm':<{col_w}}{'Time (s)':>{col_w}}{'DPQ (p=16)':>{col_w}}")
    print("─" * (col_w * 3 + 2))
    for name, elapsed, dpq in sorted(results, key=lambda x: -x[2]):
        print(f"{name:<{col_w}}{elapsed:>{col_w}.2f}{dpq:>{col_w}.4f}")
    print("─" * (col_w * 3 + 2))
    print(f"\nSorted grids saved to ./{RESULTS_DIR}/")

    # ── Build composite image ─────────────────────────────────────────────────
    # Layout: Random centred on top, then 2 rows × 3 cols for the 6 algos
    grids = {}
    for name, fn in ALGORITHMS:
        fname = os.path.join(RESULTS_DIR, f"{name.lower().replace(' ', '_')}.png")
        grids[name] = np.array(Image.open(fname))

    cell_px    = GRID_SIZE
    label_h    = 14
    pad        = 6
    cell_w     = cell_px + pad
    cell_h     = cell_px + label_h + pad

    cols, rows = 3, 2
    algo_names = ["LAS", "GradSort", "RasterFairy", "SOM", "KS", "IsoMatch"]

    total_w = cols * cell_w + pad
    total_h = (cell_h + pad) + rows * cell_h + pad

    canvas = np.full((total_h, total_w, 3), 30, dtype=np.uint8)

    from PIL import ImageDraw, ImageFont

    def paste_cell(canvas, img_arr, x, y, label):
        canvas_img = Image.fromarray(canvas)
        canvas_img.paste(Image.fromarray(img_arr), (x, y))
        draw = ImageDraw.Draw(canvas_img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 8)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (cell_px - text_w) // 2, y + cell_px + 2), label, fill=(220, 220, 220), font=font)
        return np.array(canvas_img)

    # Top row: Random centred
    random_x = (total_w - cell_px) // 2
    random_y = pad
    canvas = paste_cell(canvas, grids["Random"], random_x, random_y, "Random (baseline)")

    # 2×3 grid for the 6 algorithms
    for i, name in enumerate(algo_names):
        row = i // cols
        col = i  % cols
        x = pad + col * cell_w
        y = pad + cell_h + pad + row * cell_h
        # find dpq for label
        dpq_val = next(d for n, _, d in results if n == name)
        canvas = paste_cell(canvas, grids[name], x, y, f"{name}  DPQ {dpq_val:.3f}")

    out_path = os.path.join(RESULTS_DIR, "comparison.png")
    Image.fromarray(canvas).save(out_path)
    print(f"Composite image saved to {out_path}")


if __name__ == "__main__":
    main()