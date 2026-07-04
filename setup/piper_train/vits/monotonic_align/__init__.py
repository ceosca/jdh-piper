"""Monotonic alignment search — versión numba (sin Cython/MSVC).

Reemplaza la extensión C (core.pyx) por una implementación JIT con numba, para
poder entrenar en Windows nativo sin Visual C++ Build Tools. Algoritmo idéntico
al core.pyx original.
"""
import numpy as np
import torch
from numba import njit, prange


@njit(cache=True)
def _maximum_path_each(path, value, t_y, t_x, max_neg_val=-1e9):
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


@njit(parallel=True, cache=True)
def maximum_path_c(paths, values, t_ys, t_xs):
    b = paths.shape[0]
    for i in prange(b):
        _maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])


def maximum_path(neg_cent, mask):
    """neg_cent: [b, t_t, t_s], mask: [b, t_t, t_s] — igual que la versión Cython."""
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)
    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
