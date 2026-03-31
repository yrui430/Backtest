"""Numba-accelerated rolling computation utilities.

All rolling calculations use numba JIT — pandas rolling functions are forbidden
per project spec.
"""
from __future__ import annotations

import numpy as np
import numba


@numba.njit
def rolling_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """Rolling quantile using insertion sort within each window. No pandas."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        # Collect non-NaN values in window
        buf = np.empty(window, dtype=np.float64)
        count = 0
        for j in range(i - window + 1, i + 1):
            v = arr[j]
            if not np.isnan(v):
                buf[count] = v
                count += 1
        if count < 2:
            continue
        # Insertion sort
        for a in range(1, count):
            key = buf[a]
            b = a - 1
            while b >= 0 and buf[b] > key:
                buf[b + 1] = buf[b]
                b -= 1
            buf[b + 1] = key
        # Linear interpolation for quantile
        pos = q * (count - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, count - 1)
        frac = pos - lo
        out[i] = buf[lo] * (1.0 - frac) + buf[hi] * frac
    return out


@numba.njit
def rolling_pearson(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson correlation. No pandas."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        cnt = 0
        for j in range(i - window + 1, i + 1):
            xv = x[j]
            yv = y[j]
            if np.isnan(xv) or np.isnan(yv):
                continue
            sx += xv
            sy += yv
            sxx += xv * xv
            syy += yv * yv
            sxy += xv * yv
            cnt += 1
        if cnt < 3:
            continue
        mx = sx / cnt
        my = sy / cnt
        cov = sxy / cnt - mx * my
        vx = sxx / cnt - mx * mx
        vy = syy / cnt - my * my
        if vx <= 0.0 or vy <= 0.0:
            continue
        out[i] = cov / np.sqrt(vx * vy)
    return out


@numba.njit
def _rank_array(arr: np.ndarray, n: int) -> np.ndarray:
    """Rank values in arr[0:n], handling ties with average rank."""
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx[i] = i
    # Insertion sort by value
    for i in range(1, n):
        key_idx = idx[i]
        key_val = arr[key_idx]
        j = i - 1
        while j >= 0 and arr[idx[j]] > key_val:
            idx[j + 1] = idx[j]
            j -= 1
        idx[j + 1] = key_idx
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[idx[j + 1]] == arr[idx[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks


@numba.njit
def rolling_spearman(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling Spearman rank correlation. No pandas."""
    n = len(x)
    out = np.full(n, np.nan)
    xbuf = np.empty(window, dtype=np.float64)
    ybuf = np.empty(window, dtype=np.float64)
    for i in range(window - 1, n):
        cnt = 0
        for j in range(i - window + 1, i + 1):
            xv = x[j]
            yv = y[j]
            if np.isnan(xv) or np.isnan(yv):
                continue
            xbuf[cnt] = xv
            ybuf[cnt] = yv
            cnt += 1
        if cnt < 3:
            continue
        xranks = _rank_array(xbuf, cnt)
        yranks = _rank_array(ybuf, cnt)
        # Pearson on ranks
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for k in range(cnt):
            rx = xranks[k]
            ry = yranks[k]
            sx += rx
            sy += ry
            sxx += rx * rx
            syy += ry * ry
            sxy += rx * ry
        mx = sx / cnt
        my = sy / cnt
        cov = sxy / cnt - mx * my
        vx = sxx / cnt - mx * mx
        vy = syy / cnt - my * my
        if vx <= 0.0 or vy <= 0.0:
            continue
        out[i] = cov / np.sqrt(vx * vy)
    return out
