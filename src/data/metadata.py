"""
Core logic for metadata generation and coordinate extraction.
"""
import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def find_valid_patches_numba(frame: np.ndarray, patch_size: int, stride: int):
    """
    Scans a 2D physical field for valid patches containing no NaN values.
    Uses JIT compilation for nested loop acceleration.
    """
    frame_h, frame_w = frame.shape
    out_y, out_x, out_max = [], [], []

    for y in range(0, frame_h - patch_size + 1, stride):
        for x in range(0, frame_w - patch_size + 1, stride):
            has_nan = False
            patch_max = -np.inf

            for dy in range(patch_size):
                for dx in range(patch_size):
                    val = frame[y + dy, x + dx]
                    if np.isnan(val):
                        has_nan = True
                        break
                    if val > patch_max:
                        patch_max = val
                if has_nan:
                    break

            if not has_nan:
                out_y.append(y)
                out_x.append(x)
                out_max.append(patch_max)

    return out_y, out_x, out_max