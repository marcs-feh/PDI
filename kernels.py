import numpy as np

def uniform_blur(n: int, dtype=np.float32) -> np.ndarray:
    kern = np.zeros((n, n), dtype=dtype)
    kern += (1 / (n * n))
    return kern

def sobel_edge_detect():
    gx = np.array([
        [-1, 0, +1],
        [-2, 0, +2],
        [-1, 0, +1],
    ], dtype=np.float32)
    gy = np.array([
        [+1, +2, +1],
        [0,  0,  0],
        [-1, -2, -1],
    ], dtype=np.float32)

    return [gx, gy]

