import numpy as np

def uniform_blur_mask(n: int, dtype=np.float32) -> np.ndarray:
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

laplacian_sharpening_mask = np.array([
    [+1, +1, +1],
    [+1, -8, +1],
    [+1, +1, +1],
], dtype=np.float32)

def gaussian_blur_mask(sigma:float, n: int = 3):
    x, y = np.meshgrid(np.linspace(-1,1,n, dtype=np.float32),
                       np.linspace(-1,1,n, dtype=np.float32))

    omega = 2 * (sigma ** 2)

    factor = 1/(np.pi * omega)
    exponent = -(((x*x) + (y*y))/omega)
    res = factor * np.exp(exponent)

    res /= res.sum()

    # assert (res.max() <= 1.0001)

    return res

