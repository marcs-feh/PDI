import numpy as np

def uniform_blur(n: int, dtype=np.float32) -> np.ndarray:
    kern = np.zeros((n, n), dtype=dtype)
    kern += (1 / (n * n))
    return kern

