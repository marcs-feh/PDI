import numpy as np

def uniform_blur(n: int, channels: int= 1, dtype=np.float32) -> np.ndarray:
    kern = None
    if channels == 1:
        kern = np.zeros((n, n), dtype=dtype)
    else:
        kern = np.zeros((n, n, channels), dtype=dtype)

    kern += (1 / (n * n))
    print(kern)
    return kern

