import numpy as np

def grayscale_simple(img: np.ndarray) -> np.ndarray:
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gs = (b + g + r) / 3
    return gs

def grayscale_weighted(img: np.ndarray, rw: float = 1.0, gw: float = 1.0, bw: float = 1.0) -> np.ndarray:
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gs = ((b * bw) + (g * gw) + (r * rw)) / (rw + gw + bw)
    return gs

