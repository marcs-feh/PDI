import numpy as np

from filter import join_channels

def grayscale_simple(img: np.ndarray) -> np.ndarray:
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gs = (b + g + r) / 3
    return gs

def grayscale_weighted(img: np.ndarray, rw: float = 1.0, gw: float = 1.0, bw: float = 1.0) -> np.ndarray:
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gs = ((b * bw) + (g * gw) + (r * rw)) / (rw + gw + bw)
    return gs

def grayscale_human_weighted(img: np.ndarray) -> np.ndarray:
    return grayscale_weighted(img, 0.299, 0.587, 0.114)

def rgb_from_grayscale(img: np.ndarray) -> np.ndarray:
    assert len(img.shape) < 3

    return join_channels([img,img,img])

def negative(img: np.ndarray) -> np.ndarray:
    out = 1.0 - img
    return out

def log_transform(img: np.ndarray, c: float = 1.0) -> np.ndarray:
    out = c * np.log(1.0 + img)
    return out.clip(0, 1.0)

def gamma(img: np.ndarray, gamma_val: float, c: float = 1.0) -> np.ndarray:
    out = c * (img ** gamma_val)
    return out.clip(0, 1.0)

def brightness(img: np.ndarray, v: float) -> np.ndarray:
    out = (img + v).clip(0, 1.0)
    return out

def contrast(img: np.ndarray, v: float) -> np.ndarray:
    out = (img * v).clip(0, 1.0)
    return out

def brightness_and_contrast(img: np.ndarray, c: float, b: float) -> np.ndarray:
    out = (b + (img * c)).clip(0, 1.0)
    return out

