import numpy as np
from color import split_channels, join_channels

def threshold_rgb(img: np.ndarray, t: float, k: float = 1.0) -> np.ndarray:
    f = np.vectorize(lambda x: 0.0 if abs(x) < t else k)

    b, g, r = split_channels(img)
    out = join_channels([f(b), f(g), f(r)])

    return out

def threshold_gs(img: np.ndarray, t: float, k: float = 1.0) -> np.ndarray:
    f = np.vectorize(lambda x: 0.0 if abs(x) < t else k)
    return f(img)

def threshold(img: np.ndarray, t: float, k: float = 1.0) -> np.ndarray:
    if len(img.shape) == 3:
        return threshold_rgb(img, t, k)
    else:
        return threshold_gs(img, t, k)

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
