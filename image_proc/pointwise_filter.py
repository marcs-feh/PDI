import numpy as np
from image_proc.color import split_channels, join_channels

def threshold_rgb(img: np.ndarray, d_min: float, d_max: float, k: float = 1.0) -> np.ndarray:
    in_range = lambda x: (abs(x) > d_min) and (abs(x) < d_max)
    f = np.vectorize(lambda x: 0.0 if not in_range(x) else k)

    b, g, r = split_channels(img)
    out = join_channels([f(b), f(g), f(r)])

    return out

def threshold_gs(img: np.ndarray, d_min: float, d_max: float, k: float = 1.0) -> np.ndarray:
    in_range = lambda x: (abs(x) > d_min) and (abs(x) < d_max)
    f = np.vectorize(lambda x: 0.0 if in_range(x) else k)
    return f(img)

def threshold(img: np.ndarray, d_min: float, d_max: float, k: float = 1.0) -> np.ndarray:
    if len(img.shape) == 3:
        return threshold_rgb(img, d_min, d_max, k)
    else:
        return threshold_gs(img, d_min, d_max, k)

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

def brightness_and_contrast(img: np.ndarray, b: float, c: float) -> np.ndarray:
    out = (b + (img * c)).clip(0, 1.0)
    return out

def contrast_stretching(img: np.ndarray) -> np.ndarray:
    def stretch(px):
        if px <= (1/3):
            return px/2
        elif (px > (1/3)) and (px < (2/3)):
            return (2 * px) - 0.5
        else:
            return (px / 2) + 0.5
    
    out = np.vectorize(stretch)(img)
    return out.clip(0, 1.0)
