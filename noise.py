import numpy as np

def gaussian_noise(img:np.ndarray, std_dev: float, mean: float = 0) -> np.ndarray:
    nimg = (img + np.random.normal(mean, std_dev, img.shape)).clip(0, 1.0)
    return nimg

