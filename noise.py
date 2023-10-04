import numpy as np

def gaussian_noise(img:np.ndarray, std_dev: float, mean: float = 0) -> np.ndarray:
    noise = np.random.normal(mean, std_dev, img.shape)
    nimg = (img + noise).clip(0, 1.0)
    return nimg

