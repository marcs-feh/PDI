import numpy as np

def apply_filter(img: np.ndarray, kernel: np.ndarray, mode: str = 'convolution') -> np.ndarray:
    if mode == 'convolution':
        out = np.convolve(img, kernel).clip(0, 1.0)
        return out
    elif mode == 'correlation':
        return img
    else:
        raise Exception(f'Invalid mode: {mode}')
        # out = np.correlate(img, )
    
