from typing import Any
import numpy as np

def apply_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    chans = []
    if len(img.shape) > 2:
        chans = split_channels(img)
        for i in range(0, len(chans)):
            chans[i] = convolve2D(chans[i], kernel).clip(0, 1.0)
        return join_channels(chans)
    else:
        out = convolve2D(img, kernel).clip(0, 1.0)
        return out

def apply_filters(img: np.ndarray, kernels: list[np.ndarray]) -> list[np.ndarray]:
    results = []
    for k in kernels:
        results.append(apply_filter(img, k))
    return results

def valid_kernel_shape(shape) -> bool:
    h, w = shape[0], shape[1]
    return ((h % 2) != 0) and (w == h)

def split_channels(img: np.ndarray) -> list[np.ndarray]:
    ch = img.shape[2]
    chans = np.dsplit(img, ch)
    return chans

def join_channels(chans: list[np.ndarray]) -> np.ndarray:
    stacked = np.dstack(chans)
    return stacked

# TODO: Improve perf, python `for` is way too slow
def convolve2D(img: np.ndarray, kernel: np.ndarray, padding_val=0) -> np.ndarray:
    assert valid_kernel_shape(kernel.shape)

    off = kernel.shape[0] // 2

    padded = np.pad(img, (off, off), 'constant', constant_values=padding_val)

    height, width = padded.shape[0], padded.shape[1]
    out = np.ndarray(img.shape, dtype=np.float32)

    for row in range(off, height - off):
        for col in  range(off, width - off):
            slice = padded[row-off:row+off+1, col-off:col+off+1]
            out[row - off][col - off] = (kernel * slice).sum()
            
    return out

