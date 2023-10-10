import numpy as np
from color import join_channels, split_channels

CONVOLVE = 0
CORRELATE = 1

def apply_filter(img: np.ndarray, kernel: np.ndarray, mode: int = CONVOLVE) -> np.ndarray:
    assert (mode == CONVOLVE) or (mode == CORRELATE)
    f = convolve2D if mode == CONVOLVE else correlate2D
    chans = []
    # RGB Mode
    if len(img.shape) == 3:
        chans = split_channels(img)
        for i in range(0, len(chans)):
            chans[i] = f(chans[i], kernel).clip(0, 1.0)
        return join_channels(chans)
    # Grayscale
    else:
        out = f(img, kernel).clip(0, 1.0)
        return out

def apply_filters(img: np.ndarray, kernels: list[np.ndarray], mode: int = CONVOLVE) -> list[np.ndarray]:
    results = []
    for k in kernels:
        results.append(apply_filter(img, k, mode=mode))
    return results



def valid_kernel_shape(shape) -> bool:
    h, w = shape[0], shape[1]
    return ((h % 2) != 0) and (w == h)


# TODO: Improve perf, python `for` is way too slow
def correlate2D(img: np.ndarray, kernel: np.ndarray, padding_val=0) -> np.ndarray:
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

def convolve2D(img: np.ndarray, kernel: np.ndarray, padding_val=0) -> np.ndarray:
    return correlate2D(img, kernel[::-1], padding_val=padding_val)

def magnitude_spectrum(img: np.ndarray, fact: float = 0.1) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fft2(img))
    spec = fact * np.log(np.abs(freq))
    return spec

def magnitude_spectrum_freq(freq: np.ndarray, fact: float = 0.1) -> np.ndarray:
    mag = fact * np.log(np.abs(freq))
    return mag

