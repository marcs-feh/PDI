import numpy as np
from color import join_channels, rgb_from_grayscale, split_channels, grayscale_human_weighted
import kernels as kn

CONVOLVE = 0
CORRELATE = 1

def is_rgb(img: np.ndarray) -> bool:
    try:
        return img.shape[2] == 3
    except IndexError:
        return False

def median_filter(img: np.ndarray, neighbor_sq: int = 3,padding_val:int =0xff) -> np.ndarray:
    assert valid_kernel_shape((neighbor_sq, neighbor_sq))

    # Handle RGB

    if is_rgb(img):
        channels = split_channels(img)
        channels = [median_filter(ch, neighbor_sq, padding_val) for ch in channels]
        return join_channels(channels)

    off = neighbor_sq // 2

    padded = np.pad(img, (off, off), 'constant', constant_values=padding_val)

    height, width = padded.shape[0], padded.shape[1]
    out = np.ndarray(img.shape, dtype=np.float32)

    for row in range(off, height - off):
        for col in  range(off, width - off):
            slice = padded[row-off:row+off+1, col-off:col+off+1].flatten()
            # slice.sort()
            median = np.median(slice)
            out[row - off][col - off] = median

    return out

def unsharp_masking(img: np.ndarray, k: float = 1.0, blur_sigma: float = 1.0, blur_size: int = 3) -> np.ndarray:
    mask = k * (img - gaussian_blur(img, blur_size, blur_sigma))
    res = (img + mask).clip(0, 1.0)
    return res

def laplacian_sharpening(img: np.ndarray, c: float) -> np.ndarray:
    if len(img.shape) == 3:
        out = c * apply_filter(grayscale_human_weighted(img), kn.laplacian_sharpening_mask)
        out = rgb_from_grayscale(out)
        out += img
    else:
        out = c * apply_filter(img, kn.laplacian_sharpening_mask)
        out += img

    return out.clip(0, 1.0)

def gaussian_blur(img: np.ndarray, n: int = 3, sigma: float = 1.0) -> np.ndarray:
    kern = kn.gaussian_blur_mask(sigma, n)
    return apply_filter(img, kern)

def uniform_blur(img: np.ndarray, n: int) -> np.ndarray:
    return apply_filter(img, kn.uniform_blur_mask(n))


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

