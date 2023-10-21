import numpy as np
from os.path import exists
from pointwise_filter import negative
from img_file import img_write, img_read

def fft(img: np.ndarray):
    return np.fft.fftshift(np.fft.fft2(img))

def ifft(freq_space: np.ndarray):
    return np.abs(np.fft.ifft2(np.fft.fftshift(freq_space)))

def circular_mask(shape: tuple, radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)

    H = shape[0]
    W = shape[1]
    n = 0
    for row in range(0, H):
        for col in range(0, W):
            if (((H/2 - row)**2) + ((W/2 - col)**2)) <= (radius * radius):
                mask[row][col] = 1.0
                n += 1

    return mask

# def smooth_circular_mask(shape: tuple, inner: int, outer: int) -> np.ndarray:
#     H = shape[0]
#     W = shape[1]
#     fname = f'smooth-circle-mask-cache-{W}x{H}-{inner}:{outer}.tiff'
#     if exists(fname):
#         mask = img_read(fname, grayscale=True)
#         return mask
#     else:
#         mask = circular_mask(shape, outer)
#         for r in range(inner, outer):
#             mask += circular_mask(shape, r)
#         mask /= mask.max()
#         img_write(fname, mask)
#         return mask


def gaussian_mask(shape: tuple, radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)

    H = shape[0]
    W = shape[1]
    cr = (H//2)
    cc = (W//2)
    for row in range(0, H):
        for col in range(0, W):
            d = np.sqrt( ((row - cr) ** 2) + ((col - cc) ** 2))
            mask[row][col] = np.exp( (-(d**2)) / (2 * (radius**2)) )
    return mask

def butterworth_mask(shape: tuple, radius: int, order: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)

    H = shape[0]
    W = shape[1]
    p = order * 2

    cr = (H//2)
    cc = (W//2)
    for row in range(0, H):
        for col in range(0, W):
            d = np.sqrt( ((row - cr) ** 2) + ((col - cc) ** 2)) / radius
            mask[row][col] = 1 / (1 + (d ** p))

    return mask

def low_pass_filter(img: np.ndarray, radius: int) -> np.ndarray:
    freq = fft(img)
    mask = circular_mask(img.shape, radius)
    freq *= mask
    return ifft(freq)

def high_pass_filter(img: np.ndarray, radius: int) -> np.ndarray:
    freq = fft(img)
    mask = negative(circular_mask(img.shape, radius))
    freq *= mask
    return ifft(freq)

def low_pass_butterworth_filter(img:np.ndarray, radius: int, order: int) -> np.ndarray:
    freq = fft(img)
    mask = butterworth_mask(img.shape, radius, order)
    freq *= mask
    return ifft(freq)

def high_pass_butterworth_filter(img:np.ndarray, radius: int, order: int) -> np.ndarray:
    freq = fft(img)
    mask = negative(butterworth_mask(img.shape, radius, order))
    freq *= mask
    return ifft(freq)

def low_pass_gaussian_filter(img:np.ndarray, radius: int) -> np.ndarray:
    freq = fft(img)
    mask = gaussian_mask(img.shape, radius)
    freq *= mask
    return ifft(freq)

def high_pass_gaussian_filter(img:np.ndarray, radius: int) -> np.ndarray:
    freq = fft(img)
    mask = negative(gaussian_mask(img.shape, radius))
    freq *= mask
    return ifft(freq)

def magnitude_spectrum(img: np.ndarray, fact: float = 0.1) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fft2(img))
    spec = fact * np.log(np.abs(freq))
    return spec

def magnitude_spectrum_freq(freq: np.ndarray, fact: float = 0.1) -> np.ndarray:
    mag = fact * np.log(np.abs(freq))
    return mag

