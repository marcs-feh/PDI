import numpy as np
from image_proc.pointwise_filter import negative

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

def low_pass_ideal_filter(img: np.ndarray, radius: int) -> np.ndarray:
    freq = fft(img)
    mask = circular_mask(img.shape, radius)
    freq *= mask
    return ifft(freq)

def high_pass_ideal_filter(img: np.ndarray, radius: int) -> np.ndarray:
    freq = fft(img)
    mask = negative(circular_mask(img.shape, radius))
    freq *= mask
    return ifft(freq)

def band_pass_ideal_filter(img:np.ndarray, band: int, bandwidth: int, reject: bool = False):
    freq = fft(img)
    mask = circular_mask(img.shape, band + (bandwidth//2)) + negative(circular_mask(img.shape, band - (bandwidth//2)))
    if reject:
        mask = negative(mask)
    freq *= mask
    return ifft(freq)

def low_pass_butterworth_filter(img:np.ndarray, radius: int, order: int = 2) -> np.ndarray:
    freq = fft(img)
    mask = butterworth_mask(img.shape, radius, order)
    freq *= mask
    return ifft(freq)

def high_pass_butterworth_filter(img:np.ndarray, radius: int, order: int = 2) -> np.ndarray:
    freq = fft(img)
    mask = negative(butterworth_mask(img.shape, radius, order))
    freq *= mask
    return ifft(freq)

def band_pass_butterworth_filter(img:np.ndarray, band: int, bandwidth: int, order: int = 2, reject: bool = False):
    freq = fft(img)
    mask = (butterworth_mask(img.shape, band + (bandwidth//2), order)
         + negative(butterworth_mask(img.shape, band - (bandwidth//2), order)))

    if reject:
        mask = negative(mask)

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

def band_pass_gaussian_filter(img:np.ndarray, band: int, bandwidth: int, reject: bool = False):
    freq = fft(img)
    mask = (gaussian_mask(img.shape, band + (bandwidth//2))
         + negative(gaussian_mask(img.shape, band - (bandwidth//2))))

    if reject:
        mask = negative(mask)

    freq *= mask
    return ifft(freq)

def magnitude_spectrum(img: np.ndarray, fact: float = 0.1) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fft2(img))
    spec = fact * np.log(np.abs(freq))
    return spec

def magnitude_spectrum_freq(freq: np.ndarray, fact: float = 0.1) -> np.ndarray:
    mag = fact * np.log(np.abs(freq))
    return mag

