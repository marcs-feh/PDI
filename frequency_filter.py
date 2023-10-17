import numpy as np
from os.path import exists
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

def smooth_circular_mask(shape: tuple, inner: int, outer: int) -> np.ndarray:
    H = shape[0]
    W = shape[1]
    fname = f'mask-cache-{W}x{H}-{inner}-{outer}.tiff'
    if exists(fname):
        mask = img_read(fname, grayscale=True)
        return mask
    else:
        mask = circular_mask(shape, outer)
        for r in range(inner, outer):
            mask += circular_mask(shape, r)
        mask /= mask.max()
        img_write(fname, mask)
        return mask


