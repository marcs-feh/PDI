# TODO: everything
from functools import reduce
import numpy as np

from image_proc.spatial_filter import valid_kernel_shape

MORPH_MODES = {
    'erode',
    'dilate',
}

def morph(mode:str, img: np.ndarray, kernel: np.ndarray, padding_val: float = 0):
    assert mode in MORPH_MODES
    assert valid_kernel_shape(kernel.shape)
    off = kernel.shape[0] // 2

    mask = kernel.flatten()
    padded = np.pad(img, (off, off), 'constant', constant_values=padding_val)

    height, width = padded.shape[0], padded.shape[1]
    out = np.ndarray(img.shape, dtype=np.float32)

    for row in range(off, height - off):
        for col in  range(off, width - off):
            slice = padded[row-off:row+off+1, col-off:col+off+1].flatten()
            pixels = []
            for i in range(0, len(slice)):
                if mask[i] != 0: pixels.append(slice[i])

            pixels = np.array(pixels, dtype=np.float32)

            ok = False
            count = (pixels > 0).sum()
            if mode == 'erode':
                n = len(pixels)
                ok = count >= n
            elif mode == 'dilate':
                ok = count > 0

            out[row - off][col - off] = 1.0 if ok else 0.0

    return out

