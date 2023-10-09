from filter import *
from img_file import *
from color import *
from kernels import *
from noise import *

import cv2 as cv

def magnitude_spectrum(img: np.ndarray, fact: float = 0.1) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fft2(img))
    spec = fact * np.log(np.abs(freq))
    return spec

def magnitude_spectrum_freq(freq: np.ndarray, fact: float = 0.1) -> np.ndarray:
    mag = fact * np.log(np.abs(freq))
    return mag

def main():
    img = img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')
    # out = img
    out = grayscale_human_weighted(img)


    out = magnitude_spectrum(out)

    # out = np.abs(np.fft.ifft2(out))

    img_write('out.png', out)

if __name__ == '__main__': main()

