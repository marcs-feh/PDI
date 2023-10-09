from filter import *
from img_file import *
from color import *
from kernels import *
from noise import *
from plots import *

import cv2 as cv


def main():
    img = img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')
    # out = img
    out = grayscale_human_weighted(img)
    print(img.shape)
    plot_histogram(img)
    plot_histogram(out)


    out = magnitude_spectrum(out)

    # out = np.abs(np.fft.ifft2(out))

    img_write('out.png', out)

if __name__ == '__main__': main()

