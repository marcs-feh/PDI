from frequency_filter import *
from spatial_filter import *
from pointwise_filter import *
from img_file import *
from color import *
from noise import *
from plots import *

# TODO:
# Filtragem Espacial:
# - Edge dedect:
#   - Roberts
# OpenCV Canny (?????)

def fft(img: np.ndarray):
    return np.fft.fftshift(np.fft.fft2(img))

def ifft(freq_space: np.ndarray):
    return np.abs(np.fft.ifft2(np.fft.fftshift(freq_space)))

def main():
    img = img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')

    out = grayscale_human_weighted(img)

    out = band_pass_gaussian_filter(out, 10, 10)

    img_write('out.png', out)

if __name__ == '__main__': main()

