from spatial_filter import *
from pointwise_filter import *
from img_file import *
from color import *
from noise import *
from plots import *

def fft(img: np.ndarray):
    return np.fft.fftshift(np.fft.fft2(img))

def ifft(freq_space: np.ndarray):
    return np.abs(np.fft.ifft2(np.fft.fftshift(freq_space)))

def main():
    img = img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')
    # mag = magnitude_spectrum(img)
    out = grayscale_human_weighted(img)

    out = gaussian_blur(out, sigma=1.5)
    out = laplacian_sharpening(out, 0.3)

    img_write('out.png', out)
    # img_write('mag_spec.out.png', mag)

if __name__ == '__main__': main()

