from frequency_filter import *
from spatial_filter import *
from pointwise_filter import *
from img_file import *
from color import *
from noise import *
from plots import *

def main():
    img = img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')
    # mag = magnitude_spectrum(img)
    out = grayscale_human_weighted(img)

    INNER = 10
    OUTER = 80
    mask = smooth_circular_mask(out.shape, INNER, OUTER)

    out = fft(out) * mask
    out = grayscale_human_weighted(img) * ifft(out)
    # out = laplacian_sharpening(out, 0.3)

    img_write('out.png', out)
    # img_write('mag_spec.out.png', mag)

if __name__ == '__main__': main()

