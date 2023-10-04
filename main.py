from filter import apply_filter, apply_filters
from img_file import *
from color import *
from kernels import *
from noise import gaussian_noise

def main():
    img = img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')
    out = img
    gs = grayscale_human_weighted(img)
    # out *= rgb_from_grayscale(gaussian_noise(gs, 0.1))
    # out = apply_filter(out, uniform_blur(3))
    he, we = apply_filters(gs, sobel_edge_detect())
    out -= rgb_from_grayscale(he + we)
    img_write('out.png', out)

    return

if __name__ == '__main__': main()

