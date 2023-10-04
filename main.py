from filter import apply_filter
from img_file import *
from color import *
from kernels import uniform_blur
from noise import gaussian_noise

def main():
    img = img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')
    out = img
    gs = grayscale_simple(img)
    out *= rgb_from_grayscale(gaussian_noise(gs, 0.2))
    out = apply_filter(out, uniform_blur(3))
    img_write('out.png', out)

    return

if __name__ == '__main__': main()

