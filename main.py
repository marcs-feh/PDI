from filter import apply_filter
from img_file import *
from color import *
from kernels import uniform_blur
from noise import gaussian_noise

def main():
    img = img_read('in.png')
    print(f'loaded ({img.shape})')
    out = grayscale_human_weighted(img)
    out = gaussian_noise(out, 0.2)
    out = apply_filter(out, uniform_blur(5))
    img_write('out.png', out)

    return

if __name__ == '__main__': main()

