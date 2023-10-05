from filter import *
from img_file import *
from color import *
from kernels import *
from noise import *
from time import sleep

def main():
    img = img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')
    out = np.fft.fft2(img)

    out += threshold(out, 0.2)

    out = np.fft.ifft2(out)
    img_write('out.png', out)

if __name__ == '__main__': main()

