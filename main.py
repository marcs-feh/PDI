from filter import *
from img_file import *
from color import *
from kernels import *
from noise import *
from time import sleep

def main():
    img = img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')
    out = img

    N = 14
    for i in range(0, int(N)):
        n = (1.0/N) * i
        out *= threshold(img, n)
    img_write('out.png', out)

if __name__ == '__main__': main()

