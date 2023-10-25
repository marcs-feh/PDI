import image_proc as ip
import numpy as np

# TODO:
# Filtragem Espacial:
# OpenCV Canny (?????)

def main():
    img = ip.img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')

    out = ip.grayscale_weighted(img)
    out = ip.roberts_edge_detect(out)

    ip.img_write('out.png', out)

if __name__ == '__main__': main()

