import image_proc as ip
import numpy as np

# TODO:
# Filtragem Espacial:
# - Edge dedect:
#   - Roberts
# OpenCV Canny (?????)

def main():
    img = ip.img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')

    out = ip.brightness_and_contrast(img, -1, 2)

    ip.img_write('out.png', out)

if __name__ == '__main__': main()

