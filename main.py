
from numpy.ma import negative
import image_proc as ip
import numpy as np

from image_proc.spatial_filter import gaussian_blur


k = np.array(
    [
    [1,0,0,0,0],
    [1,1,1,0,0],
    [0,1,1,1,0],
    [0,0,1,1,1],
    [0,0,0,0,1],
    ],
    dtype=np.float32
); 

def main():
    img = ip.img_read('in.png')
    print(f'loaded {np.prod(img.shape) * 4} bytes')

    out = ip.grayscale_weighted(img)
    thresh = ip.threshold(out, 0.2, 1.0)
    e = ip.morph('erode', thresh, k)
    d = ip.morph('dilate', thresh, k)
    out = out - 0.15 * gaussian_blur(d - e)
    # out = ip.roberts_edge_detect(out)

    ip.img_write('out.png', out)

if __name__ == '__main__': main()

