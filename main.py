from numpy.ma import negative
import image_proc as ip
import numpy as np
from image_proc.morphological_filter import morph
from image_proc.pointwise_filter import threshold

from image_proc.spatial_filter import gaussian_blur

k = np.array(
    [[1,1,1],
    [1,1,1],
    [1,1,1]],
    dtype=np.float32
); 

def main():
    img = ip.img_read('in.png')
    gs = ip.grayscale_weighted(img)
    out = threshold(gs, 0.3, 1.1)
    print(f'loaded {np.prod(img.shape) * 4} bytes')

    out = morph('open', out, k)
    out = morph('close', out, k)
    out *= gs

    ip.img_write('out.png', out)

if __name__ == '__main__': main()

