from filter import apply_filter
from img_file import *
from color import grayscale_simple, grayscale_weighted
from noise import gaussian_noise

def main():
    uniform_blur = np.array([[1,1,1], [1,1,1], [1,1,1]], dtype=np.float32) * (1/9)
    img = img_read('in.png')
    out = grayscale_weighted(img, 0.299, 0.587, 0.114)
    out = gaussian_noise(out, 0.2)
    out = apply_filter(img, uniform_blur)
    # out = grayscale_simple(img)
    img_write('out.png', out)

    return

if __name__ == '__main__': main()

