import numpy as np
from cv2 import imread as _imread
from cv2 import imwrite as _imwrite

U8_MAX = 0xff

def img_read(path: str) -> np.ndarray:
    img = _imread(path).astype(np.float32)
    img /= U8_MAX
    return img

def img_write(outpath: str, img: np.ndarray):
    denorm = (img * U8_MAX).clip(0, U8_MAX).astype(np.uint8)
    _imwrite(outpath, denorm)

