import numpy as np
from cv2 import imread as _imread
from cv2 import imwrite as _imwrite

U8_MAX = 0xff

def img_read(path: str) -> np.ndarray:
    img = _imread(path)
    return img_normalize(img)

def img_normalize(img: np.ndarray):
    out = img.astype(np.float32)
    out /= U8_MAX
    return out

def img_denormalize(img: np.ndarray):
    return (img * U8_MAX).clip(0, U8_MAX).astype(np.uint8)

def img_write(outpath: str, img: np.ndarray):
    denorm = img_denormalize(img)
    _imwrite(outpath, denorm)

