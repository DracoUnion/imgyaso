import subprocess as subp
import tempfile
import uuid
import os
from os import path
from PIL import Image, ImageFile
from io import BytesIO
import numpy as np
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMWRITE_PNG_FLAG = [cv2.IMWRITE_PNG_COMPRESSION, 9]
IMWRITE_PNG_BW_FLAG = [cv2.IMWRITE_PNG_BILEVEL, 1]

headers = {
    'png': b'\x89PNG',
    # 'svg': b'<?xml',
    'jpg': b'\xff\xd8\xff',
    'gif': b'GIF',
    'tif': b'II\x2a\x00',
    'bmp': b'BM',
    'webp': b'RIFF',
}

def is_img_data(img):
    for _, hdr in headers.items():
        l = len(hdr)
        if img[:l] == hdr:
            return True
    return False

def conv2png(img):
    if img[:4] == headers['png']:
        return img
    img = Image.open(BytesIO(img))
    bio = BytesIO()
    img.convert('RGBA').save(bio, 'png')
    return bio.getvalue()

def ensure_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
           if img.ndim == 3 else img