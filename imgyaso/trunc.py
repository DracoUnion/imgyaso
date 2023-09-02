import sys
import cv2
import numpy as np
from .util import *

def trunc(img, l=4):
    bytes_type = isinstance(img, bytes)
    if bytes_type:
        if not is_img_data(img): return img
        img = cv2.imdecode(np.frombuffer(img, np.uint8), 
                           cv2.IMREAD_GRAYSCALE)
        if img is None: return None
    img = ensure_grayscale(img)
       
    colors = np.linspace(0, 255, l).astype(int)
    
    img_3d = np.expand_dims(img, 2)
    dist = np.abs(img_3d - colors)
    idx = np.argmin(dist, axis=2)
    img = colors[idx]
    
    return img

def main():
    fname = sys.argv[1]
    img = open(fname, 'rb').read()
    img = trunc_bts(img)
    with open(fname, 'wb') as f:
        f.write(img)
    
if __name__ == '__main__': main()