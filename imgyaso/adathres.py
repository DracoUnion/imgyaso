import numpy as np
from scipy import signal
import cv2
import re
import os
from os import path
import sys
from .util import *

def adathres(img, win=9, beta=0.9):
    bytes_type = isinstance(img, bytes)
    if bytes_type:
        if not is_img_data(img): return img
        img = cv2.imdecode(np.frombuffer(img, np.uint8), 
                           cv2.IMREAD_GRAYSCALE)
        if img is None: return None
    img = ensure_grayscale(img)
   
    if win % 2 == 0: win = win - 1
    # 边界的均值有点麻烦
    # 这里分别计算和和邻居数再相除
    kern = np.ones([win, win])
    sums = signal.correlate2d(img, kern, 'same')
    cnts = signal.correlate2d(np.ones_like(img), kern, 'same')
    means = sums // cnts
    # 如果直接采用均值作为阈值，背景会变花
    # 但是相邻背景颜色相差不大
    # 所以乘个系数把它们过滤掉
    img = np.where(img < means * beta, 0, 255)
    
    if bytes_type:
        img = bytes(cv2.imencode('.png', img, IMWRITE_PNG_BW_FLAG)[1])
    return img
    
adathres_bts = adathres
    
def main():
    fname = sys.argv[1]
    img = open(fname, 'rb').read()
    img = adathres_bts(img)
    with open(fname, 'wb') as f:
        f.write(img)

if __name__ == '__main__': main()