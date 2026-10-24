import numpy as np
from scipy import signal
import cv2
import re
import os
from os import path
import sys
from .util import *

def adathreas_quant(img, ncolor=8, win=9, beta=0.9):
    bytes_type = isinstance(img, bytes)
    if bytes_type:
        if not is_img_data(img): return img
        img = cv2.imdecode(np.frombuffer(img, np.uint8), 
                           cv2.IMREAD_COLOR)
        if img is None: return None
    img = ensure_bgr(img)

    # 2) BGR -> HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3) 归一化到 [0,1] 方便算
    H_norm = hsv[..., 0].astype(np.float32) / 180.0
    S_norm = hsv[..., 1].astype(np.float32) / 255.0

    # 4) V 通道二值化
    V_bin = adathres(hsv[..., 2], win, beta)

    # 5) HS 两维 KMeans（OpenCV 实现，需 float32 连续数组）
    HS_norm = np.ascontiguousarray(
        np.stack([H_norm.ravel(), S_norm.ravel()], axis=1), dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    _, labels, centers = cv2.kmeans(
        HS_norm, ncolor, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()


    # 6) 用簇中心替换 HS，然后还原 uint8（注意 H 上界是 179）
    H_quant = centers[labels, 0].reshape(H_norm.shape)
    S_quant = centers[labels, 1].reshape(S_norm.shape)
    H_quant = np.clip(H_quant * 180.0, 0, 179).round().astype(np.uint8)
    S_quant = np.clip(S_quant * 255.0, 0, 255).round().astype(np.uint8)

    # 7) 组装 HSV 
    hsv_quant = np.empty_like(hsv)
    hsv_quant[..., 0] = H_quant
    hsv_quant[..., 1] = S_quant
    hsv_quant[..., 2] = V_bin

    # 8) HSV -> BGR
    img_quant = cv2.cvtColor(hsv_quant, cv2.COLOR_HSV2BGR)
    if bytes_type:
        img_quant = bytes(cv2.imencode(
            '.png', img_quant,
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )[1])
    return img_quant

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
    img = np.where(img < means * beta, 0, 255).astype(np.uint8)
    
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