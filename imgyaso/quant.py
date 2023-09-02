import sys
import cv2
import libimagequant as liq
from PIL import Image, ImageFile
import numpy as np
from io import BytesIO
from .util import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pngquant_bts(img, ncolors=8):
    nparr_fmt = isinstance(img, np.ndarray)
    if nparr_fmt:
        img = bytes(cv2.imencode('.png', img, IMWRITE_PNG_FLAG)[1])

    if not is_img_data(img): return img
    img = conv2png(img)
    img = Image.open(BytesIO(img)).convert('RGBA')
    w, h = img.width, img.height
    bytes = img.tobytes()
    
    attr = liq.Attr()
    attr.max_colors = ncolors
    img = attr.create_rgba(bytes, w, h, 0)
    res = img.quantize(attr)
    res.dithering_level = 1.0
    bytes = res.remap_image(img)
    palette = res.get_palette()
    
    img = Image.frombytes('P', (w, h), bytes)
    
    palette_data = []
    for color in palette:
        if color.a == 0:
            palette_data.extend([255, 255, 255])
        else:
            palette_data.extend([color.r, color.g, color.b])
    img.putpalette(palette_data)
    
    bio = BytesIO()
    img.save(bio, 'PNG', optimize=True)
    img = bio.getvalue()
    
    if nparr_fmt:
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_UNCHANGED)
    return img

pngquant = pngquant_bts

def main():
    fname = sys.argv[1]
    img = open(fname, 'rb').read()
    img = pngquant_bts(img)
    with open(fname, 'wb') as f:
        f.write(img)
    
if __name__ == '__main__': main()