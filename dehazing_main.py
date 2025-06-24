import sys
import dehaze
import numpy as np
import skimage.io
import os
from skimage import img_as_ubyte

# ======= ここで画像パスと出力フォルダを指定 =======
input_image_path = "img05.jpg"
output_dir = "output"

patch_size = 15
top_portion = 0.001
t0 = 0.1
omega = 0.95
# ==============================================

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = skimage.io.imread(input_image_path).astype(np.float32)

    img_dehaze = dehaze.dehaze(
        img, patch_size, top_portion, t0, omega, output_dir
    )
    img_dehaze_uint8 = np.clip(img_dehaze, 0, 255).astype(np.uint8)
    skimage.io.imsave(os.path.join(output_dir, 'img05_black.jpg'), img_dehaze_uint8)

if __name__ == "__main__":
    main()
