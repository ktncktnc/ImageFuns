import cv2
import numpy as np
import os
from PIL import Image, ImagePalette
import matplotlib.cm as matcm
def main(path='/root/xview/dataset/test/targets', savepath = '/root/xview/dataset/test/color_targets'):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    np.random.seed(42)
    colormap = np.zeros((256, 3))
    colormap[:5, :3] = matcm.jet(np.linspace(0, 1, 5))[:, :3]

    masks = os.listdir(path)

    for m in masks:
        mask = cv2.imread(os.path.join(path, m)).astype('int')[:, :, 0]
        mask = mask - 1
        mask[mask < 0] = 0

        cm_img = Image.fromarray(mask.astype(np.uint8))
        cm_img = cm_img.convert("P")
        cm_img.putpalette((colormap * 255).astype(np.uint8).flatten(), rawmode='RGB')
        cm_img.save(os.path.join(savepath, m))


if __name__ == '__main__':
    main()

