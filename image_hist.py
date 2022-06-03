import os
import subprocess
from tqdm import tqdm
import numpy as np
import cv2
import argparse


def cal_image_hist(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]

    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges  # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]

    hist = cv2.calcHist([image], channels, None, histSize, ranges, accumulate=False)
    return hist


def color_balancing(source, source_hist, references, references_hists, output):
    source_images = os.listdir(source)

    r_images = os.listdir(references)
    r_hists = os.listdir(references_hists)

    if not os.path.exists(output):
        os.mkdir(output)

    for s_image in tqdm(source_images):
        s_image_path = os.path.join(source, s_image)
        s_hist_path = os.path.join(source_hist, s_image[:-3] + "txt")
        s_hist = np.loadtxt(s_hist_path, dtype=np.float32)

        best_image = r_images[0]
        best_corr = 0.0
        for r_hist_path in r_hists:
            r_hist = os.path.join(references_hists, r_hist_path)
            r_hist = np.loadtxt(r_hist, dtype=np.float32)

            corr = cv2.compareHist(s_hist, r_hist, 0)
            if corr - best_corr > 0.000001:
                best_image = r_hist_path[:-3] + "png"
                best_corr = corr

        r_image_path = os.path.join(references, best_image)
        o_image_path = os.path.join(output, s_image)

        subprocess.run(["rio", "hist", "--color-space", "RGB", s_image_path, r_image_path, o_image_path])


def cal_hist(input_dir, output_dir):
    files = os.listdir(input_dir)
    for img_file in files:
        img =cv2.imread(os.path.join(input_dir, img_file))
        hist = cal_image_hist(img)
        np.savetxt(os.path.join(output_dir, img_file[:-3] + "txt"), hist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument('--colorbalancing', action='store_true')
    parser.add_argument('--no-colorbalancing', dest='colorbalancing', action='store_false')
    parser.set_defaults(colorbalancing=True)
    parser.add_argument("-s", type=str)
    parser.add_argument("-sh", type=str)
    parser.add_argument("-r", type=str)
    parser.add_argument("-rh", type=str)
    parser.add_argument("-o", type=str)

    args = parser.parse_args()
    if args.colorbalancing:
        color_balancing(args.s, args.sh, args.r, args.rh, args.o)
    else:
        cal_hist(args.s, args.o)