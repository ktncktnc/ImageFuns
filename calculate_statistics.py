import numpy as np
import os
import cv2
import argparse


def calculate_mean_sdt(path):
    images = os.listdir(path)

    counter = 0
    means = np.array([0.0, 0.0, 0.0])
    stds = np.array([0.0, 0.0, 0.0])

    for img_path in images:
        image = cv2.imread(os.path.join(path, img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        means = means + np.mean(image, axis=(0, 1))
        stds = stds + np.power(np.std(image, axis=(0, 1)), 2)
        counter += 1

    means = means/counter
    stds = np.sqrt(stds/counter)
    print(means)
    print(stds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument("-p", type=str)
    args = parser.parse_args()
    calculate_mean_sdt(args.p)