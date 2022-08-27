import os
import random
import shutil
import argparse


def main(s: str, o: str, size=300):
    images = os.listdir(s)
    random.shuffle(images)
    for i in range(size):
        shutil.copyfile(os.path.join(s, images[i]), os.path.join(o, images[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument("-s", type=str)
    parser.add_argument("-o", type=str)
    parser.add_argument("--size", type=int, default=300)

    args = parser.parse_args()
    main(args.s, args.o, args.size)
