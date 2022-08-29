import argparse
import os
import random


def main(s: str, o: str, train_percent=0.85):
    images = os.listdir(s)
    random.shuffle(images)

    size = len(images)
    train_images = images[:int(size*train_percent)]
    test_images = images[int(size*train_percent):]

    train_file = open(os.path.join(o, 'train.txt'), 'w')
    test_file = open(os.path.join(o, 'test.txt'), 'w')
    for img in train_images:
        train_file.write("%s\n" % os.path.join(s, img))
    for img in test_images:
        test_file.write("%s\n" % os.path.join(s, img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alabama train test split")
    parser.add_argument("-s", type=str)
    parser.add_argument("-o", type=str)
    parser.add_argument("-p", type=float, default=0.85)
    args = parser.parse_args()
    main(args.s, args.o, args.p)
