import random
import subprocess
import os
import argparse

def color_balancing(source, target1, target2, output1, output2):
    source_images = os.listdir(source)
    target_images = os.listdir(target1)

    if not os.path.exists(output1):
        os.mkdir(output1)

    if not os.path.exists(output2):
        os.mkdir(output2)

    for t_image in target_images[:200]:
        s_image = random.choice(source_images)

        t1_image_path = os.path.join(target1, t_image)
        t2_image_path = os.path.join(target2, t_image)
        s_image_path = os.path.join(source, s_image)
        o1_image_path = os.path.join(output1, t_image)
        o2_image_path = os.path.join(output2, t_image)

        subprocess.run(["rio", "hist", "--color-space", "RGB", t1_image_path, s_image_path, o1_image_path])
        subprocess.run(["rio", "hist", "--color-space", "RGB", t2_image_path, s_image_path, o2_image_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument("-s", type=str)
    parser.add_argument("-t1", type= str)
    parser.add_argument("-t2", type= str)
    parser.add_argument("-o1", type= str)
    parser.add_argument("-o2", type= str)

    args = parser.parse_args()
    color_balancing(args.s, args.t1, args.t2, args.o1, args.o2)