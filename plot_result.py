import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

img1_path = 'D:/S2Looking/test/Image1'
img2_path = 'D:/S2Looking/test/Image2'
# img3_path = 'D:/HOC/images/bettermask/label'
# img4_path = 'D:/HOC/images/bettermask/cd'

lbl1_path = 'D:/S2Looking/test/label'
lbl2_path = 'D:/HOC/Thesis result/CD/s2looking_scratch_no_segmentationt_normal_dset/cd'
lbl3_path = 'D:/HOC/Thesis result/CD/s2looking_scratch_segmentation_input_normal_dset_epoch_20_prev/cd'

imgs = [268, 283, 323, 383]
def main():
    cols = ['Image 1', 'Image 2', 'Groundtruth', 'Baseline mask', 'Object-level model mask']
    #rows = ['Row {}'.format(row) for row in ['A', 'B', 'C', 'D']]

    fig, axes = plt.subplots(nrows=len(imgs), ncols=5, figsize=(20, 16))

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=20)

    for ax, row in zip(axes[:,0], imgs):
        print(row)
        ax.set_ylabel(row, fontsize=20)

    i = 0
    for img in imgs:
        i1 = Image.open(os.path.join(img1_path, str(img) + '.png'))
        i2 = Image.open(os.path.join(img2_path, str(img) + '.png'))

        l1 = Image.open(os.path.join(lbl1_path, str(img) + '.png'))
        l2 = Image.open(os.path.join(lbl2_path, "baseline-cd_" + str(img) + '.png'))
        l3 = Image.open(os.path.join(lbl3_path, "cd_" + str(img) + '.png'))

        axes[i, 0].imshow(i1)
        axes[i, 1].imshow(i2)
        axes[i, 2].imshow(l1, cmap=plt.cm.gray)
        axes[i, 3].imshow(l2, cmap=plt.cm.gray)
        axes[i, 4].imshow(l3, cmap=plt.cm.gray)

        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])
        axes[i, 4].set_xticks([])
        axes[i, 4].set_yticks([])

        i += 1
    
    fig.tight_layout()
    #plt.show()
    plt.savefig("D:/objectlevelbetter.png")
        
if __name__ == '__main__':
    main()

        