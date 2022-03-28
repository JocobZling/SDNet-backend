import cv2
import torch
import torch.quantization
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision
import os
from matplotlib import pyplot as plt
from torchvision import utils as vutils
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def plotHist(img_path, savePosition):
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = cv2.imread(img_path)
    chans = cv2.split(img)
    linestyles = ('-', ':', '-.')
    plt.figure()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel("Pixel Range", font1)
    plt.ylabel("Pixel Count", font1)

    colors = ('r', 'g', 'b')
    for (chan, color, linestyle) in zip(chans, colors, linestyles):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color, linestyle=linestyle)
        plt.xlim([0, 256])

    plt.legend(labels=['r', 'g', 'b'], ncol=1, prop=font1)
    plt.savefig(savePosition, pad_inches=0.1, bbox_inches='tight')


# 读取一张测试
def readImg(img_path, savePosition1, savePosition2, histPositon1, histPosition2, histPosition3):
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    plotHist(img_path, histPositon1)
    image = Image.open(img_path)

    image = data_transform(image)

    image_1 = torch.tensor(np.random.uniform(0, 255, image.shape))
    image_2 = image - image_1

    image_1 = torchvision.utils.make_grid(image_1, padding=0, normalize=False, range=None, scale_each=False)
    image_2 = torchvision.utils.make_grid(image_2, padding=0, normalize=False, range=None, scale_each=False)

    vutils.save_image(image_1, savePosition1, normalize=True)
    vutils.save_image(image_2, savePosition2, normalize=True)
    plotHist(savePosition1, histPosition2)
    plotHist(savePosition2, histPosition3)

    return "completed!"


data_transform = transforms.Compose([
    transforms.ToTensor()
])

if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a.append(sys.argv[i])
    print(readImg(a[0], a[1], a[2], a[3], a[4], a[5]))
