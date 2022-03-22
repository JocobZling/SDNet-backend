import torch
import torch.quantization
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision
import os
from torchvision import utils as vutils
import sys

import numpy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def randomImage(image):
    image_1 = torch.tensor(np.random.uniform(0, 20, image.shape))
    return image_1


# 读取一张测试
def readImg(img_path, savePosition1, savePosition2):
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    image = Image.open(img_path)

    image = data_transform(image)

    image_1 = torch.tensor(np.random.uniform(0, 255, image.shape))
    image_2 = image - image_1

    image_1 = torchvision.utils.make_grid(image_1, padding=0, normalize=False, range=None, scale_each=False)
    image_2 = torchvision.utils.make_grid(image_2, padding=0, normalize=False, range=None, scale_each=False)
    # vutils.save_image(image_1, 'D:\\pythonStudy\\zlProject\\ImageTest\\A_a.png', normalize=False)
    # vutils.save_image(image_2, 'D:\\pythonStudy\\zlProject\\ImageTest\\B_b.png', normalize=False)
    vutils.save_image(image_1, savePosition1, normalize=True)
    vutils.save_image(image_2, savePosition2, normalize=True)
    return "ok"

img_size = {"B0": 224,
            "B1": 240,
            "B2": 260,
            "B3": 300,
            "B4": 380,
            "B5": 456,
            "B6": 528,
            "B7": 600}
num_model = "B0"

data_transform = transforms.Compose([
    # transforms.Resize(img_size[num_model]),
    # transforms.CenterCrop(img_size[num_model]),
    transforms.ToTensor()
])

if __name__ == '__main__':
    # img_path = "D:\\文件资料\\论文\\数据集\\test\\0_fake\\Tp_D_CRN_M_N_cha00050_cha00026_11787.png"
    a = []
    for i in range(1, len(sys.argv)):
        a.append(sys.argv[i])
    print(readImg(a[0], a[1], a[2]))
