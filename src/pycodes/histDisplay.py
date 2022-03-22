from matplotlib import pyplot as plt
import cv2
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def plotHist(img_path, savePosition):
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = cv2.imread(img_path)
    chans = cv2.split(img)
    linestyles = ('-', ':', '-.')
    # makers=('o','v','s')
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


if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a.append(sys.argv[i])
    print(plotHist(a[0], a[1]))


