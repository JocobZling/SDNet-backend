from datetime import datetime
import torch.nn as nn
import torch
import torch.quantization
import torchvision
import torchvision.transforms as transforms
from torch.multiprocessing import Manager
import numpy as np
import random
import MyResNet50
from PIL import Image
import sys
import os
import json

from redis import ConnectionPool, Redis

pool = ConnectionPool(host='localhost', port=6379, db=0, decode_responses=True, encoding='UTF-8')

tran = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def generate_share2(Conv, a, b, c):
    A = torch.ones_like(Conv) * a
    B = torch.ones_like(Conv) * b
    C = torch.ones_like(Conv) * c

    V = torch.ones_like(Conv) * random.uniform(0, 1)
    Alpha1 = Conv - A
    Beta1 = V - B

    return A.data, B.data, C.data, V.data, Alpha1.data, Beta1.data


def serverSigmoidAll1(u1, c1_mul, c1_res, t1, a1, b1, c1, dict_manager, event1, event2, event3, event4, event5, event6):
    x = torch.exp(-u1)
    A1 = x / c1_mul

    # exp
    dict_manager.update({'A1': A1})

    event2.set()
    event1.wait()

    A2 = dict_manager['A2']

    A = A1.mul(A2)
    e_u1 = A.mul(c1_res)

    x1 = torch.ones_like(u1) * 1 / 2
    y1 = e_u1 + torch.ones_like(u1) * 1 / 2

    # div
    # mul
    a1, b1, c1, _, Alpha1, Beta1 = generate_share2(y1, a1, b1, c1)

    e1 = y1 - a1
    f1 = t1 - b1
    e3 = x1 - a1
    f3 = t1 - b1

    dict_manager.update({'e1': e1, 'f1': f1, 'e3': e3, 'f3': f3})

    event4.set()
    event3.wait()
    event2.clear()

    e2 = dict_manager['e2']
    f2 = dict_manager['f2']
    e4 = dict_manager['e4']
    f4 = dict_manager['f4']

    eA = e1 + e2
    fA = f1 + f2
    e = e3 + e4
    f = f3 + f4

    ty1 = c1 + b1 * eA + a1 * fA + eA * fA
    tx1 = c1 + b1 * e + a1 * f + e * f

    dict_manager.update({'ty1': ty1})

    event6.set()
    event5.wait()
    event4.clear()

    ty2 = dict_manager['ty2']
    event6.clear()

    ty = ty1 + ty2
    res1 = tx1 / ty

    return res1


def serverSigmoidAll2(u2, c2_mul, c2_res, t2, a2, b2, c2, dict_manager, event1, event2, event3, event4, event5, event6):
    # exp
    x = torch.exp(-u2)
    A2 = x / c2_mul

    dict_manager.update({'A2': A2})

    event1.set()
    event2.wait()

    A1 = dict_manager['A1']

    A = A1.mul(A2)
    e_u2 = A.mul(c2_res)

    x2 = torch.ones_like(u2) * 1 / 2
    y2 = e_u2 + torch.ones_like(u2) * 1 / 2

    # div
    # mul1
    a2, b2, c2, _, Alpha1, Beta1 = generate_share2(y2, a2, b2, c2)
    e2 = y2 - a2
    f2 = t2 - b2
    e4 = x2 - a2
    f4 = t2 - b2

    dict_manager.update({'e2': e2, 'f2': f2, 'e4': e4, 'f4': f4})

    event3.set()
    event4.wait()
    event1.clear()

    e1 = dict_manager['e1']
    f1 = dict_manager['f1']
    e3 = dict_manager['e3']
    f3 = dict_manager['f3']

    eA = e1 + e2
    fA = f1 + f2
    e = e3 + e4
    f = f3 + f4

    tx2 = c2 + b2 * e + a2 * f
    ty2 = c2 + b2 * eA + a2 * fA

    dict_manager.update({'ty2': ty2})

    event5.set()
    event6.wait()
    event3.clear()

    ty1 = dict_manager['ty1']
    event5.clear()

    ty = ty1 + ty2
    res1 = tx2 / ty

    return res1


def generate_sigmoid_share(input, a1, b1, c1, c1_res, c1_mul, t1):
    c1_mul_ = torch.ones_like(input) * c1_mul
    c1_res_ = torch.ones_like(input) * c1_res
    t1_ = torch.ones_like(input) * t1
    a1_ = torch.ones_like(input) * a1
    b1_ = torch.ones_like(input) * b1
    c1_ = torch.ones_like(input) * c1

    return c1_mul_, c1_res_, t1_, a1_, b1_, c1_


def sigmoidServer1(input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul,
                   t1, event5, event6):
    (c1_mul_, c1_res_, t1_, a1_, b1_, c1_) = generate_sigmoid_share(input, a1, b1, c1, c1_res, c1_mul, t1)
    sig = serverSigmoidAll1(input.data, c1_mul_, c1_res_, t1_, a1_, b1_, c1_, dict_manager, event1, event2, event3,
                            event4, event5, event6)
    return sig


def sigmoidServer2(input, dict_manager, event1, event2, event3, event4, a2, b2, c2, c2_res, c2_mul,
                   t2, event5, event6):
    (c2_mul_, c2_res_, t2_, a2_, b2_, c2_) = generate_sigmoid_share(input, a2, b2, c2, c2_res, c2_mul, t2)
    sig = serverSigmoidAll2(input.data, c2_mul_, c2_res_, t2_, a2_, b2_, c2_, dict_manager, event1, event2, event3,
                            event4, event5, event6)
    return sig


def random_c():
    c = random.uniform(0, 1)
    while (c - 0 < 1e-32):
        c = random.uniform(0, 1)
    return c


def getModel():
    resNet50 = MyResNet50.ResNet50()
    resNet50.load_state_dict(
        torch.load(r'D:\project\SDNet\SDNet-backend\src\pycodes\model\deepfakeepoch13.pkl', map_location='cpu'))
    return resNet50


def generate_share(Conv, a, b, c):
    # 生成与input形状相同、元素全为1的张量
    A = torch.ones_like(Conv) * a
    B = torch.ones_like(Conv) * b
    C = torch.ones_like(Conv) * c

    V = torch.ones_like(Conv) * random.uniform(0, 1)
    Alpha1 = Conv - A
    Beta1 = V - B

    return A.data, B.data, C.data, V.data, Alpha1.data, Beta1.data


# 密文下的ReLU函数实现
def reluOnCiph(F_enc, F):
    one = torch.ones_like(F_enc)
    # 生成与input形状相同、元素全为0的张量
    zero = torch.zeros_like(F_enc)
    F_label = torch.where(F_enc > 0, one, zero)
    return F.mul(F_label)


def reluForServer1(input, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId):
    (A1, B1, C1, _, Alpha1, Beta1) = generate_share(input, a1, b1, c1)

    dict_manager.update({'Alpha1': Alpha1, 'Beta1': Beta1})
    event2.set()
    event1.wait()

    F1 = C1 + B1.mul(Alpha1 + dict_manager['Alpha2']) + A1.mul(Beta1 + dict_manager['Beta2'])
    dict_manager.update({'F1': F1})

    rdb = Redis(connection_pool=pool)
    rdb.rpush(detectionId, '> S1:运行ReLU协议中的SecMul()')
    # print(rdb.lrange(1, 0, -1))

    event4.set()
    event3.wait()
    event2.clear()

    F_enc = F1 + dict_manager['F2']
    event4.clear()

    rdb.rpush(detectionId, '> S1:获取边缘服务器S2运算结果，进行恢复')
    # print(rdb.lrange(1, 0, -1))

    return reluOnCiph(F_enc, input)


def reluForServer2(input, dict_manager, event1, event2, event3, event4, a2, b2, c2, detectionId):
    (A2, B2, C2, _, Alpha2, Beta2) = generate_share(input, a2, b2, c2)

    dict_manager.update({'Alpha2': Alpha2, 'Beta2': Beta2})

    event1.set()
    event2.wait()

    F2 = C2 + B2.mul(dict_manager['Alpha1'] + Alpha2) + A2.mul(dict_manager['Beta1'] + Beta2) \
         + (dict_manager['Alpha1'] + Alpha2).mul(dict_manager['Beta1'] + Beta2)
    dict_manager.update({'F2': F2})

    rdb = Redis(connection_pool=pool)
    rdb.rpush(detectionId, '> S2:运行ReLU协议中的SecMul()')
    # print(rdb.lrange(1, 0, -1))

    event3.set()
    event4.wait()
    event1.clear()

    F_enc = dict_manager['F1'] + F2
    event3.clear()

    rdb.rpush(detectionId, '> S2:获取边缘服务器S1运算结果，进行恢复')
    # print(rdb.lrange(1, 0, -1))

    return reluOnCiph(F_enc, input)


def divideOnEnc(Time, a, b, c):  # compute Pool/(Time1+ Time2)
    V = torch.ones_like(Time) * random.uniform(0, 1)
    A = torch.ones_like(Time) * a
    B = torch.ones_like(Time) * b
    C = torch.ones_like(Time) * c

    Alpha1 = Time - A
    Beta1 = V - B

    return A, B, C, V, Alpha1, Beta1


def maxPoolForServer1(input, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId):
    maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1, dilation=1, ceil_mode=False)
    (A1, B1, C1, V1, Alpha1, Beta1) = generate_share(input, a1, b1, c1)
    dict_manager.update({'Alpha1': Alpha1, 'Beta1': Beta1})

    event2.set()
    event1.wait()

    F1 = C1 + B1.mul(Alpha1 + dict_manager['Alpha2']) + A1.mul(Beta1 + dict_manager['Beta2'])
    dict_manager.update({'F1': F1})

    rdb = Redis(connection_pool=pool)
    rdb.rpush(detectionId, '> S1:运行SecMaxPooling中的第一次SecMul()')
    # print(rdb.lrange(1, 0, -1))

    event2.clear()
    event4.set()
    event3.wait()

    F_enc = F1 + dict_manager['F2']

    pool_enc = maxpool(F_enc)
    Times1 = maxpool(V1)

    (A1, B1, C1, V1, Alpha1, Beta1) = divideOnEnc(Times1, a1, b1, c1)
    dict_manager.update({'Alpha1': Alpha1, 'Beta1': Beta1})
    event4.clear()
    event2.set()
    event1.wait()

    rdb.rpush(detectionId, '> S1:运行SecMaxPooling中的第二次SecMul()')
    # print(rdb.lrange(1, 0, -1))

    F1 = C1 + B1.mul(Alpha1 + dict_manager['Alpha2']) + A1.mul(Beta1 + dict_manager['Beta2'])

    dict_manager.update({'F1': F1})
    event2.clear()
    event4.set()
    event3.wait()

    Times = F1 + dict_manager['F2']
    event4.clear()

    rdb.rpush(detectionId, '> S1: 在SecMaxPooling中获取S2计算结果，完成最大池化运算')
    # print(rdb.lrange(1, 0, -1))

    return (pool_enc * V1).div(Times)


def maxPoolForServer2(input, dict_manager, event1, event2, event3, event4, a2, b2, c2, detectionId):
    maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1, dilation=1, ceil_mode=False)
    (A2, B2, C2, V2, Alpha2, Beta2) = generate_share(input, a2, b2, c2)
    dict_manager.update({'Alpha2': Alpha2, 'Beta2': Beta2})

    event1.set()
    event2.wait()

    F2 = C2 + B2.mul(dict_manager['Alpha1'] + Alpha2) + A2.mul(dict_manager['Beta1'] + Beta2) + \
         (dict_manager['Alpha1'] + Alpha2) * (dict_manager['Beta1'] + Beta2)
    dict_manager.update({'F2': F2})

    rdb = Redis(connection_pool=pool)
    rdb.rpush(detectionId, '> S2:运行SecMaxPooling中的第一次SecMul()')
    # print(rdb.lrange(1, 0, -1))

    event1.clear()
    event3.set()
    event4.wait()

    F_enc = dict_manager['F1'] + F2

    pool_enc = maxpool(F_enc)
    Times2 = maxpool(V2)

    (A2, B2, C2, V2, Alpha2, Beta2) = divideOnEnc(Times2, a2, b2, c2)
    dict_manager.update({'Alpha2': Alpha2, 'Beta2': Beta2})
    event3.clear()
    event1.set()
    event2.wait()

    F2 = C2 + B2.mul(dict_manager['Alpha1'] + Alpha2) + A2.mul(dict_manager['Beta1'] + Beta2) + \
         (dict_manager['Alpha1'] + Alpha2) * (dict_manager['Beta1'] + Beta2)

    rdb.rpush(detectionId, '> S2:运行SecMaxPooling中的第二次SecMul()')
    # print(rdb.lrange(1, 0, -1))

    dict_manager.update({'F2': F2})
    event1.clear()
    event3.set()
    event4.wait()

    Times = dict_manager['F1'] + F2
    event3.clear()

    rdb.rpush(detectionId, '> S2: 在SecMaxPooling中获取S1计算结果，完成最大池化运算')
    # print(rdb.lrange(1, 0, -1))

    return (pool_enc * V2).div(Times)


def server1_ResNet50(event1, event2, event3, event4, image_1, a1, b1, c1,
                     dict_manager,
                     conv0_1, bn0_1, conv1_1_1, bn1_1_1, conv1_1_2,
                     bn1_1_2, conv1_1_3, bn1_1_3, conv1_1_4, bn1_1_4, conv1_2_1,
                     bn1_2_1,
                     conv1_2_2, bn1_2_2, conv1_2_3, bn1_2_3, conv1_3_1, bn1_3_1,
                     conv1_3_2, bn1_3_2, conv1_3_3, bn1_3_3,
                     conv2_1_1, bn2_1_1, conv2_1_2, bn2_1_2, conv2_1_3, bn2_1_3,
                     conv2_1_4, bn2_1_4, conv2_2_1, bn2_2_1, conv2_2_2, bn2_2_2,
                     conv2_2_3, bn2_2_3,
                     conv2_3_1, bn2_3_1, conv2_3_2, bn2_3_2, conv2_3_3, bn2_3_3,
                     conv2_4_1, bn2_4_1, conv2_4_2, bn2_4_2, conv2_4_3, bn2_4_3,
                     conv3_1_1, bn3_1_1, conv3_1_2, bn3_1_2, conv3_1_3, bn3_1_3,
                     conv3_1_4, bn3_1_4, conv3_2_1, bn3_2_1, conv3_2_2, bn3_2_2,
                     conv3_2_3, bn3_2_3,
                     conv3_3_1, bn3_3_1, conv3_3_2, bn3_3_2, conv3_3_3, bn3_3_3,
                     conv3_4_1, bn3_4_1, conv3_4_2, bn3_4_2, conv3_4_3, bn3_4_3,
                     conv3_5_1, bn3_5_1, conv3_5_2, bn3_5_2, conv3_5_3, bn3_5_3,
                     conv3_6_1, bn3_6_1, conv3_6_2, bn3_6_2, conv3_6_3, bn3_6_3,
                     conv4_1_1, bn4_1_1, conv4_1_2, bn4_1_2, conv4_1_3, bn4_1_3,
                     conv4_1_4, bn4_1_4, conv4_2_1, bn4_2_1, conv4_2_2, bn4_2_2,
                     conv4_2_3, bn4_2_3,
                     conv4_3_1, bn4_3_1, conv4_3_2, bn4_3_2, conv4_3_3, bn4_3_3, fc,
                     q, event5, event6, c1_mul, c1_res, t1, detectionId):
    event2.clear()
    event4.clear()

    rdb = Redis(connection_pool=pool)

    image_1 = image_1.clone().detach()
    image_1 = image_1.to(torch.float32)

    # conv1
    conv0_1a = conv0_1(image_1)
    # bn1
    conv0_1a = bn0_1(conv0_1a)

    # relu
    conv0_1a = reluForServer1(conv0_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    # maxpool
    start = datetime.now()
    conv0_1a = maxPoolForServer1(conv0_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    end = datetime.now()
    # print('maxpool time', end - start)
    # layer1
    # bottleneck0
    conv1_1_1a = conv1_1_1(conv0_1a)
    conv1_1_1a = bn1_1_1(conv1_1_1a)
    conv1_1_1a = reluForServer1(conv1_1_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_1_2a = conv1_1_2(conv1_1_1a)
    conv1_1_2a = bn1_1_2(conv1_1_2a)
    conv1_1_2a = reluForServer1(conv1_1_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_1_3a = conv1_1_3(conv1_1_2a)
    conv1_1_3a = bn1_1_3(conv1_1_3a)
    # residual
    residual = conv1_1_4(conv0_1a)
    residual = bn1_1_4(residual)
    conv1_1_3a = conv1_1_3a + residual
    conv1_1_3a = reluForServer1(conv1_1_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck1
    conv1_2_1a = conv1_2_1(conv1_1_3a)
    conv1_2_1a = bn1_2_1(conv1_2_1a)
    conv1_2_1a = reluForServer1(conv1_2_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_2_2a = conv1_2_2(conv1_2_1a)
    conv1_2_2a = bn1_2_2(conv1_2_2a)
    conv1_2_2a = reluForServer1(conv1_2_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_2_3a = conv1_2_3(conv1_2_2a)
    conv1_2_3a = bn1_2_3(conv1_2_3a)
    conv1_2_3a = conv1_2_3a + conv1_1_3a
    conv1_2_3a = reluForServer1(conv1_2_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck2
    conv1_3_1a = conv1_3_1(conv1_2_3a)
    conv1_3_1a = bn1_3_1(conv1_3_1a)
    conv1_3_1a = reluForServer1(conv1_3_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_3_2a = conv1_3_2(conv1_3_1a)
    conv1_3_2a = bn1_3_2(conv1_3_2a)
    conv1_3_2a = reluForServer1(conv1_3_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_3_3a = conv1_3_3(conv1_3_2a)
    conv1_3_3a = bn1_3_3(conv1_3_3a)
    conv1_3_3a = conv1_3_3a + conv1_2_3a
    conv1_3_3a = reluForServer1(conv1_3_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # layer2
    # bottleneck0
    conv2_1_1a = conv2_1_1(conv1_3_3a)
    conv2_1_1a = bn2_1_1(conv2_1_1a)
    conv2_1_1a = reluForServer1(conv2_1_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_1_2a = conv2_1_2(conv2_1_1a)
    conv2_1_2a = bn2_1_2(conv2_1_2a)
    conv2_1_2a = reluForServer1(conv2_1_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_1_3a = conv2_1_3(conv2_1_2a)
    conv2_1_3a = bn2_1_3(conv2_1_3a)
    residual = conv2_1_4(conv1_3_3a)
    residual = bn2_1_4(residual)
    conv2_1_3a = conv2_1_3a + residual
    conv2_1_3a = reluForServer1(conv2_1_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck1
    conv2_2_1a = conv2_2_1(conv2_1_3a)
    conv2_2_1a = bn2_2_1(conv2_2_1a)
    conv2_2_1a = reluForServer1(conv2_2_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_2_2a = conv2_2_2(conv2_2_1a)
    conv2_2_2a = bn2_2_2(conv2_2_2a)
    conv2_2_2a = reluForServer1(conv2_2_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_2_3a = conv2_2_3(conv2_2_2a)
    conv2_2_3a = bn2_2_3(conv2_2_3a)
    conv2_2_3a = conv2_2_3a + conv2_1_3a
    conv2_2_3a = reluForServer1(conv2_2_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck2
    conv2_3_1a = conv2_3_1(conv2_2_3a)
    conv2_3_1a = bn2_3_1(conv2_3_1a)
    conv2_3_1a = reluForServer1(conv2_3_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_3_2a = conv2_3_2(conv2_3_1a)
    conv2_3_2a = bn2_3_2(conv2_3_2a)
    conv2_3_2a = reluForServer1(conv2_3_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_3_3a = conv2_3_3(conv2_3_2a)
    conv2_3_3a = bn2_3_3(conv2_3_3a)
    conv2_3_3a = conv2_3_3a + conv2_2_3a
    conv2_3_3a = reluForServer1(conv2_3_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck3
    conv2_4_1a = conv2_4_1(conv2_3_3a)
    conv2_4_1a = bn2_4_1(conv2_4_1a)
    conv2_4_1a = reluForServer1(conv2_4_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_4_2a = conv2_4_2(conv2_4_1a)
    conv2_4_2a = bn2_4_2(conv2_4_2a)
    conv2_4_2a = reluForServer1(conv2_4_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_4_3a = conv2_4_3(conv2_4_2a)
    conv2_4_3a = bn2_4_3(conv2_4_3a)
    conv2_4_3a = conv2_4_3a + conv2_3_3a
    conv2_4_3a = reluForServer1(conv2_4_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # layer3
    # bottleneck 0
    conv3_1_1a = conv3_1_1(conv2_4_3a)
    conv3_1_1a = bn3_1_1(conv3_1_1a)
    conv3_1_1a = reluForServer1(conv3_1_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_1_2a = conv3_1_2(conv3_1_1a)
    conv3_1_2a = bn3_1_2(conv3_1_2a)
    conv3_1_2a = reluForServer1(conv3_1_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_1_3a = conv3_1_3(conv3_1_2a)
    conv3_1_3a = bn3_1_3(conv3_1_3a)
    residual = conv3_1_4(conv2_4_3a)
    residual = bn3_1_4(residual)
    conv3_1_3a = residual + conv3_1_3a
    conv3_1_3a = reluForServer1(conv3_1_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck 1
    conv3_2_1a = conv3_2_1(conv3_1_3a)
    conv3_2_1a = bn3_2_1(conv3_2_1a)
    conv3_2_1a = reluForServer1(conv3_2_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_2_2a = conv3_2_2(conv3_2_1a)
    conv3_2_2a = bn3_2_2(conv3_2_2a)
    conv3_2_2a = reluForServer1(conv3_2_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_2_3a = conv3_2_3(conv3_2_2a)
    conv3_2_3a = bn3_2_3(conv3_2_3a)
    conv3_2_3a = conv3_2_3a + conv3_1_3a
    conv3_2_3a = reluForServer1(conv3_2_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck2
    conv3_3_1a = conv3_3_1(conv3_2_3a)
    conv3_3_1a = bn3_3_1(conv3_3_1a)
    conv3_3_1a = reluForServer1(conv3_3_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_3_2a = conv3_3_2(conv3_3_1a)
    conv3_3_2a = bn3_3_2(conv3_3_2a)
    conv3_3_2a = reluForServer1(conv3_3_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_3_3a = conv3_3_3(conv3_3_2a)
    conv3_3_3a = bn3_3_3(conv3_3_3a)
    conv3_3_3a = conv3_3_3a + conv3_2_3a
    conv3_3_3a = reluForServer1(conv3_3_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck3
    conv3_4_1a = conv3_4_1(conv3_3_3a)
    conv3_4_1a = bn3_4_1(conv3_4_1a)
    conv3_4_1a = reluForServer1(conv3_4_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_4_2a = conv3_4_2(conv3_4_1a)
    conv3_4_2a = bn3_4_2(conv3_4_2a)
    conv3_4_2a = reluForServer1(conv3_4_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_4_3a = conv3_4_3(conv3_4_2a)
    conv3_4_3a = bn3_4_3(conv3_4_3a)
    conv3_4_3a = conv3_4_3a + conv3_3_3a
    conv3_4_3a = reluForServer1(conv3_4_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck4
    conv3_5_1a = conv3_5_1(conv3_4_3a)
    conv3_5_1a = bn3_5_1(conv3_5_1a)
    conv3_5_1a = reluForServer1(conv3_5_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_5_2a = conv3_5_2(conv3_5_1a)
    conv3_5_2a = bn3_5_2(conv3_5_2a)
    conv3_5_2a = reluForServer1(conv3_5_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_5_3a = conv3_5_3(conv3_5_2a)
    conv3_5_3a = bn3_5_3(conv3_5_3a)
    conv3_5_3a = conv3_5_3a + conv3_4_3a
    conv3_5_3a = reluForServer1(conv3_5_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck5
    conv3_6_1a = conv3_6_1(conv3_5_3a)
    conv3_6_1a = bn3_6_1(conv3_6_1a)
    conv3_6_1a = reluForServer1(conv3_6_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_6_2a = conv3_6_2(conv3_6_1a)
    conv3_6_2a = bn3_6_2(conv3_6_2a)
    conv3_6_2a = reluForServer1(conv3_6_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_6_3a = conv3_6_3(conv3_6_2a)
    conv3_6_3a = bn3_6_3(conv3_6_3a)
    conv3_6_3a = conv3_6_3a + conv3_5_3a
    conv3_6_3a = reluForServer1(conv3_6_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # layer4
    # bottleneck0
    conv4_1_1a = conv4_1_1(conv3_6_3a)
    conv4_1_1a = bn4_1_1(conv4_1_1a)
    conv4_1_1a = reluForServer1(conv4_1_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_1_2a = conv4_1_2(conv4_1_1a)
    conv4_1_2a = bn4_1_2(conv4_1_2a)
    conv4_1_2a = reluForServer1(conv4_1_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_1_3a = conv4_1_3(conv4_1_2a)
    conv4_1_3a = bn4_1_3(conv4_1_3a)
    residual = conv4_1_4(conv3_6_3a)
    residual = bn4_1_4(residual)
    conv4_1_3a = conv4_1_3a + residual
    conv4_1_3a = reluForServer1(conv4_1_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck1
    conv4_2_1a = conv4_2_1(conv4_1_3a)
    conv4_2_1a = bn4_2_1(conv4_2_1a)
    conv4_2_1a = reluForServer1(conv4_2_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_2_2a = conv4_2_2(conv4_2_1a)
    conv4_2_2a = bn4_2_2(conv4_2_2a)
    conv4_2_2a = reluForServer1(conv4_2_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_2_3a = conv4_2_3(conv4_2_2a)
    conv4_2_3a = bn4_2_3(conv4_2_3a)
    conv4_2_3a = conv4_2_3a + conv4_1_3a
    conv4_2_3a = reluForServer1(conv4_2_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck2
    conv4_3_1a = conv4_3_1(conv4_2_3a)
    conv4_3_1a = bn4_3_1(conv4_3_1a)
    conv4_3_1a = reluForServer1(conv4_3_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_3_2a = conv4_3_2(conv4_3_1a)
    conv4_3_2a = bn4_3_2(conv4_3_2a)
    conv4_3_2a = reluForServer1(conv4_3_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_3_3a = conv4_3_3(conv4_3_2a)
    conv4_3_3a = bn4_3_3(conv4_3_3a)
    conv4_3_3a = conv4_3_3a + conv4_2_3a
    conv4_3_3a = reluForServer1(conv4_3_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    result = avg(conv4_3_3a)
    res = result.view(-1, 2048)
    start = datetime.now()
    res1 = fc(res)
    res1 = sigmoidServer1(res1, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul,
                          t1, event5, event6)
    end = datetime.now()
    # print('fc time', end - start)

    result1 = res1.detach().numpy()

    result1 = result1[0, 0:2]
    result1 = result1.reshape(1, 2)

    q.put(result1)


def server2_ResNet50(event1, event2, event3, event4, image_, a1, b1, c1,
                     dict_manager,
                     conv0_1, bn0_1, conv1_1_1, bn1_1_1, conv1_1_2,
                     bn1_1_2, conv1_1_3, bn1_1_3, conv1_1_4, bn1_1_4, conv1_2_1,
                     bn1_2_1,
                     conv1_2_2, bn1_2_2, conv1_2_3, bn1_2_3, conv1_3_1, bn1_3_1,
                     conv1_3_2, bn1_3_2, conv1_3_3, bn1_3_3,
                     conv2_1_1, bn2_1_1, conv2_1_2, bn2_1_2, conv2_1_3, bn2_1_3,
                     conv2_1_4, bn2_1_4, conv2_2_1, bn2_2_1, conv2_2_2, bn2_2_2,
                     conv2_2_3, bn2_2_3,
                     conv2_3_1, bn2_3_1, conv2_3_2, bn2_3_2, conv2_3_3, bn2_3_3,
                     conv2_4_1, bn2_4_1, conv2_4_2, bn2_4_2, conv2_4_3, bn2_4_3,
                     conv3_1_1, bn3_1_1, conv3_1_2, bn3_1_2, conv3_1_3, bn3_1_3,
                     conv3_1_4, bn3_1_4, conv3_2_1, bn3_2_1, conv3_2_2, bn3_2_2,
                     conv3_2_3, bn3_2_3,
                     conv3_3_1, bn3_3_1, conv3_3_2, bn3_3_2, conv3_3_3, bn3_3_3,
                     conv3_4_1, bn3_4_1, conv3_4_2, bn3_4_2, conv3_4_3, bn3_4_3,
                     conv3_5_1, bn3_5_1, conv3_5_2, bn3_5_2, conv3_5_3, bn3_5_3,
                     conv3_6_1, bn3_6_1, conv3_6_2, bn3_6_2, conv3_6_3, bn3_6_3,
                     conv4_1_1, bn4_1_1, conv4_1_2, bn4_1_2, conv4_1_3, bn4_1_3,
                     conv4_1_4, bn4_1_4, conv4_2_1, bn4_2_1, conv4_2_2, bn4_2_2,
                     conv4_2_3, bn4_2_3,
                     conv4_3_1, bn4_3_1, conv4_3_2, bn4_3_2, conv4_3_3, bn4_3_3, fc,
                     q, event5, event6, c1_mul, c1_res, t1, detectionId):
    event1.clear()
    event3.clear()

    # image_ = torch.tensor(image_, dtype=torch.float32)
    image_ = image_.clone().detach()
    image_ = image_.to(torch.float32)

    rdb = Redis(connection_pool=pool)
    # conv1
    conv0_1a = conv0_1(image_)
    # bn1
    conv0_1a = bn0_1(conv0_1a)
    # relu
    conv0_1a = reluForServer2(conv0_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    # maxpool
    conv0_1a = maxPoolForServer2(conv0_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # layer1
    # bottleneck0
    conv1_1_1a = conv1_1_1(conv0_1a)
    conv1_1_1a = bn1_1_1(conv1_1_1a)
    conv1_1_1a = reluForServer2(conv1_1_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_1_2a = conv1_1_2(conv1_1_1a)
    conv1_1_2a = bn1_1_2(conv1_1_2a)
    conv1_1_2a = reluForServer2(conv1_1_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_1_3a = conv1_1_3(conv1_1_2a)
    conv1_1_3a = bn1_1_3(conv1_1_3a)
    # residual
    residual = conv1_1_4(conv0_1a)
    residual = bn1_1_4(residual)
    conv1_1_3a = conv1_1_3a + residual
    conv1_1_3a = reluForServer2(conv1_1_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck1
    conv1_2_1a = conv1_2_1(conv1_1_3a)
    conv1_2_1a = bn1_2_1(conv1_2_1a)
    conv1_2_1a = reluForServer2(conv1_2_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_2_2a = conv1_2_2(conv1_2_1a)
    conv1_2_2a = bn1_2_2(conv1_2_2a)
    conv1_2_2a = reluForServer2(conv1_2_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_2_3a = conv1_2_3(conv1_2_2a)
    conv1_2_3a = bn1_2_3(conv1_2_3a)
    conv1_2_3a = conv1_2_3a + conv1_1_3a
    conv1_2_3a = reluForServer2(conv1_2_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck2
    conv1_3_1a = conv1_3_1(conv1_2_3a)
    conv1_3_1a = bn1_3_1(conv1_3_1a)
    conv1_3_1a = reluForServer2(conv1_3_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_3_2a = conv1_3_2(conv1_3_1a)
    conv1_3_2a = bn1_3_2(conv1_3_2a)
    conv1_3_2a = reluForServer2(conv1_3_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv1_3_3a = conv1_3_3(conv1_3_2a)
    conv1_3_3a = bn1_3_3(conv1_3_3a)
    conv1_3_3a = conv1_3_3a + conv1_2_3a
    conv1_3_3a = reluForServer2(conv1_3_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # layer2
    # bottleneck0
    conv2_1_1a = conv2_1_1(conv1_3_3a)
    conv2_1_1a = bn2_1_1(conv2_1_1a)
    conv2_1_1a = reluForServer2(conv2_1_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_1_2a = conv2_1_2(conv2_1_1a)
    conv2_1_2a = bn2_1_2(conv2_1_2a)
    conv2_1_2a = reluForServer2(conv2_1_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_1_3a = conv2_1_3(conv2_1_2a)
    conv2_1_3a = bn2_1_3(conv2_1_3a)
    residual = conv2_1_4(conv1_3_3a)
    residual = bn2_1_4(residual)
    conv2_1_3a = conv2_1_3a + residual
    conv2_1_3a = reluForServer2(conv2_1_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck1
    conv2_2_1a = conv2_2_1(conv2_1_3a)
    conv2_2_1a = bn2_2_1(conv2_2_1a)
    conv2_2_1a = reluForServer2(conv2_2_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_2_2a = conv2_2_2(conv2_2_1a)
    conv2_2_2a = bn2_2_2(conv2_2_2a)
    conv2_2_2a = reluForServer2(conv2_2_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_2_3a = conv2_2_3(conv2_2_2a)
    conv2_2_3a = bn2_2_3(conv2_2_3a)
    conv2_2_3a = conv2_2_3a + conv2_1_3a
    conv2_2_3a = reluForServer2(conv2_2_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck2
    conv2_3_1a = conv2_3_1(conv2_2_3a)
    conv2_3_1a = bn2_3_1(conv2_3_1a)
    conv2_3_1a = reluForServer2(conv2_3_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_3_2a = conv2_3_2(conv2_3_1a)
    conv2_3_2a = bn2_3_2(conv2_3_2a)
    conv2_3_2a = reluForServer2(conv2_3_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_3_3a = conv2_3_3(conv2_3_2a)
    conv2_3_3a = bn2_3_3(conv2_3_3a)
    conv2_3_3a = conv2_3_3a + conv2_2_3a
    conv2_3_3a = reluForServer2(conv2_3_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck3
    conv2_4_1a = conv2_4_1(conv2_3_3a)
    conv2_4_1a = bn2_4_1(conv2_4_1a)
    conv2_4_1a = reluForServer2(conv2_4_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_4_2a = conv2_4_2(conv2_4_1a)
    conv2_4_2a = bn2_4_2(conv2_4_2a)
    conv2_4_2a = reluForServer2(conv2_4_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv2_4_3a = conv2_4_3(conv2_4_2a)
    conv2_4_3a = bn2_4_3(conv2_4_3a)
    conv2_4_3a = conv2_4_3a + conv2_3_3a
    conv2_4_3a = reluForServer2(conv2_4_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # layer3
    # bottleneck 0
    conv3_1_1a = conv3_1_1(conv2_4_3a)
    conv3_1_1a = bn3_1_1(conv3_1_1a)
    conv3_1_1a = reluForServer2(conv3_1_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_1_2a = conv3_1_2(conv3_1_1a)
    conv3_1_2a = bn3_1_2(conv3_1_2a)
    conv3_1_2a = reluForServer2(conv3_1_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_1_3a = conv3_1_3(conv3_1_2a)
    conv3_1_3a = bn3_1_3(conv3_1_3a)
    residual = conv3_1_4(conv2_4_3a)
    residual = bn3_1_4(residual)
    conv3_1_3a = residual + conv3_1_3a
    conv3_1_3a = reluForServer2(conv3_1_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck 1
    conv3_2_1a = conv3_2_1(conv3_1_3a)
    conv3_2_1a = bn3_2_1(conv3_2_1a)
    conv3_2_1a = reluForServer2(conv3_2_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_2_2a = conv3_2_2(conv3_2_1a)
    conv3_2_2a = bn3_2_2(conv3_2_2a)
    conv3_2_2a = reluForServer2(conv3_2_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_2_3a = conv3_2_3(conv3_2_2a)
    conv3_2_3a = bn3_2_3(conv3_2_3a)
    conv3_2_3a = conv3_2_3a + conv3_1_3a
    conv3_2_3a = reluForServer2(conv3_2_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck2
    conv3_3_1a = conv3_3_1(conv3_2_3a)
    conv3_3_1a = bn3_3_1(conv3_3_1a)
    conv3_3_1a = reluForServer2(conv3_3_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_3_2a = conv3_3_2(conv3_3_1a)
    conv3_3_2a = bn3_3_2(conv3_3_2a)
    conv3_3_2a = reluForServer2(conv3_3_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_3_3a = conv3_3_3(conv3_3_2a)
    conv3_3_3a = bn3_3_3(conv3_3_3a)
    conv3_3_3a = conv3_3_3a + conv3_2_3a
    conv3_3_3a = reluForServer2(conv3_3_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck3
    conv3_4_1a = conv3_4_1(conv3_3_3a)
    conv3_4_1a = bn3_4_1(conv3_4_1a)
    conv3_4_1a = reluForServer2(conv3_4_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_4_2a = conv3_4_2(conv3_4_1a)
    conv3_4_2a = bn3_4_2(conv3_4_2a)
    conv3_4_2a = reluForServer2(conv3_4_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_4_3a = conv3_4_3(conv3_4_2a)
    conv3_4_3a = bn3_4_3(conv3_4_3a)
    conv3_4_3a = conv3_4_3a + conv3_3_3a
    conv3_4_3a = reluForServer2(conv3_4_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck4
    conv3_5_1a = conv3_5_1(conv3_4_3a)
    conv3_5_1a = bn3_5_1(conv3_5_1a)
    conv3_5_1a = reluForServer2(conv3_5_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_5_2a = conv3_5_2(conv3_5_1a)
    conv3_5_2a = bn3_5_2(conv3_5_2a)
    conv3_5_2a = reluForServer2(conv3_5_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_5_3a = conv3_5_3(conv3_5_2a)
    conv3_5_3a = bn3_5_3(conv3_5_3a)
    conv3_5_3a = conv3_5_3a + conv3_4_3a
    conv3_5_3a = reluForServer2(conv3_5_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck5
    conv3_6_1a = conv3_6_1(conv3_5_3a)
    conv3_6_1a = bn3_6_1(conv3_6_1a)
    conv3_6_1a = reluForServer2(conv3_6_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_6_2a = conv3_6_2(conv3_6_1a)
    conv3_6_2a = bn3_6_2(conv3_6_2a)
    conv3_6_2a = reluForServer2(conv3_6_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv3_6_3a = conv3_6_3(conv3_6_2a)
    conv3_6_3a = bn3_6_3(conv3_6_3a)
    conv3_6_3a = conv3_6_3a + conv3_5_3a
    conv3_6_3a = reluForServer2(conv3_6_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # layer4
    # bottleneck0
    conv4_1_1a = conv4_1_1(conv3_6_3a)
    conv4_1_1a = bn4_1_1(conv4_1_1a)
    conv4_1_1a = reluForServer2(conv4_1_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_1_2a = conv4_1_2(conv4_1_1a)
    conv4_1_2a = bn4_1_2(conv4_1_2a)
    conv4_1_2a = reluForServer2(conv4_1_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_1_3a = conv4_1_3(conv4_1_2a)
    conv4_1_3a = bn4_1_3(conv4_1_3a)
    residual = conv4_1_4(conv3_6_3a)
    residual = bn4_1_4(residual)
    conv4_1_3a = conv4_1_3a + residual
    conv4_1_3a = reluForServer2(conv4_1_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck1
    conv4_2_1a = conv4_2_1(conv4_1_3a)
    conv4_2_1a = bn4_2_1(conv4_2_1a)
    conv4_2_1a = reluForServer2(conv4_2_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_2_2a = conv4_2_2(conv4_2_1a)
    conv4_2_2a = bn4_2_2(conv4_2_2a)
    conv4_2_2a = reluForServer2(conv4_2_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_2_3a = conv4_2_3(conv4_2_2a)
    conv4_2_3a = bn4_2_3(conv4_2_3a)
    conv4_2_3a = conv4_2_3a + conv4_1_3a
    conv4_2_3a = reluForServer2(conv4_2_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    # bottleneck2
    conv4_3_1a = conv4_3_1(conv4_2_3a)
    conv4_3_1a = bn4_3_1(conv4_3_1a)
    conv4_3_1a = reluForServer2(conv4_3_1a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_3_2a = conv4_3_2(conv4_3_1a)
    conv4_3_2a = bn4_3_2(conv4_3_2a)
    conv4_3_2a = reluForServer2(conv4_3_2a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)
    conv4_3_3a = conv4_3_3(conv4_3_2a)
    conv4_3_3a = bn4_3_3(conv4_3_3a)
    conv4_3_3a = conv4_3_3a + conv4_2_3a
    conv4_3_3a = reluForServer2(conv4_3_3a, dict_manager, event1, event2, event3, event4, a1, b1, c1, detectionId)

    avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    result = avg(conv4_3_3a)
    res = result.view(-1, 2048)

    res2 = fc(res)
    res2 = sigmoidServer2(res2, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul,
                          t1, event5, event6)

    result2 = res2.detach().numpy()
    result2 = result2[0, 0:2]
    result2 = result2.reshape(1, 2)

    q.put(result2)


def SecResNet50(image_1, image_2, conv0_1, bn0_1, conv1_1_1, bn1_1_1, conv1_1_2, bn1_1_2, conv1_1_3, bn1_1_3, conv1_1_4,
                bn1_1_4, conv1_2_1, bn1_2_1,
                conv1_2_2, bn1_2_2, conv1_2_3, bn1_2_3, conv1_3_1, bn1_3_1, conv1_3_2, bn1_3_2, conv1_3_3, bn1_3_3,
                conv2_1_1, bn2_1_1, conv2_1_2, bn2_1_2, conv2_1_3, bn2_1_3, conv2_1_4, bn2_1_4, conv2_2_1, bn2_2_1,
                conv2_2_2, bn2_2_2, conv2_2_3, bn2_2_3,
                conv2_3_1, bn2_3_1, conv2_3_2, bn2_3_2, conv2_3_3, bn2_3_3, conv2_4_1, bn2_4_1, conv2_4_2, bn2_4_2,
                conv2_4_3, bn2_4_3,
                conv3_1_1, bn3_1_1, conv3_1_2, bn3_1_2, conv3_1_3, bn3_1_3, conv3_1_4, bn3_1_4, conv3_2_1, bn3_2_1,
                conv3_2_2, bn3_2_2, conv3_2_3, bn3_2_3,
                conv3_3_1, bn3_3_1, conv3_3_2, bn3_3_2, conv3_3_3, bn3_3_3, conv3_4_1, bn3_4_1, conv3_4_2, bn3_4_2,
                conv3_4_3, bn3_4_3,
                conv3_5_1, bn3_5_1, conv3_5_2, bn3_5_2, conv3_5_3, bn3_5_3, conv3_6_1, bn3_6_1, conv3_6_2, bn3_6_2,
                conv3_6_3, bn3_6_3,
                conv4_1_1, bn4_1_1, conv4_1_2, bn4_1_2, conv4_1_3, bn4_1_3, conv4_1_4, bn4_1_4, conv4_2_1, bn4_2_1,
                conv4_2_2, bn4_2_2, conv4_2_3, bn4_2_3,
                conv4_3_1, bn4_3_1, conv4_3_2, bn4_3_2, conv4_3_3, bn4_3_3, fc, detectionId):
    event1 = torch.multiprocessing.Event()
    event2 = torch.multiprocessing.Event()
    event3 = torch.multiprocessing.Event()
    event4 = torch.multiprocessing.Event()
    event5 = torch.multiprocessing.Event()
    event6 = torch.multiprocessing.Event()

    # global dict_manager
    dict_manager = Manager().dict()

    rdb = Redis(connection_pool=pool)
    rdb.rpush(detectionId, '>T: 模拟诚实的第三方服务器生成随机数')
    # print(rdb.lrange(1, 0, -1))

    # 乘法的
    a = random_c()
    b = random_c()
    a1 = random_c()
    a2 = a - a1
    b1 = random_c()
    b2 = b - b1
    c = a * b
    c1 = random_c()
    c2 = c - c1

    # 指数的
    c1_mul = random_c()
    c2_mul = random_c()
    c = c1_mul * c2_mul
    c1_res = random_c()
    c2_res = c - c1_res

    # 除法的
    t = random_c()
    t1 = random_c()
    t2 = t - t1

    rdb.rpush(detectionId, '>T: 随机数生成完毕')
    # print(rdb.lrange(1, 0, -1))

    q = torch.multiprocessing.Queue()
    jobs = []
    server1_process = torch.multiprocessing.Process(target=server1_ResNet50,
                                                    args=(
                                                        event1, event2, event3, event4, image_1, a1, b1, c1,
                                                        dict_manager,
                                                        conv0_1, bn0_1, conv1_1_1, bn1_1_1, conv1_1_2,
                                                        bn1_1_2, conv1_1_3, bn1_1_3, conv1_1_4, bn1_1_4, conv1_2_1,
                                                        bn1_2_1,
                                                        conv1_2_2, bn1_2_2, conv1_2_3, bn1_2_3, conv1_3_1, bn1_3_1,
                                                        conv1_3_2, bn1_3_2, conv1_3_3, bn1_3_3,
                                                        conv2_1_1, bn2_1_1, conv2_1_2, bn2_1_2, conv2_1_3, bn2_1_3,
                                                        conv2_1_4, bn2_1_4, conv2_2_1, bn2_2_1, conv2_2_2, bn2_2_2,
                                                        conv2_2_3, bn2_2_3,
                                                        conv2_3_1, bn2_3_1, conv2_3_2, bn2_3_2, conv2_3_3, bn2_3_3,
                                                        conv2_4_1, bn2_4_1, conv2_4_2, bn2_4_2, conv2_4_3, bn2_4_3,
                                                        conv3_1_1, bn3_1_1, conv3_1_2, bn3_1_2, conv3_1_3, bn3_1_3,
                                                        conv3_1_4, bn3_1_4, conv3_2_1, bn3_2_1, conv3_2_2, bn3_2_2,
                                                        conv3_2_3, bn3_2_3,
                                                        conv3_3_1, bn3_3_1, conv3_3_2, bn3_3_2, conv3_3_3, bn3_3_3,
                                                        conv3_4_1, bn3_4_1, conv3_4_2, bn3_4_2, conv3_4_3, bn3_4_3,
                                                        conv3_5_1, bn3_5_1, conv3_5_2, bn3_5_2, conv3_5_3, bn3_5_3,
                                                        conv3_6_1, bn3_6_1, conv3_6_2, bn3_6_2, conv3_6_3, bn3_6_3,
                                                        conv4_1_1, bn4_1_1, conv4_1_2, bn4_1_2, conv4_1_3, bn4_1_3,
                                                        conv4_1_4, bn4_1_4, conv4_2_1, bn4_2_1, conv4_2_2, bn4_2_2,
                                                        conv4_2_3, bn4_2_3,
                                                        conv4_3_1, bn4_3_1, conv4_3_2, bn4_3_2, conv4_3_3, bn4_3_3, fc,
                                                        q, event5, event6, c1_mul, c1_res, t1, detectionId))

    server2_process = torch.multiprocessing.Process(target=server2_ResNet50,
                                                    args=(
                                                        event1, event2, event3, event4, image_2, a2, b2, c2,
                                                        dict_manager,
                                                        conv0_1, bn0_1, conv1_1_1, bn1_1_1, conv1_1_2,
                                                        bn1_1_2, conv1_1_3, bn1_1_3, conv1_1_4, bn1_1_4, conv1_2_1,
                                                        bn1_2_1,
                                                        conv1_2_2, bn1_2_2, conv1_2_3, bn1_2_3, conv1_3_1, bn1_3_1,
                                                        conv1_3_2, bn1_3_2, conv1_3_3, bn1_3_3,
                                                        conv2_1_1, bn2_1_1, conv2_1_2, bn2_1_2, conv2_1_3, bn2_1_3,
                                                        conv2_1_4, bn2_1_4, conv2_2_1, bn2_2_1, conv2_2_2, bn2_2_2,
                                                        conv2_2_3, bn2_2_3,
                                                        conv2_3_1, bn2_3_1, conv2_3_2, bn2_3_2, conv2_3_3, bn2_3_3,
                                                        conv2_4_1, bn2_4_1, conv2_4_2, bn2_4_2, conv2_4_3, bn2_4_3,
                                                        conv3_1_1, bn3_1_1, conv3_1_2, bn3_1_2, conv3_1_3, bn3_1_3,
                                                        conv3_1_4, bn3_1_4, conv3_2_1, bn3_2_1, conv3_2_2, bn3_2_2,
                                                        conv3_2_3, bn3_2_3,
                                                        conv3_3_1, bn3_3_1, conv3_3_2, bn3_3_2, conv3_3_3, bn3_3_3,
                                                        conv3_4_1, bn3_4_1, conv3_4_2, bn3_4_2, conv3_4_3, bn3_4_3,
                                                        conv3_5_1, bn3_5_1, conv3_5_2, bn3_5_2, conv3_5_3, bn3_5_3,
                                                        conv3_6_1, bn3_6_1, conv3_6_2, bn3_6_2, conv3_6_3, bn3_6_3,
                                                        conv4_1_1, bn4_1_1, conv4_1_2, bn4_1_2, conv4_1_3, bn4_1_3,
                                                        conv4_1_4, bn4_1_4, conv4_2_1, bn4_2_1, conv4_2_2, bn4_2_2,
                                                        conv4_2_3, bn4_2_3,
                                                        conv4_3_1, bn4_3_1, conv4_3_2, bn4_3_2, conv4_3_3, bn4_3_3, fc,
                                                        q, event5, event6, c2_mul, c2_res, t2, detectionId))

    server1_process.start()
    jobs.append(server1_process)

    server2_process.start()
    jobs.append(server2_process)

    server1_process.join()
    server2_process.join()

    results = [q.get() for j in jobs]

    res1 = torch.tensor(results[0], dtype=torch.float32)
    res2 = torch.tensor(results[1], dtype=torch.float32)

    # print('true result:', (res1 + res2))

    return res1, res2


if __name__ == '__main__':
    net = getModel()
    net.eval()
    resNet50 = net.state_dict()

    # layer0:
    conv0_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    bn0_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn0_1.eval()

    # layer1 bottleneck0
    conv1_1_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
    bn1_1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn1_1_1.eval()
    conv1_1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
    bn1_1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn1_1_2.eval()
    conv1_1_3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
    bn1_1_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn1_1_3.eval()
    conv1_1_4 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)  # downsample
    bn1_1_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn1_1_4.eval()

    # layer1 bottleneck1
    conv1_2_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
    bn1_2_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn1_2_1.eval()
    conv1_2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
    bn1_2_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn1_2_2.eval()
    conv1_2_3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
    bn1_2_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn1_2_3.eval()

    # layer1 bottleneck2
    conv1_3_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
    bn1_3_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn1_3_1.eval()
    conv1_3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
    bn1_3_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn1_3_2.eval()
    conv1_3_3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
    bn1_3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn1_3_3.eval()

    # layer2 bottleneck0
    conv2_1_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
    bn2_1_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_1_1.eval()
    conv2_1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
    bn2_1_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_1_2.eval()
    conv2_1_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
    bn2_1_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_1_3.eval()
    conv2_1_4 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
    bn2_1_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_1_4.eval()

    # layer2 bottleneck1
    conv2_2_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
    bn2_2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_2_1.eval()
    conv2_2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
    bn2_2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_2_2.eval()
    conv2_2_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
    bn2_2_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_2_3.eval()

    # layer2 bottleneck2
    conv2_3_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
    bn2_3_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_3_1.eval()
    conv2_3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
    bn2_3_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_3_2.eval()
    conv2_3_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
    bn2_3_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_3_3.eval()

    # layer2 bottleneck3
    conv2_4_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
    bn2_4_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_4_1.eval()
    conv2_4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
    bn2_4_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_4_2.eval()
    conv2_4_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
    bn2_4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn2_4_3.eval()

    # layer3 bottleneck0
    conv3_1_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
    bn3_1_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_1_1.eval()
    conv3_1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
    bn3_1_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_1_2.eval()
    conv3_1_3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
    bn3_1_3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_1_3.eval()
    conv3_1_4 = nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False)
    bn3_1_4 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_1_4.eval()

    # layer3 bottleneck1
    conv3_2_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
    bn3_2_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_2_1.eval()
    conv3_2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
    bn3_2_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_2_2.eval()
    conv3_2_3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
    bn3_2_3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_2_3.eval()

    # layer3 bottleneck2
    conv3_3_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
    bn3_3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_3_1.eval()
    conv3_3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
    bn3_3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_3_2.eval()
    conv3_3_3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
    bn3_3_3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_3_3.eval()

    # layer3 bottleneck3
    conv3_4_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
    bn3_4_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_4_1.eval()
    conv3_4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
    bn3_4_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_4_2.eval()
    conv3_4_3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
    bn3_4_3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_4_3.eval()

    # layer3 bottleneck4
    conv3_5_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
    bn3_5_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_5_1.eval()
    conv3_5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
    bn3_5_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_5_2.eval()
    conv3_5_3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
    bn3_5_3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_5_3.eval()

    # layer3 bottleneck5
    conv3_6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
    bn3_6_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_6_1.eval()
    conv3_6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
    bn3_6_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_6_2.eval()
    conv3_6_3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
    bn3_6_3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn3_6_3.eval()

    # layer4 bottleneck0
    conv4_1_1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
    bn4_1_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn4_1_1.eval()
    conv4_1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
    bn4_1_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn4_1_2.eval()
    conv4_1_3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
    bn4_1_3 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn4_1_3.eval()
    conv4_1_4 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False)
    bn4_1_4 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn4_1_4.eval()

    # layer4 bottleneck1
    conv4_2_1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
    bn4_2_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn4_2_1.eval()
    conv4_2_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
    bn4_2_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn4_2_2.eval()
    conv4_2_3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
    bn4_2_3 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn4_2_3.eval()

    # layer4 bottleneck2
    conv4_3_1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
    bn4_3_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn4_3_1.eval()
    conv4_3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
    bn4_3_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn4_3_2.eval()
    conv4_3_3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
    bn4_3_3 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    bn4_3_3.eval()

    fc = nn.Linear(2048, 2, bias=True)

    w0_1 = resNet50['conv1.weight']
    bn1_weight = resNet50['bn1.weight']
    bn1_bias = resNet50['bn1.bias'] / 2
    bn1_mean = resNet50['bn1.running_mean'] / 2
    bn1_var = resNet50['bn1.running_var']

    w1_1_1 = resNet50['layer1.0.conv1.weight']
    w1_1_2 = resNet50['layer1.0.conv2.weight']
    w1_1_3 = resNet50['layer1.0.conv3.weight']
    w1_1_4 = resNet50['layer1.0.downsample.0.weight']

    w1_2_1 = resNet50['layer1.1.conv1.weight']
    w1_2_2 = resNet50['layer1.1.conv2.weight']
    w1_2_3 = resNet50['layer1.1.conv3.weight']

    w1_3_1 = resNet50['layer1.2.conv1.weight']
    w1_3_2 = resNet50['layer1.2.conv2.weight']
    w1_3_3 = resNet50['layer1.2.conv3.weight']

    w2_1_1 = resNet50['layer2.0.conv1.weight']
    w2_1_2 = resNet50['layer2.0.conv2.weight']
    w2_1_3 = resNet50['layer2.0.conv3.weight']
    w2_1_4 = resNet50['layer2.0.downsample.0.weight']

    w2_2_1 = resNet50['layer2.1.conv1.weight']
    w2_2_2 = resNet50['layer2.1.conv2.weight']
    w2_2_3 = resNet50['layer2.1.conv3.weight']

    w2_3_1 = resNet50['layer2.2.conv1.weight']
    w2_3_2 = resNet50['layer2.2.conv2.weight']
    w2_3_3 = resNet50['layer2.2.conv3.weight']

    w2_4_1 = resNet50['layer2.3.conv1.weight']
    w2_4_2 = resNet50['layer2.3.conv2.weight']
    w2_4_3 = resNet50['layer2.3.conv3.weight']

    w3_1_1 = resNet50['layer3.0.conv1.weight']
    w3_1_2 = resNet50['layer3.0.conv2.weight']
    w3_1_3 = resNet50['layer3.0.conv3.weight']
    w3_1_4 = resNet50['layer3.0.downsample.0.weight']

    w3_2_1 = resNet50['layer3.1.conv1.weight']
    w3_2_2 = resNet50['layer3.1.conv2.weight']
    w3_2_3 = resNet50['layer3.1.conv3.weight']

    w3_3_1 = resNet50['layer3.2.conv1.weight']
    w3_3_2 = resNet50['layer3.2.conv2.weight']
    w3_3_3 = resNet50['layer3.2.conv3.weight']

    w3_4_1 = resNet50['layer3.3.conv1.weight']
    w3_4_2 = resNet50['layer3.3.conv2.weight']
    w3_4_3 = resNet50['layer3.3.conv3.weight']

    w3_5_1 = resNet50['layer3.4.conv1.weight']
    w3_5_2 = resNet50['layer3.4.conv2.weight']
    w3_5_3 = resNet50['layer3.4.conv3.weight']

    w3_6_1 = resNet50['layer3.5.conv1.weight']
    w3_6_2 = resNet50['layer3.5.conv2.weight']
    w3_6_3 = resNet50['layer3.5.conv3.weight']

    w4_1_1 = resNet50['layer4.0.conv1.weight']
    w4_1_2 = resNet50['layer4.0.conv2.weight']
    w4_1_3 = resNet50['layer4.0.conv3.weight']
    w4_1_4 = resNet50['layer4.0.downsample.0.weight']

    w4_2_1 = resNet50['layer4.1.conv1.weight']
    w4_2_2 = resNet50['layer4.1.conv2.weight']
    w4_2_3 = resNet50['layer4.1.conv3.weight']

    w4_3_1 = resNet50['layer4.2.conv1.weight']
    w4_3_2 = resNet50['layer4.2.conv2.weight']
    w4_3_3 = resNet50['layer4.2.conv3.weight']

    fc_w = resNet50['fc.weight']
    fc_bias = resNet50['fc.bias']

    # 参数赋值

    conv0_1.weight = nn.Parameter(w0_1)
    bn0_1.weight = nn.Parameter(bn1_weight)
    bn0_1.bias = nn.Parameter(bn1_bias)
    bn0_1.running_mean = bn1_mean
    bn0_1.running_var = bn1_var
    conv1_1_1.weight = nn.Parameter(w1_1_1)
    bn1_1_1.weight = nn.Parameter(resNet50['layer1.0.bn1.weight'])
    bn1_1_1.bias = nn.Parameter(resNet50['layer1.0.bn1.bias'] / 2)
    bn1_1_1.running_mean = resNet50['layer1.0.bn1.running_mean'] / 2
    bn1_1_1.running_var = resNet50['layer1.0.bn1.running_var']

    conv1_1_2.weight = nn.Parameter(w1_1_2)
    bn1_1_2.weight = nn.Parameter(resNet50['layer1.0.bn2.weight'])
    bn1_1_2.bias = nn.Parameter(resNet50['layer1.0.bn2.bias'] / 2)
    bn1_1_2.running_mean = resNet50['layer1.0.bn2.running_mean'] / 2
    bn1_1_2.running_var = resNet50['layer1.0.bn2.running_var']

    conv1_1_3.weight = nn.Parameter(w1_1_3)
    bn1_1_3.weight = nn.Parameter(resNet50['layer1.0.bn3.weight'])
    bn1_1_3.bias = nn.Parameter(resNet50['layer1.0.bn3.bias'] / 2)
    bn1_1_3.running_mean = resNet50['layer1.0.bn3.running_mean'] / 2
    bn1_1_3.running_var = resNet50['layer1.0.bn3.running_var']

    conv1_1_4.weight = nn.Parameter(w1_1_4)
    bn1_1_4.weight = nn.Parameter(resNet50['layer1.0.downsample.1.weight'])
    bn1_1_4.bias = nn.Parameter(resNet50['layer1.0.downsample.1.bias'] / 2)
    bn1_1_4.running_mean = resNet50['layer1.0.downsample.1.running_mean'] / 2
    bn1_1_4.running_var = resNet50['layer1.0.downsample.1.running_var']

    conv1_2_1.weight = nn.Parameter(w1_2_1)
    bn1_2_1.weight = nn.Parameter(resNet50['layer1.1.bn1.weight'])
    bn1_2_1.bias = nn.Parameter(resNet50['layer1.1.bn1.bias'] / 2)
    bn1_2_1.running_mean = resNet50['layer1.1.bn1.running_mean'] / 2
    bn1_2_1.running_var = resNet50['layer1.1.bn1.running_var']
    conv1_2_2.weight = nn.Parameter(w1_2_2)
    bn1_2_2.weight = nn.Parameter(resNet50['layer1.1.bn2.weight'])
    bn1_2_2.bias = nn.Parameter(resNet50['layer1.1.bn2.bias'] / 2)
    bn1_2_2.running_mean = resNet50['layer1.1.bn2.running_mean'] / 2
    bn1_2_2.running_var = resNet50['layer1.1.bn2.running_var']
    conv1_2_3.weight = nn.Parameter(w1_2_3)
    bn1_2_3.weight = nn.Parameter(resNet50['layer1.1.bn3.weight'])
    bn1_2_3.bias = nn.Parameter(resNet50['layer1.1.bn3.bias'] / 2)
    bn1_2_3.running_mean = resNet50['layer1.1.bn3.running_mean'] / 2
    bn1_2_3.running_var = resNet50['layer1.1.bn3.running_var']

    conv1_3_1.weight = nn.Parameter(w1_3_1)
    bn1_3_1.weight = nn.Parameter(resNet50['layer1.2.bn1.weight'])
    bn1_3_1.bias = nn.Parameter(resNet50['layer1.2.bn1.bias'] / 2)
    bn1_3_1.running_mean = resNet50['layer1.2.bn1.running_mean'] / 2
    bn1_3_1.running_var = resNet50['layer1.2.bn1.running_var']
    conv1_3_2.weight = nn.Parameter(w1_3_2)
    bn1_3_2.weight = nn.Parameter(resNet50['layer1.2.bn2.weight'])
    bn1_3_2.bias = nn.Parameter(resNet50['layer1.2.bn2.bias'] / 2)
    bn1_3_2.running_mean = resNet50['layer1.2.bn2.running_mean'] / 2
    bn1_3_2.running_var = resNet50['layer1.2.bn2.running_var']
    conv1_3_3.weight = nn.Parameter(w1_3_3)
    bn1_3_3.weight = nn.Parameter(resNet50['layer1.2.bn3.weight'])
    bn1_3_3.bias = nn.Parameter(resNet50['layer1.2.bn3.bias'] / 2)
    bn1_3_3.running_mean = resNet50['layer1.2.bn3.running_mean'] / 2
    bn1_3_3.running_var = resNet50['layer1.2.bn3.running_var']

    conv2_1_1.weight = nn.Parameter(w2_1_1)
    bn2_1_1.weight = nn.Parameter(resNet50['layer2.0.bn1.weight'])
    bn2_1_1.bias = nn.Parameter(resNet50['layer2.0.bn1.bias'] / 2)
    bn2_1_1.running_mean = resNet50['layer2.0.bn1.running_mean'] / 2
    bn2_1_1.running_var = resNet50['layer2.0.bn1.running_var']
    conv2_1_2.weight = nn.Parameter(w2_1_2)
    bn2_1_2.weight = nn.Parameter(resNet50['layer2.0.bn2.weight'])
    bn2_1_2.bias = nn.Parameter(resNet50['layer2.0.bn2.bias'] / 2)
    bn2_1_2.running_mean = resNet50['layer2.0.bn2.running_mean'] / 2
    bn2_1_2.running_var = resNet50['layer2.0.bn2.running_var']
    conv2_1_3.weight = nn.Parameter(w2_1_3)
    bn2_1_3.weight = nn.Parameter(resNet50['layer2.0.bn3.weight'])
    bn2_1_3.bias = nn.Parameter(resNet50['layer2.0.bn3.bias'] / 2)
    bn2_1_3.running_mean = resNet50['layer2.0.bn3.running_mean'] / 2
    bn2_1_3.running_var = resNet50['layer2.0.bn3.running_var']
    conv2_1_4.weight = nn.Parameter(w2_1_4)
    bn2_1_4.weight = nn.Parameter(resNet50['layer2.0.downsample.1.weight'])
    bn2_1_4.bias = nn.Parameter(resNet50['layer2.0.downsample.1.bias'] / 2)
    bn2_1_4.running_mean = resNet50['layer2.0.downsample.1.running_mean'] / 2
    bn2_1_4.running_var = resNet50['layer2.0.downsample.1.running_var']

    conv2_2_1.weight = nn.Parameter(w2_2_1)
    bn2_2_1.weight = nn.Parameter(resNet50['layer2.1.bn1.weight'])
    bn2_2_1.bias = nn.Parameter(resNet50['layer2.1.bn1.bias'] / 2)
    bn2_2_1.running_mean = resNet50['layer2.1.bn1.running_mean'] / 2
    bn2_2_1.running_var = resNet50['layer2.1.bn1.running_var']
    conv2_2_2.weight = nn.Parameter(w2_2_2)
    bn2_2_2.weight = nn.Parameter(resNet50['layer2.1.bn2.weight'])
    bn2_2_2.bias = nn.Parameter(resNet50['layer2.1.bn2.bias'] / 2)
    bn2_2_2.running_mean = resNet50['layer2.1.bn2.running_mean'] / 2
    bn2_2_2.running_var = resNet50['layer2.1.bn2.running_var']
    conv2_2_3.weight = nn.Parameter(w2_2_3)
    bn2_2_3.weight = nn.Parameter(resNet50['layer2.1.bn3.weight'])
    bn2_2_3.bias = nn.Parameter(resNet50['layer2.1.bn3.bias'] / 2)
    bn2_2_3.running_mean = resNet50['layer2.1.bn3.running_mean'] / 2
    bn2_2_3.running_var = resNet50['layer2.1.bn3.running_var']

    conv2_3_1.weight = nn.Parameter(w2_3_1)
    bn2_3_1.weight = nn.Parameter(resNet50['layer2.2.bn1.weight'])
    bn2_3_1.bias = nn.Parameter(resNet50['layer2.2.bn1.bias'] / 2)
    bn2_3_1.running_mean = resNet50['layer2.2.bn1.running_mean'] / 2
    bn2_3_1.running_var = resNet50['layer2.2.bn1.running_var']
    conv2_3_2.weight = nn.Parameter(w2_3_2)
    bn2_3_2.weight = nn.Parameter(resNet50['layer2.2.bn2.weight'])
    bn2_3_2.bias = nn.Parameter(resNet50['layer2.2.bn2.bias'] / 2)
    bn2_3_2.running_mean = resNet50['layer2.2.bn2.running_mean'] / 2
    bn2_3_2.running_var = resNet50['layer2.2.bn2.running_var']
    conv2_3_3.weight = nn.Parameter(w2_3_3)
    bn2_3_3.weight = nn.Parameter(resNet50['layer2.2.bn3.weight'])
    bn2_3_3.bias = nn.Parameter(resNet50['layer2.2.bn3.bias'] / 2)
    bn2_3_3.running_mean = resNet50['layer2.2.bn3.running_mean'] / 2
    bn2_3_3.running_var = resNet50['layer2.2.bn3.running_var']

    conv2_4_1.weight = nn.Parameter(w2_4_1)
    bn2_4_1.weight = nn.Parameter(resNet50['layer2.3.bn1.weight'])
    bn2_4_1.bias = nn.Parameter(resNet50['layer2.3.bn1.bias'] / 2)
    bn2_4_1.running_mean = resNet50['layer2.3.bn1.running_mean'] / 2
    bn2_4_1.running_var = resNet50['layer2.3.bn1.running_var']
    conv2_4_2.weight = nn.Parameter(w2_4_2)
    bn2_4_2.weight = nn.Parameter(resNet50['layer2.3.bn2.weight'])
    bn2_4_2.bias = nn.Parameter(resNet50['layer2.3.bn2.bias'] / 2)
    bn2_4_2.running_mean = resNet50['layer2.3.bn2.running_mean'] / 2
    bn2_4_2.running_var = resNet50['layer2.3.bn2.running_var']
    conv2_4_3.weight = nn.Parameter(w2_4_3)
    bn2_4_3.weight = nn.Parameter(resNet50['layer2.3.bn3.weight'])
    bn2_4_3.bias = nn.Parameter(resNet50['layer2.3.bn3.bias'] / 2)
    bn2_4_3.running_mean = resNet50['layer2.3.bn3.running_mean'] / 2
    bn2_4_3.running_var = resNet50['layer2.3.bn3.running_var']

    conv3_1_1.weight = nn.Parameter(w3_1_1)
    bn3_1_1.weight = nn.Parameter(resNet50['layer3.0.bn1.weight'])
    bn3_1_1.bias = nn.Parameter(resNet50['layer3.0.bn1.bias'] / 2)
    bn3_1_1.running_mean = resNet50['layer3.0.bn1.running_mean'] / 2
    bn3_1_1.running_var = resNet50['layer3.0.bn1.running_var']
    conv3_1_2.weight = nn.Parameter(w3_1_2)
    bn3_1_2.weight = nn.Parameter(resNet50['layer3.0.bn2.weight'])
    bn3_1_2.bias = nn.Parameter(resNet50['layer3.0.bn2.bias'] / 2)
    bn3_1_2.running_mean = resNet50['layer3.0.bn2.running_mean'] / 2
    bn3_1_2.running_var = resNet50['layer3.0.bn2.running_var']
    conv3_1_3.weight = nn.Parameter(w3_1_3)
    bn3_1_3.weight = nn.Parameter(resNet50['layer3.0.bn3.weight'])
    bn3_1_3.bias = nn.Parameter(resNet50['layer3.0.bn3.bias'] / 2)
    bn3_1_3.running_mean = resNet50['layer3.0.bn3.running_mean'] / 2
    bn3_1_3.running_var = resNet50['layer3.0.bn3.running_var']
    conv3_1_4.weight = nn.Parameter(w3_1_4)
    bn3_1_4.weight = nn.Parameter(resNet50['layer3.0.downsample.1.weight'])
    bn3_1_4.bias = nn.Parameter(resNet50['layer3.0.downsample.1.bias'] / 2)
    bn3_1_4.running_mean = resNet50['layer3.0.downsample.1.running_mean'] / 2
    bn3_1_4.running_var = resNet50['layer3.0.downsample.1.running_var']

    conv3_2_1.weight = nn.Parameter(w3_2_1)
    bn3_2_1.weight = nn.Parameter(resNet50['layer3.1.bn1.weight'])
    bn3_2_1.bias = nn.Parameter(resNet50['layer3.1.bn1.bias'] / 2)
    bn3_2_1.running_mean = resNet50['layer3.1.bn1.running_mean'] / 2
    bn3_2_1.running_var = resNet50['layer3.1.bn1.running_var']
    conv3_2_2.weight = nn.Parameter(w3_2_2)
    bn3_2_2.weight = nn.Parameter(resNet50['layer3.1.bn2.weight'])
    bn3_2_2.bias = nn.Parameter(resNet50['layer3.1.bn2.bias'] / 2)
    bn3_2_2.running_mean = resNet50['layer3.1.bn2.running_mean'] / 2
    bn3_2_2.running_var = resNet50['layer3.1.bn2.running_var']
    conv3_2_3.weight = nn.Parameter(w3_2_3)
    bn3_2_3.weight = nn.Parameter(resNet50['layer3.1.bn3.weight'])
    bn3_2_3.bias = nn.Parameter(resNet50['layer3.1.bn3.bias'] / 2)
    bn3_2_3.running_mean = resNet50['layer3.1.bn3.running_mean'] / 2
    bn3_2_3.running_var = resNet50['layer3.1.bn3.running_var']

    conv3_3_1.weight = nn.Parameter(w3_3_1)
    bn3_3_1.weight = nn.Parameter(resNet50['layer3.2.bn1.weight'])
    bn3_3_1.bias = nn.Parameter(resNet50['layer3.2.bn1.bias'] / 2)
    bn3_3_1.running_mean = resNet50['layer3.2.bn1.running_mean'] / 2
    bn3_3_1.running_var = resNet50['layer3.2.bn1.running_var']
    conv3_3_2.weight = nn.Parameter(w3_3_2)
    bn3_3_2.weight = nn.Parameter(resNet50['layer3.2.bn2.weight'])
    bn3_3_2.bias = nn.Parameter(resNet50['layer3.2.bn2.bias'] / 2)
    bn3_3_2.running_mean = resNet50['layer3.2.bn2.running_mean'] / 2
    bn3_3_2.running_var = resNet50['layer3.2.bn2.running_var']
    conv3_3_3.weight = nn.Parameter(w3_3_3)
    bn3_3_3.weight = nn.Parameter(resNet50['layer3.2.bn3.weight'])
    bn3_3_3.bias = nn.Parameter(resNet50['layer3.2.bn3.bias'] / 2)
    bn3_3_3.running_mean = resNet50['layer3.2.bn3.running_mean'] / 2
    bn3_3_3.running_var = resNet50['layer3.2.bn3.running_var']

    conv3_4_1.weight = nn.Parameter(w3_4_1)
    bn3_4_1.weight = nn.Parameter(resNet50['layer3.3.bn1.weight'])
    bn3_4_1.bias = nn.Parameter(resNet50['layer3.3.bn1.bias'] / 2)
    bn3_4_1.running_mean = resNet50['layer3.3.bn1.running_mean'] / 2
    bn3_4_1.running_var = resNet50['layer3.3.bn1.running_var']
    conv3_4_2.weight = nn.Parameter(w3_4_2)
    bn3_4_2.weight = nn.Parameter(resNet50['layer3.3.bn2.weight'])
    bn3_4_2.bias = nn.Parameter(resNet50['layer3.3.bn2.bias'] / 2)
    bn3_4_2.running_mean = resNet50['layer3.3.bn2.running_mean'] / 2
    bn3_4_2.running_var = resNet50['layer3.3.bn2.running_var']
    conv3_4_3.weight = nn.Parameter(w3_4_3)
    bn3_4_3.weight = nn.Parameter(resNet50['layer3.3.bn3.weight'])
    bn3_4_3.bias = nn.Parameter(resNet50['layer3.3.bn3.bias'] / 2)
    bn3_4_3.running_mean = resNet50['layer3.3.bn3.running_mean'] / 2
    bn3_4_3.running_var = resNet50['layer3.3.bn3.running_var']

    conv3_5_1.weight = nn.Parameter(w3_5_1)
    bn3_5_1.weight = nn.Parameter(resNet50['layer3.4.bn1.weight'])
    bn3_5_1.bias = nn.Parameter(resNet50['layer3.4.bn1.bias'] / 2)
    bn3_5_1.running_mean = resNet50['layer3.4.bn1.running_mean'] / 2
    bn3_5_1.running_var = resNet50['layer3.4.bn1.running_var']
    conv3_5_2.weight = nn.Parameter(w3_5_2)
    bn3_5_2.weight = nn.Parameter(resNet50['layer3.4.bn2.weight'])
    bn3_5_2.bias = nn.Parameter(resNet50['layer3.4.bn2.bias'] / 2)
    bn3_5_2.running_mean = resNet50['layer3.4.bn2.running_mean'] / 2
    bn3_5_2.running_var = resNet50['layer3.4.bn2.running_var']
    conv3_5_3.weight = nn.Parameter(w3_5_3)
    bn3_5_3.weight = nn.Parameter(resNet50['layer3.4.bn3.weight'])
    bn3_5_3.bias = nn.Parameter(resNet50['layer3.4.bn3.bias'] / 2)
    bn3_5_3.running_mean = resNet50['layer3.4.bn3.running_mean'] / 2
    bn3_5_3.running_var = resNet50['layer3.4.bn3.running_var']

    conv3_6_1.weight = nn.Parameter(w3_6_1)
    bn3_6_1.weight = nn.Parameter(resNet50['layer3.5.bn1.weight'])
    bn3_6_1.bias = nn.Parameter(resNet50['layer3.5.bn1.bias'] / 2)
    bn3_6_1.running_mean = resNet50['layer3.5.bn1.running_mean'] / 2
    bn3_6_1.running_var = resNet50['layer3.5.bn1.running_var']
    conv3_6_2.weight = nn.Parameter(w3_6_2)
    bn3_6_2.weight = nn.Parameter(resNet50['layer3.5.bn2.weight'])
    bn3_6_2.bias = nn.Parameter(resNet50['layer3.5.bn2.bias'] / 2)
    bn3_6_2.running_mean = resNet50['layer3.5.bn2.running_mean'] / 2
    bn3_6_2.running_var = resNet50['layer3.5.bn2.running_var']
    conv3_6_3.weight = nn.Parameter(w3_6_3)
    bn3_6_3.weight = nn.Parameter(resNet50['layer3.5.bn3.weight'])
    bn3_6_3.bias = nn.Parameter(resNet50['layer3.5.bn3.bias'] / 2)
    bn3_6_3.running_mean = resNet50['layer3.5.bn3.running_mean'] / 2
    bn3_6_3.running_var = resNet50['layer3.5.bn3.running_var']

    conv4_1_1.weight = nn.Parameter(w4_1_1)
    bn4_1_1.weight = nn.Parameter(resNet50['layer4.0.bn1.weight'])
    bn4_1_1.bias = nn.Parameter(resNet50['layer4.0.bn1.bias'] / 2)
    bn4_1_1.running_mean = resNet50['layer4.0.bn1.running_mean'] / 2
    bn4_1_1.running_var = resNet50['layer4.0.bn1.running_var']
    conv4_1_2.weight = nn.Parameter(w4_1_2)
    bn4_1_2.weight = nn.Parameter(resNet50['layer4.0.bn2.weight'])
    bn4_1_2.bias = nn.Parameter(resNet50['layer4.0.bn2.bias'] / 2)
    bn4_1_2.running_mean = resNet50['layer4.0.bn2.running_mean'] / 2
    bn4_1_2.running_var = resNet50['layer4.0.bn2.running_var']
    conv4_1_3.weight = nn.Parameter(w4_1_3)
    bn4_1_3.weight = nn.Parameter(resNet50['layer4.0.bn3.weight'])
    bn4_1_3.bias = nn.Parameter(resNet50['layer4.0.bn3.bias'] / 2)
    bn4_1_3.running_mean = resNet50['layer4.0.bn3.running_mean'] / 2
    bn4_1_3.running_var = resNet50['layer4.0.bn3.running_var']
    conv4_1_4.weight = nn.Parameter(w4_1_4)
    bn4_1_4.weight = nn.Parameter(resNet50['layer4.0.downsample.1.weight'])
    bn4_1_4.bias = nn.Parameter(resNet50['layer4.0.downsample.1.bias'] / 2)
    bn4_1_4.running_mean = resNet50['layer4.0.downsample.1.running_mean'] / 2
    bn4_1_4.running_var = resNet50['layer4.0.downsample.1.running_var']

    conv4_2_1.weight = nn.Parameter(w4_2_1)
    bn4_2_1.weight = nn.Parameter(resNet50['layer4.1.bn1.weight'])
    bn4_2_1.bias = nn.Parameter(resNet50['layer4.1.bn1.bias'] / 2)
    bn4_2_1.running_mean = resNet50['layer4.1.bn1.running_mean'] / 2
    bn4_2_1.running_var = resNet50['layer4.1.bn1.running_var']
    conv4_2_2.weight = nn.Parameter(w4_2_2)
    bn4_2_2.weight = nn.Parameter(resNet50['layer4.1.bn2.weight'])
    bn4_2_2.bias = nn.Parameter(resNet50['layer4.1.bn2.bias'] / 2)
    bn4_2_2.running_mean = resNet50['layer4.1.bn2.running_mean'] / 2
    bn4_2_2.running_var = resNet50['layer4.1.bn2.running_var']
    conv4_2_3.weight = nn.Parameter(w4_2_3)
    bn4_2_3.weight = nn.Parameter(resNet50['layer4.1.bn3.weight'])
    bn4_2_3.bias = nn.Parameter(resNet50['layer4.1.bn3.bias'] / 2)
    bn4_2_3.running_mean = resNet50['layer4.1.bn3.running_mean'] / 2
    bn4_2_3.running_var = resNet50['layer4.1.bn3.running_var']

    conv4_3_1.weight = nn.Parameter(w4_3_1)
    bn4_3_1.weight = nn.Parameter(resNet50['layer4.2.bn1.weight'])
    bn4_3_1.bias = nn.Parameter(resNet50['layer4.2.bn1.bias'] / 2)
    bn4_3_1.running_mean = resNet50['layer4.2.bn1.running_mean'] / 2
    bn4_3_1.running_var = resNet50['layer4.2.bn1.running_var']
    conv4_3_2.weight = nn.Parameter(w4_3_2)
    bn4_3_2.weight = nn.Parameter(resNet50['layer4.2.bn2.weight'])
    bn4_3_2.bias = nn.Parameter(resNet50['layer4.2.bn2.bias'] / 2)
    bn4_3_2.running_mean = resNet50['layer4.2.bn2.running_mean'] / 2
    bn4_3_2.running_var = resNet50['layer4.2.bn2.running_var']
    conv4_3_3.weight = nn.Parameter(w4_3_3)
    bn4_3_3.weight = nn.Parameter(resNet50['layer4.2.bn3.weight'])
    bn4_3_3.bias = nn.Parameter(resNet50['layer4.2.bn3.bias'] / 2)
    bn4_3_3.running_mean = resNet50['layer4.2.bn3.running_mean'] / 2
    bn4_3_3.running_var = resNet50['layer4.2.bn3.running_var']

    fc.weight = nn.Parameter(fc_w)
    fc.bias = nn.Parameter(fc_bias / 2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    a = []
    for i in range(1, len(sys.argv)):
        a.append(sys.argv[i])

    img_path = a[0]
    detectionId = a[1]

    # img_path = "D:\\pythonStudy\\zlProject\\wuwei\\MultiResNet\\Image\\0000.png"

    img = Image.open(img_path)
    img = tran(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    image_1 = torch.tensor(np.random.random(img.shape))
    image_2 = img - image_1

    rdb = Redis(connection_pool=pool)
    rdb.rpush(detectionId, 'begin')
    # print(rdb.lrange(1, 0, -1))

    results0, results1 = SecResNet50(image_1, image_2, conv0_1, bn0_1, conv1_1_1, bn1_1_1, conv1_1_2, bn1_1_2,
                                     conv1_1_3,
                                     bn1_1_3, conv1_1_4, bn1_1_4, conv1_2_1, bn1_2_1,
                                     conv1_2_2, bn1_2_2, conv1_2_3, bn1_2_3, conv1_3_1, bn1_3_1, conv1_3_2, bn1_3_2,
                                     conv1_3_3, bn1_3_3,
                                     conv2_1_1, bn2_1_1, conv2_1_2, bn2_1_2, conv2_1_3, bn2_1_3, conv2_1_4, bn2_1_4,
                                     conv2_2_1, bn2_2_1, conv2_2_2, bn2_2_2, conv2_2_3, bn2_2_3,
                                     conv2_3_1, bn2_3_1, conv2_3_2, bn2_3_2, conv2_3_3, bn2_3_3, conv2_4_1, bn2_4_1,
                                     conv2_4_2, bn2_4_2, conv2_4_3, bn2_4_3,
                                     conv3_1_1, bn3_1_1, conv3_1_2, bn3_1_2, conv3_1_3, bn3_1_3, conv3_1_4, bn3_1_4,
                                     conv3_2_1, bn3_2_1, conv3_2_2, bn3_2_2, conv3_2_3, bn3_2_3,
                                     conv3_3_1, bn3_3_1, conv3_3_2, bn3_3_2, conv3_3_3, bn3_3_3, conv3_4_1, bn3_4_1,
                                     conv3_4_2, bn3_4_2, conv3_4_3, bn3_4_3,
                                     conv3_5_1, bn3_5_1, conv3_5_2, bn3_5_2, conv3_5_3, bn3_5_3, conv3_6_1, bn3_6_1,
                                     conv3_6_2, bn3_6_2, conv3_6_3, bn3_6_3,
                                     conv4_1_1, bn4_1_1, conv4_1_2, bn4_1_2, conv4_1_3, bn4_1_3, conv4_1_4, bn4_1_4,
                                     conv4_2_1, bn4_2_1, conv4_2_2, bn4_2_2, conv4_2_3, bn4_2_3,
                                     conv4_3_1, bn4_3_1, conv4_3_2, bn4_3_2, conv4_3_3, bn4_3_3, fc, detectionId)

    net_path = r'D:\project\SDNet\SDNet-backend\src\pycodes\model\deepfakeepoch13.pkl'

    resnet50 = torchvision.models.resnet50(pretrained=False)
    fc_featrue = resnet50.fc.in_features
    resnet50.fc = nn.Linear(fc_featrue, 2)
    resnet50.eval()
    resnet50.load_state_dict(torch.load(net_path, map_location="cpu"))
    out = resnet50(img)
    result = results0 + results1

    res1 = results0.numpy()[0].tolist()
    print([float('{:.4f}'.format(i)) for i in res1])
    res2 = results1.numpy()[0].tolist()
    print([float('{:.4f}'.format(i)) for i in res2])

    rdb.rpush(detectionId, 'end')
    # print(rdb.lrange(1, 0, -1))
