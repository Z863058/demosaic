import cv2
import numpy as np
import os
import random
from .image_processing import resize, ch_one2three, mask_area


def addmosaic(img, mask, opt):
    if opt.mosaic_mod == 'random':
        img = addmosaic_random(img, mask)
    elif opt.mosaic_size == 0:
        img = addmosaic_autosize(img, mask, opt.mosaic_mod)
    else:
        img = addmosaic_base(img, mask, opt.mosaic_size, opt.output_size, model=opt.mosaic_mod)
    return img


def addmosaic_base(img, mask, n, out_size=0, model='squa_avg', rect_rat=1.6, feather=0):
    '''
    img: input image
    mask: input mask
    n: mosaic size
    out_size: output size  0->original
    model : squa_avg squa_mid squa_random squa_avg_circle_edge rect_avg
    rect_rat: if model==rect_avg , mosaic w/h=rect_rat
    father : father size, -1->no 0->auto
    '''
    n = int(n)
    if out_size:
        img = resize(img, out_size)
    h, w = img.shape[:2]
    mask = cv2.resize(mask, (w, h))
    img_mosaic = img.copy()

    if model == 'squa_avg':
        for i in range(int(h / n)):
            for j in range(int(w / n)):
                if mask[int(i * n + n / 2), int(j * n + n / 2)] == 255:
                    img_mosaic[i * n:(i + 1) * n, j * n:(j + 1) * n, :] = img[i * n:(i + 1) * n, j * n:(j + 1) * n,
                                                                          :].mean(0).mean(0)

    elif model == 'squa_mid':
        for i in range(int(h / n)):
            for j in range(int(w / n)):
                if mask[int(i * n + n / 2), int(j * n + n / 2)] == 255:
                    img_mosaic[i * n:(i + 1) * n, j * n:(j + 1) * n, :] = img[i * n + int(n / 2), j * n + int(n / 2), :]

    elif model == 'squa_random':
        for i in range(int(h / n)):
            for j in range(int(w / n)):
                if mask[int(i * n + n / 2), int(j * n + n / 2)] == 255:
                    img_mosaic[i * n:(i + 1) * n, j * n:(j + 1) * n, :] = img[int(i * n - n / 2 + n * random.random()),
                                                                          int(j * n - n / 2 + n * random.random()), :]

    elif model == 'squa_avg_circle_edge':
        for i in range(int(h / n)):
            for j in range(int(w / n)):
                img_mosaic[i * n:(i + 1) * n, j * n:(j + 1) * n, :] = img[i * n:(i + 1) * n, j * n:(j + 1) * n, :].mean(
                    0).mean(0)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        _mask = ch_one2three(mask)
        mask_inv = cv2.bitwise_not(_mask)
        imgroi1 = cv2.bitwise_and(_mask, img_mosaic)
        imgroi2 = cv2.bitwise_and(mask_inv, img)
        img_mosaic = cv2.add(imgroi1, imgroi2)

    elif model == 'rect_avg':
        n_h = n
        n_w = int(n * rect_rat)
        for i in range(int(h / n_h)):
            for j in range(int(w / n_w)):
                if mask[int(i * n_h + n_h / 2), int(j * n_w + n_w / 2)] == 255:
                    img_mosaic[i * n_h:(i + 1) * n_h, j * n_w:(j + 1) * n_w, :] = img[i * n_h:(i + 1) * n_h,
                                                                                  j * n_w:(j + 1) * n_w, :].mean(
                        0).mean(0)

    if feather != -1:
        if feather == 0:
            mask = (cv2.blur(mask, (n, n)))
        else:
            mask = (cv2.blur(mask, (feather, feather)))
        mask = ch_one2three(mask) / 255.0
        img_mosaic = (img * (1 - mask) + img_mosaic * mask).astype('uint8')

    return img_mosaic


def get_autosize(img, mask, area_type='normal'):
    h, w = img.shape[:2]
    size = np.min([h, w])
    mask = resize(mask, size)
    alpha = size / 512
    try:
        if area_type == 'normal':
            area = mask_area(mask)
        elif area_type == 'bounding':
            w, h = cv2.boundingRect(mask)[2:]
            area = w * h
    except:
        area = 0
    area = area / (alpha * alpha)
    if area > 50000:
        size = alpha * ((area - 50000) / 50000 + 12)
    elif 20000 < area <= 50000:
        size = alpha * ((area - 20000) / 30000 + 8)
    elif 5000 < area <= 20000:
        size = alpha * ((area - 5000) / 20000 + 7)
    elif 0 <= area <= 5000:
        size = alpha * ((area - 0) / 5000 + 6)
    else:
        pass
    return size


epoch_global = 0


# demosaic_200406_286to256_random_1
# def get_random_parameter(img, mask):
#     # mosaic size
#     # p = np.array([0.5, 0.5])
#     # mod = np.random.choice(['normal', 'bounding'], p=p.ravel())
#     mod = 'normal'
#     mosaic_size = get_autosize(img, mask, area_type=mod)
#     # mosaic_size = int(mosaic_size * random.uniform(0.9, 2.1))
#     r = random.uniform(3.0, 4.5)
#     mosaic_size = int(mosaic_size * r)
#
#     # mosaic mod
#     # p = np.array([0.25, 0.25, 0.1, 0.4])
#     # mod = np.random.choice(['squa_mid', 'squa_avg', 'squa_avg_circle_edge', 'rect_avg'], p=p.ravel())
#     mod = 'squa_avg'
#
#     # rect_rat for rect_avg
#     rect_rat = random.uniform(1.1, 1.6)
#
#     # feather size  羽化效果
#     feather = int(mosaic_size * random.uniform(0, 1.5))
#
#     return mosaic_size, mod, rect_rat, feather

# demosaic_200406_286to256_random_2
# def get_random_parameter(img, mask):
#     # mosaic size
#     # p = np.array([0.5, 0.5])
#     # mod = np.random.choice(['normal', 'bounding'], p=p.ravel())
#     mod = 'normal'
#     mosaic_size = get_autosize(img, mask, area_type=mod)
#     # mosaic_size = int(mosaic_size * random.uniform(0.9, 2.1))
#     if epoch_global < 20:
#         r = random.uniform(0.9, 1.2)
#     elif epoch_global < 40:
#         r = random.uniform(1.3, 1.6)
#     elif epoch_global < 60:
#         r = random.uniform(1.7, 2.0)
#     elif epoch_global < 90:
#         r = random.uniform(2.1, 2.4)
#     elif epoch_global < 120:
#         r = random.uniform(2.5, 2.8)
#     elif epoch_global < 150:
#         r = random.uniform(2.9, 3.2)
#     elif epoch_global < 200:
#         r = random.uniform(3.3, 3.6)
#     elif epoch_global < 250:
#         r = random.uniform(3.7, 4.0)
#     elif epoch_global < 300:
#         r = random.uniform(4.1, 4.4)
#     else:
#         r = random.uniform(0.9, 4.4)
#
#
#     mosaic_size = int(mosaic_size * r)
#
#     # mosaic mod
#     # p = np.array([0.25, 0.25, 0.1, 0.4])
#     # mod = np.random.choice(['squa_mid', 'squa_avg', 'squa_avg_circle_edge', 'rect_avg'], p=p.ravel())
#     mod = 'squa_avg'
#
#     # rect_rat for rect_avg
#     rect_rat = random.uniform(1.1, 1.6)
#
#     # feather size  羽化效果
#     feather = int(mosaic_size * random.uniform(0, 1.5))
#
#     return mosaic_size, mod, rect_rat, feather

# demosaic_200406_286to256_random_3
def get_random_parameter(img, mask):
    # mosaic size
    # p = np.array([0.5, 0.5])
    # mod = np.random.choice(['normal', 'bounding'], p=p.ravel())
    mod = 'normal'
    mosaic_size = get_autosize(img, mask, area_type=mod)
    # mosaic_size = int(mosaic_size * random.uniform(0.9, 2.1))
    r = random.uniform(3.3, 4.3)
    mosaic_size = int(mosaic_size * r)

    # mosaic mod
    # p = np.array([0.25, 0.25, 0.1, 0.4])
    # mod = np.random.choice(['squa_mid', 'squa_avg', 'squa_avg_circle_edge', 'rect_avg'], p=p.ravel())
    mod = 'squa_avg'

    # rect_rat for rect_avg
    rect_rat = random.uniform(1.1, 1.6)

    # feather size  羽化效果
    feather = int(mosaic_size * random.uniform(0, 1.5))

    return mosaic_size, mod, rect_rat, feather


def addmosaic_autosize(img, mask, model, area_type='normal'):
    mosaic_size = get_autosize(img, mask, area_type='normal')
    img_mosaic = addmosaic_base(img, mask, mosaic_size, model=model)
    return img_mosaic


def addmosaic_random(img, mask):
    mosaic_size, mod, rect_rat, feather = get_random_parameter(img, mask)
    img_mosaic = addmosaic_base(img, mask, mosaic_size, model=mod, rect_rat=rect_rat, feather=feather)
    return img_mosaic
