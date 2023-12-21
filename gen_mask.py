import torch
import matplotlib.pyplot as plt
import math
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
from multiprocessing import Pool
from functools import partial
from PIL import Image


# Generate a mask on the image
def G(imgs, x, y):
    bx, by = x.mean(), y.mean()
    coord_list = [(x, y) for x, y in zip(x, y)]
    polygon = Polygon(coord_list)
    point_0, point_1, point_3 = Point(coord_list[0]), Point(coord_list[1]), Point(coord_list[3])
    box_w, box_h = point_0.distance(point_1), point_0.distance(point_3)

    for p_x in range(x.min().astype(int), x.max().astype(int) + 1):  # 列
        for p_y in range(y.min().astype(int), y.max().astype(int) + 1):  # 行
            point = Point(p_x, p_y)
            if polygon.contains(point):
                # imgs[0][p_y][p_x] = Gaussian(p_x, p_y, bx, by, 2, 0.001* box_w, 1, 0.001* box_h)
                imgs[p_y][p_x] = 255


# Analyze the coordinates from the text
def analyze_coord(str):
    split_str = str.split(' ')[:8]
    coord = [(float(split_str[2 * i]), float(split_str[2 * i + 1])) for i in range(len(split_str) // 2)]
    x = [c[0] for c in coord]
    y = [c[1] for c in coord]
    return np.array(x), np.array(y)


def mask_single_img(labels_path, mask_imgs_path, pic_name):
    imgs = np.zeros((1024, 1024))
    assert os.path.exists(os.path.join(labels_path, pic_name))
    with open(os.path.join(labels_path, pic_name), 'r') as f:
        strlines = f.readlines()
        # print(strlines)
        for str in strlines:
            x, y = analyze_coord(str)
            G(imgs, x, y)

    img_save = Image.fromarray(imgs)
    img_save.save(os.path.join(mask_imgs_path, pic_name.replace('.txt', '.jpg')))


def main():
    # 进程数量
    num_process = 20
    pool = Pool(num_process)
    base_path = '/home/ubuntu/Dataset/DOTA-Split/trainSplit-1024'
    mask_imgs_path = os.path.join(base_path, 'mask_images')
    if not os.path.exists(mask_imgs_path):
        os.mkdir(mask_imgs_path)
    labels_path = os.path.join(base_path, 'labelTxt')
    pic_name_list = os.listdir(labels_path)
    print("Fount {} Labels".format(len(pic_name_list)))
    # for pic_name in pic_name_list:
    worker = partial(mask_single_img, labels_path=labels_path, mask_imgs_path=mask_imgs_path)
    pool.map(worker, pic_name_list)


if __name__ == "__main__":
    main()
