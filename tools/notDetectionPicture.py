import os
from PIL import Image
import numpy as np
from multiprocessing import Pool
from functools import partial


# import utils


def single_img(x, save_path):
    img_path, need_copy = x
    if need_copy:
        img = Image.open(img_path)
        img_tmp = img.copy()
        img_tmp.save(os.path.join(save_path, img_path.split('/')[-1]), 'PNG')


def pred_obj_img(preds, img_path, save_path, label, best_threshold=0.5):
    isObj = preds >= best_threshold
    # 原本为背景 识别为目标 label:0 isObj:1
    FP = isObj > label
    # 原本为目标 识别为背景 label：1 isObj:0
    FN = isObj < label

    # bg = Image.new('RGB', (1024, 1024))
    worker = partial(single_img, save_path=os.path.join(save_path, 'FP'))
    pool.map(worker, zip(img_path, FP))

    worker = partial(single_img, save_path=os.path.join(save_path, 'FN'))
    pool.map(worker, zip(img_path, FN))


if __name__ == '__main__':
    # 进程数量
    num_process = 24
    pool = Pool(num_process)

    img_path = np.load('NumpyFiles/val_img_path.npy')
    preds = np.load('NumpyFiles/predicts.npy')
    label = np.load('NumpyFiles/val_images_label.npy')
    with open('NumpyFiles/best_threshold.txt', 'r') as f:
        best_threshold = float(f.readline())
    s_path = '/home/ubuntu/Dataset/DOTA-Split-classification/hard_pred/'
    pred_obj_img(preds, img_path, s_path, label, best_threshold)
