import os
from PIL import Image
import numpy as np
from multiprocessing import Pool
from functools import partial


def single_img(x, save_path, bg):
    img_path, is_background = x
    if is_background:
        bg = Image.new('RGB', (1024, 1024))
        img_tmp = bg  # .copy()
    else:
        img = Image.open(img_path)
        img_tmp = img  # .copy()
    img_tmp.save(os.path.join(save_path, img_path.split('/')[-1]), 'PNG')


def pred_obj_img(preds, img_path, save_path, best_threshold=0.5):
    is_background = preds < best_threshold
    bg = Image.new('RGB', (1024, 1024))

    worker = partial(single_img, save_path=save_path, bg=bg)
    pool.map(worker, zip(img_path, is_background))


if __name__ == '__main__':
    # 进程数量
    num_process = 24
    pool = Pool(num_process)

    img_path = np.load('NumpyFiles/val_img_path.npy')
    preds = np.load('NumpyFiles/predicts.npy')
    with open('NumpyFiles/best_threshold.txt', 'r') as f:
        best_threshold = float(f.readline())
    s_path = '/home/ubuntu/Dataset/DOTA-Split-MMR/pred/'
    pred_obj_img(preds, img_path, s_path, best_threshold)
