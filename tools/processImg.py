import os
from PIL import Image
import numpy as np
from multiprocessing import Pool
from functools import partial


def single_img(x, save_path, bg):
    img_path, is_obj = x
    if not is_obj:
        # bg = Image.new('RGB', (1024, 1024))
        img_tmp = bg  # .copy()
    else:
        img = Image.open(img_path)
        img_tmp = img  # .copy()

    # notice: linux: '/'  windows: '\\'
    path = os.path.join(save_path, img_path.split('\\')[-1])
    img_tmp.save(path, 'PNG')


def pred_obj_img(preds, img_path, save_path):
    bg = Image.new('RGB', (1024, 1024))

    worker = partial(single_img, save_path=save_path, bg=bg)
    pool.map(worker, zip(img_path, preds))


if __name__ == '__main__':
    # 进程数量
    num_process = 24
    pool = Pool(num_process)

    img_path = np.load('../NumpyFiles/val_img_path.npy')
    preds = np.load('../NumpyFiles/predicts_num.npy')

    s_path = r"E:\Dataset\DOTA-Classifier-1.5\pred"
    pred_obj_img(preds, img_path, s_path)
