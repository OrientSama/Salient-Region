import os
import numpy as np
from multiprocessing import Pool
from functools import partial


def rmSingle(x, path):
    ann_path, is_obj = x
    if not is_obj:
        real_path = os.path.join(path, ann_path.replace(".png", ".txt").split('\\')[-1])
        try:
            os.remove(real_path)
        except FileNotFoundError:
            pass


def removeAnn(path, ann_path, preds):
    # 分类器识别为非目标的图片
    worker = partial(rmSingle, path=path)
    pool.map(worker, zip(ann_path, preds))


def main():
    path = r'E:\Dataset\DOTA-Classifier-1.5\test\annfiles'
    ann_path = np.load('../NumpyFiles/val_img_path.npy')
    preds = np.load('../NumpyFiles/predicts_num.npy')
    removeAnn(path, ann_path, preds)


if __name__ == '__main__':
    # 进程数量
    num_process = 24
    pool = Pool(num_process)
    main()
