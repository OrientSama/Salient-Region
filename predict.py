import argparse
import os
import json
import sys

import numpy as np
import torch
import pandas as pd
import seaborn as sns
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from my_dataset import MyDataSet
from sklearn.metrics import roc_curve, precision_recall_curve


def main(args):
    if args.size == 's':
        from model import efficientnetv2_s as create_model
    elif args.size == 'm':
        from model import efficientnetv2_m as create_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dota_class = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
                  'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
                  'swimming-pool', 'helicopter', 'background']
    dota_class.sort()
    dota_class.insert(0, 'None')
    name_dict = dict((k, v) for k, v in enumerate(dota_class))
    name_dict['All'] = 'All'

    img_size = {"s": [300, 384],  # train_size, val_size
                # "m": [384, 480],
                "m": [512, 600],
                "l": [384, 480]}
    num_model = args.size

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    # img_path = r"D:\project\Data\valSplit-1024\images\P0003__1__0___0.png"

    # img = Image.open(img_path)
    # plt.imshow(img)
    # # [N, C, H, W]
    # img = data_transform(img)
    # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indices = json.load(f)

    multilabel = args.num_classes > 2
    _, _, val_images_path, val_images_label = utils.read_split_data(r"/home/ubuntu/Dataset/DOTA-Split", multilabel)

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform)
    batch_size = 48
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 20])  # number of workers

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # create model
    model = create_model(num_classes=args.num_classes).to(device)
    # load model weights
    model_weight_path = args.weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    m = torch.nn.Sigmoid()
    model.eval()
    predicts = torch.tensor([]).to(device)
    with torch.no_grad():
        # predict class
        sample_num = 0
        data_loader = tqdm(val_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]

            pred = model(images.to(device))
            # 存储 原始的预测数据， 之后计算 每一项 的最优 阈值
            # if multilabel:
            pred = m(pred)
            # pred_classes = torch.round(pred)
            if not multilabel:
                pred = pred.squeeze()
            predicts = torch.cat([predicts, pred], dim=0)

            # else:
            #     # pred_classes = torch.max(pred, dim=1)[1]
            #     predicts = torch.cat([predicts, pred_classes], dim=0)

    #  confusion matrix
    na = {0: 'Background', 1: 'Object', 'All': 'All'}

    if multilabel:
        background = [0.0] * 15  # 15个0 代表 背景
        # predicts [4055, 15] torch cuda:0
        best_thresholds = []
        predicts = predicts.cpu().numpy()
        labels = np.array(val_images_label)  # 转化为 np
        if args.threshold is None:
            for i in range(15):
                precision, recall, thresholds = precision_recall_curve(labels[:, i], predicts[:, i])
                F1 = 2 * precision * recall / (precision + recall)
                idx = F1.argmax()
                best_thresholds.append(thresholds[idx])
        else:
            best_thresholds = [args.threshold] * 15

            # FPR, recall, thresholds = roc_curve(labels[:, i], predicts[:, i])
            # maxindex = (recall - FPR).argmax()
            # best_thresholds.append(thresholds[maxindex])
        predicts_num = (predicts > np.array(best_thresholds)).astype(float)

        labels_num = [int(label != background) for label in labels.tolist()]
        predicts_num = [int(pred != background) for pred in predicts_num.tolist()]
        # 0 background   1 objects
        labels_num, predicts_num = np.array(labels_num), np.array(predicts_num)  # 转化成np格式
        confusion_matrix = pd.crosstab(labels_num.flatten(), predicts_num.flatten(), margins=True)
        confusion_matrix.rename(index=na, columns=na, inplace=True)
    else:
        labels_num = np.array(val_images_label)
        predicts = predicts.cpu().numpy()
        # FPR 假正率  FPR = FP/(FP + TN)
        # FPR, recall, thresholds = roc_curve(labels_num, predicts)
        if args.threshold is None:
            precision, recall, thresholds = precision_recall_curve(labels_num, predicts)
            F1 = 2 * precision * recall / (precision + recall)
            idx = F1.argmax()
            # maxindex = (recall - FPR).argmax()
            best_threshold = thresholds[idx]
        else:
            best_threshold = args.threshold

        predicts_num = (predicts > best_threshold).astype(float)
        confusion_matrix = pd.crosstab(labels_num.flatten(), predicts_num.flatten(), margins=True)
        confusion_matrix.rename(index=na, columns=na, inplace=True)
    print(confusion_matrix)

    plt.figure(figsize=(8, 8), dpi=200)
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu', annot_kws={'fontsize': 'small'})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')

    plt.title('Confusion Matrix', fontsize=15)
    plt.ylabel('True Value', fontsize=14)
    plt.xlabel('Predict Value', fontsize=14)
    # 保存 图片
    plt.savefig(fname=model_weight_path.split('/')[-1].split('.')[0] + "-{}Class-{}.png".format(args.num_classes, args.threshold),
                bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--weights', type=str,
                        default=r"/home/ubuntu/PycharmProjects/DeepLearn/Test2_MultiLabel/15c_10blocks_Finetune-2023-12-10-15:32/15-F-pre_efficientnetv2-m-model-5.pth",
                        help='initial weights path')
    parser.add_argument('--size', type=str,
                        default="m",
                        help='model size')
    parser.add_argument('--threshold', type=float,
                        default=0.5,
                        help='if None means that auto compute the best threshold, else use given data as threshold')
    opt = parser.parse_args()
    main(opt)
