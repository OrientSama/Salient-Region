import os
import sys
import json
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda import amp
import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def read_label(root: str, class_indices: dict, multilabel: bool):
    labels = []
    label_txt = [os.path.join(root, i) for i in os.listdir(root)]
    label_txt.sort()
    if multilabel:
        for i in label_txt:
            with open(i, "r") as f:
                label = [0.0] * 16
                data = f.readlines()
                if len(data) != 0:
                    tmp = set()
                    for line in data:
                        tmp.add(line.split(' ')[-2])
                    tmp_l = list(tmp)
                    for t in tmp_l:
                        label[int(class_indices[t])] = 1.0
                # else:
                #     label[int(class_indices['background'])] = 1.0
                #     # 背景， 图片里无目标
                labels.append(label)
    # 2 label
    else:
        for i in label_txt:
            with open(i, "r") as f:
                data = f.readlines()
                if len(data) != 0:
                    label = 1.0
                else:
                    label = 0.0
                    # 15 代表背景， 图片里无目标
                labels.append(label)
    return labels


def read_split_data(root: str, multilabel: bool):
    # random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    train_path = os.path.join(root, "train")
    val_path = os.path.join(root, "val")
    # 遍历文件夹，一个文件夹对应一个类别

    dota_class = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
                  'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
                  'swimming-pool', 'helicopter', 'container-crane']
    dota_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(dota_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    train_img_path = os.path.join(train_path, "images")
    train_label_path = os.path.join(train_path, "annfiles")
    train_images_path = [os.path.join(train_img_path, i) for i in os.listdir(train_img_path)]  # 存储训练集的所有图片路径
    train_images_path.sort()
    train_images_label = read_label(train_label_path, class_indices, multilabel)  # 存储训练集图片对应索引信息

    val_img_path = os.path.join(val_path, "images")
    val_label_path = os.path.join(val_path, "annfiles")
    val_images_path = [os.path.join(val_img_path, i) for i in os.listdir(val_img_path)]  # 存储验证集的所有图片路径
    val_images_path.sort()
    val_images_label = read_label(val_label_path, class_indices, multilabel)  # 存储验证集图片对应索引信息

    train_mask_img_path = os.path.join(train_path, "mask_images")  # mask
    train_mask_images_path = [os.path.join(train_mask_img_path, i) for i in os.listdir(train_mask_img_path)]
    train_mask_images_path.sort()

    val_mask_img_path = os.path.join(val_path, "mask_images")
    val_mask_images_path = [os.path.join(val_mask_img_path, i) for i in os.listdir(val_mask_img_path)]
    val_mask_images_path.sort()

    # print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images of tarin mask.".format(len(train_mask_images_path)))
    print("{} images of val mask.".format(len(val_mask_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    assert len(train_mask_images_path) > 0, "number of train mask images must greater than 0."
    assert len(val_mask_images_path) > 0, "number of val mask images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label, train_mask_images_path, val_mask_images_path


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def train_one_epoch(model, optimizer, data_loader, device, epoch, warmup, multilabel, scaler):
    model.train()
    # if multilabel:
    loss_function = torch.nn.BCEWithLogitsLoss()
    focal_loss = FocalLoss(loss_function)
    m = torch.nn.Sigmoid()
    b = torch.ones(1, device=device) * 0.03

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, masks = data
        sample_num += images.shape[0]
        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with amp.autocast(enabled=scaler is not None):
            # model with FPN
            pred, output = model(images.to(device))

            # efficientnet
            # pred, _ = model(images.to(device))
            # resnet50
            # pred = model(images.to(device))
            # if multilabel:
            m_pred = m(pred)
            pred_classes = torch.round(m_pred)
            if not multilabel:
                labels = labels.unsqueeze(dim=1)
            # flooding    https://arxiv.org/pdf/2002.08709.pdf
            loss = torch.abs_(focal_loss(pred, labels.to(device)) - b) + b + 0.05 * loss_function(output, masks.to(device)).sum() / output.size(0)
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, multilabel):
    # if multilabel:
    loss_function = torch.nn.BCEWithLogitsLoss()
    focal_loss = FocalLoss(loss_function)
    m = torch.nn.Sigmoid()
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, masks = data
        sample_num += images.shape[0]

        # model with FPN
        pred, output = model(images.to(device))
        # resnet50
        # pred = model(images.to(device))
        m_pred = m(pred)
        pred_classes = torch.round(m_pred)
        if not multilabel:
            labels = labels.unsqueeze(dim=1)
        loss = focal_loss(pred, labels.to(device)) + 0.05 * loss_function(output, masks.to(device)).sum() / output.size(0)
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        accu_loss += loss.detach()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
