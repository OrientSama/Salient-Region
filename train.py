import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import efficientnetv2_m as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
import datetime


def save_weights(args, epoch, model):
    if not os.path.exists(args.weights_path):
        os.mkdir(args.weights_path)
    torch.save(model.state_dict(), "./{}/{}-{}-model-{}.pth".format(args.weights_path, args.num_classes,
                                                                       "NF" if args.weights == ""
                                                                       else "F-" + args.weights.split('/')[-1].split('.')[0],
                                                                       epoch))


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    # 获取当前日期和时间
    now = datetime.datetime.now()
    # 格式化日期和时间
    formatted_date = now.strftime("%Y-%m-%d")
    formatted_time = now.strftime("%H:%M")
    fdt = formatted_date + '-' + formatted_time
    args.weights_path += '-' + fdt
    print('Start Tensorboard with "tensorboard --logdir {}", view at http://localhost:6006/'.format(fdt))
    comment = '_{}_{}'.format(args.num_classes, args.epochs)
    tb_writer = SummaryWriter(log_dir="./{}".format(fdt), comment=comment)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    multi_label = True if args.num_classes > 1 else False
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path,
                                                                                               multi_label)

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [512, 600],
                "l": [384, 480]}
    num_model = "m"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 20])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)

    # 最多训练 head 和 block 的后10层
    li = [str(n) for n in range(46, 57)] + ['head']

    # head层的名字、参数
    head_para = [k for k, v in model.named_parameters() if 'head' in k]
    headpara = [p for n, p in model.named_parameters() if p.requires_grad and 'head' in n]
    # 后10层 block层的名字、参数
    block_para = [k for k, v in model.named_parameters() if k.split('.')[1] in li]
    basepara = [p for n, p in model.named_parameters() if p.requires_grad and n.split('.')[1] in li]

    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    # 分层控制学习率
    optimizer = optim.AdamW([{'params': basepara}, {'params': headpara, 'lr': args.lr * 10}], lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if opt.amp else None

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
            if opt.amp and "scaler" in weights_dict:
                scaler.load_state_dict(weights_dict["scaler"])
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    tags = ["loss", "acc", "learning_rate"]
    # best_val_loss, best_val_acc = 1, 0
    # init_epochs = 5
    # # 先冻结其他层， 训练分类层
    # for name, para in model.named_parameters():
    #     # 除head外，其他权重全部冻结
    #     if "head" not in name:
    #         para.requires_grad_(False)
    #     else:
    #         print("training {}".format(name))
    #
    # for epoch in range(init_epochs):
    #     # warmup
    #     train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
    #                                             device=device, epoch=epoch, multilabel=multi_label, warmup=True, scaler=scaler)
    #     # validate
    #     val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch,
    #                                  multilabel=multi_label)
    #     tb_writer.add_scalars(tags[0], {'Train': train_loss}, epoch)
    #     tb_writer.add_scalars(tags[0], {'Val': val_loss}, epoch)
    #     tb_writer.add_scalars(tags[1], {'Train': train_acc}, epoch)
    #     tb_writer.add_scalars(tags[1], {'Val': val_acc}, epoch)
    #     tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
    #     # if val_loss < best_val_loss or val_acc > best_val_acc:
    #     #     # 如果 loss 小 或者 acc 大（比之前最好的） 则 保存 权重文件
    #     #     best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss
    #     #     best_val_acc = val_acc if val_acc > best_val_acc else best_val_acc
    #     save_weights(args, epoch, model)

    # 是否冻结权重
    for name, para in model.named_parameters():
        # 除head外，其他权重全部冻结
        if name not in head_para and name not in block_para:
            para.requires_grad_(False)
        else:
            para.requires_grad_(True)
            print("training {}".format(name))

    for epoch in range(0, args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                device=device,
                                                epoch=epoch, multilabel=multi_label, warmup=True, scaler=scaler)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device,
                                     epoch=epoch, multilabel=multi_label)

        tb_writer.add_scalars(tags[0], {'Train': train_loss}, epoch)
        tb_writer.add_scalars(tags[0], {'Val': val_loss}, epoch)
        tb_writer.add_scalars(tags[1], {'Train': train_acc}, epoch)
        tb_writer.add_scalars(tags[1], {'Val': val_acc}, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        # if val_loss <= best_val_loss and val_acc >= best_val_acc:
        #     # 如果 loss 小 或者 acc 大（比之前最好的） 则 保存 权重文件
        #     best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss
        #     best_val_acc = val_acc if val_acc > best_val_acc else best_val_acc
        save_weights(args, epoch, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weights_path', type=str, default='15c_10blocks_Finetune')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default=r"/home/ubuntu/Dataset/DOTA-Split")

    # download model weights
    # 链接: https://pan.baidu.com/s/1uZX36rvrfEss-JGj4yfzbQ  密码: 5gu1
    # 预训练模型 E:\Dataset\pre_efficientnetv2-s.pth
    parser.add_argument('--weights', type=str,
                        default=r"/home/ubuntu/PreTrainWeights/torch_efficientnetv2/pre_efficientnetv2-m.pth",
                        help='initial weights path')
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
