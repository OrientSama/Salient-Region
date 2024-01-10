import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import efficientnetv2_m as create_model
from mode_with_fpn import ModelWithFPN
from resnet import resnet50
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
    # args.weights_path += '-' + fdt
    print('Start Tensorboard with "tensorboard --logdir {}", view at http://localhost:6006/'.format(fdt))
    comment = '_{}_{}'.format(args.num_classes, args.epochs)
    tb_writer = SummaryWriter(log_dir="./{}".format(fdt), comment=comment)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    multi_label = True if args.num_classes > 1 else False
    train_images_path, train_images_label, val_images_path, val_images_label, train_mask_images_path, val_mask_images_path = read_split_data(
        args.data_path,
        multi_label)

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [512, 512],
                "l": [384, 480]}
    num_model = "m"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0], antialias=True),  # antialias=True 打开抗锯齿
                                     transforms.RandomHorizontalFlip()]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1], antialias=True),
                                   transforms.CenterCrop(img_size[num_model][1])]),
        "norm": transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              mask_images_path=train_mask_images_path,
                              transform=data_transform["train"],
                              norm=data_transform["norm"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            mask_images_path=val_mask_images_path,
                            transform=data_transform["val"],
                            norm=data_transform["norm"])

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
    # model = create_model(num_classes=args.num_classes).to(device)
    # model with FPN
    # model = ModelWithFPN(num_classes=args.num_classes).to(device)
    # ResNet50
    model = resnet50(num_classes=args.num_classes).to(device)

    # 最多训练 head 和 block 的后10层
    # li = [str(n) for n in range(3, 57)] + ['head']

    block_para = [v for k, v in model.named_parameters() if 'fc' not in k]

    # 分层控制学习率
    # optimizer = optim.AdamW([{'params': basepara}, {'params': headpara, 'lr': args.lr * 10}], lr=args.lr, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    optimizer = optim.SGD([{'params': model.fc.parameters(), 'lr': args.lr * 10}, {'params': block_para}], lr=args.lr, weight_decay=1e-4,
                          momentum=0.9)
    scaler = torch.cuda.amp.GradScaler() if opt.amp else None

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location='cpu')
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

    # 是否冻结权重
    # for name, para in model.named_parameters():
    #     # 除head外，其他权重全部冻结
    #     if name not in head_para and name not in block_para:
    #         para.requires_grad_(False)
    #     else:
    #         para.requires_grad_(True)
    #         print("training {}".format(name))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                device=device,
                                                epoch=epoch, multilabel=multi_label, warmup=True, scaler=scaler)

        scheduler.step()

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        # validate
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device,
                                         epoch=epoch, multilabel=multi_label)
            tb_writer.add_scalars(tags[0], {'Val': val_loss}, epoch)
            tb_writer.add_scalars(tags[1], {'Val': val_acc}, epoch)

        tb_writer.add_scalars(tags[0], {'Train': train_loss}, epoch)
        tb_writer.add_scalars(tags[1], {'Train': train_acc}, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        torch.save(save_file, f"save_weights/model_{epoch}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument("--eval-interval", default=5, type=int, help="validation interval default 10 Epochs")
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--resume', default='/home/ubuntu/PycharmProjects/DeepLearn/Test3_Salient_Region/save_weights/model_1.pth',
                        help='resume from checkpoint')
    parser.add_argument('--weights_path', type=str, default='')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default=r"/home/ubuntu/Dataset/DOTA-Split-mmr")

    # download model weights
    # 链接: https://pan.baidu.com/s/1uZX36rvrfEss-JGj4yfzbQ  密码: 5gu1
    # 预训练模型 E:\Dataset\pre_efficientnetv2-s.pth
    # r"/home/ubuntu/PreTrainWeights/torch_efficientnetv2/pre_efficientnetv2-m.pth"
    parser.add_argument('--weights', type=str,
                        default="/home/ubuntu/PreTrainWeights/ResNet/resnet50-19c8e357.pth",
                        help='initial weights path')
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
