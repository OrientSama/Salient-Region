### 记录实验结果
***
#### 实验参数
* 二分类 epoch：250 batch_size:16 lr=0.001 lrf=0.01 eval_interval=5
* 10419 images for training. 3317 images for validation.
* acc: ![acc](./log_src/acc_efficientnet_2.png)
* loss:![loss](./log_src/loss_efficientnet_2.png)
* 混淆矩阵：![cm](./log_src/efficient_model_130-1Class-None.png)
* 分类消耗时间：1分30秒  batch_size:64 速度 1.74s/it