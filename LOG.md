### 记录实验结果
***
#### 实验参数
* 二分类 epoch：250 batch_size:16 lr=0.001 lrf=0.01 eval_interval=5
* 10419 images for training. 3317 images for validation.
* acc: ![acc](./log_src/acc_efficientnet_2.png)
* loss:![loss](./log_src/loss_efficientnet_2.png)
* 混淆矩阵：![cm](./log_src/efficient_model_130-1Class-None.png)
* 分类消耗时间：1分30秒  batch_size:64 速度 1.74s/it
* 将检测无目标变为黑图，测试3317/3317, 21.3 task/s, elapsed: 155s
* 剔除分类器识别无目标的图片， 测试3036/3036, 20.8 task/s, elapsed: 146s
* 测试结果：![result](./log_src/efficient_reslut.png)
* 总耗时：90 + 146 = 236s

***

* 多标签分类 epoch：250 batch_size:16 lr=0.001 lrf=0.01 eval_interval=5
* 10419 images for training. 3317 images for validation.
* acc: ![acc_efficientnet_16.png](log_src%2Facc_efficientnet_16.png)
* loss: ![loss_efficientnet_16.png](log_src%2Floss_efficientnet_16.png)
* 混淆矩阵： ![efficient_180_16c-16Class-None.png](log_src%2Fefficient_180_16c-16Class-None.png)
* 分类消耗时间：1分32秒  batch_size:64 速度 1.77s/it
* 