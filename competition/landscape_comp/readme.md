# Jittor 第二届草图生成风景比赛 baseline


## Requirements

```
jittor
pillow
opencv-python
```

## Train

单卡训练，需要修改脚本里的数据路径
```
bash scripts/single_gpu.sh
```

多卡训练，需要修改脚本里的数据路径
```
bash scripts/multi_gpu.sh
```

注：代码中注释掉了eval的部分，等到测试数据发布之后，您可以取消注释进行评测。也可在训练阶段自动分配一部分数据集为测试集进行训练。