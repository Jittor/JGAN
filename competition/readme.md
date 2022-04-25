## 第二届计图人工智能挑战赛

### 计图挑战热身赛

本赛题将会提供数字图片数据集 MNIST，参赛选手需要训练一个将随机噪声和类别标签映射为数字图片的Conditional GAN模型，并生成注册时绑定的手机号（如果没有绑定手机号请先绑定再进行提交）。

本赛题提供示例代码框架，提供数据下载、模型定义、训练步骤等功能。

选手可以基于示例代码填充注释为 TODO 的部分完成该赛题。

```
git clone https://github.com/Jittor/gan-jittor.git
cd gan-jittor/
sudo python3.7 -m pip install -r requirements.txt
cd competition/warm_up_comp
修改 CGAN.py 使其运行
```

### 赛题一：风景图片生成赛题

图像生成任务一直以来都是十分具有应用场景的计算机视觉任务，从语义分割图生成有意义、高质量的图片仍然存在诸多挑战，如保证生成图片的真实性、清晰程度、多样性、美观性等。

清华大学计算机系图形学实验室从Flickr官网收集了1万张高清（宽1024、高768）的风景图片，并制作了它们的语义分割图。其中，1万对图片被用来训练。训练数据集可以从[这里](https://cloud.tsinghua.edu.cn/f/1d734cbb68b545d6bdf2/?dl=1)下载。

```
git clone https://github.com/Jittor/gan-jittor.git
cd gan-jittor/
sudo python3.7 -m pip install -r requirements.txt
cd competition/landscape_comp

# 单卡训练，需要修改脚本里的数据路径
bash scripts/single_gpu.sh

# 多卡训练，需要修改脚本里的数据路径
bash scripts/multi_gpu.sh
```

注：代码中注释掉了eval的部分，等到测试数据发布之后，您可以取消注释进行评测。也可在训练阶段自动分配一部分数据集为测试集进行训练。

下面展示了一些 baseline 训练完的结果。
![1650852742(1)](https://user-images.githubusercontent.com/43036573/165009453-3301902a-8616-4b61-b861-262b81c78747.jpg)

