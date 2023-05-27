# MNIST图像识别与生成对抗样本攻击

## 1 目录结构

- data/   存放MNIST数据集，可新增其他数据集
- GAN/
  - gan_images/
  - DCGAN_Images/
  - GAN.py  GAN算法主要代码
  - DCGAN.py  DCGAN算法主要代码
- FGSM.py  FGSM算法主要代码
- main.py  CNN主要代码
- model.pth  训练后生成的模型
- optimizer.pth  训练后生成的优化器

## 2 CNN模型训练与推断

### 模型详情

2个卷积+1个dropout+2个全连接

- 卷积层1：输入通道数为1，输出通道数为10，使用5x5卷积核，然后用 2x2 的最大池化层下采样，减少计算量，最后用 ReLU 激活。

- 卷积层2：输入通道数为10，输出通道数为20，使用5x5卷积核，然后用 Dropout减小过拟合， 2x2 的最大池化层下采样，使用 ReLU 激活

- 特征图展，形成320长度的向量，成为全连接层的输入；其中view函数的参数-1表示根据其他维度的大小自动推断。

- 全连接层1+ReLU 激活。Dropout。全连接层2。

- 最后用 log-softmax 函数，得到每个类别的预测概率。
- 模型采用随机梯度下降SGD优化
- 损失函数：负对数似然损失函数

### 实验结果

最终模型预测的准确率为98%

## 3 FGSM生成对抗样本

FGSM（Fast Gradient Sign Method）是一种用于生成对抗样本的简单但有效的攻击方法。它通过在原始输入数据上添加扰动，使得经过微小扰动后的样本被误分类。

定义与（一）中一样的CNN模型，初始化网络后，加载main.py中已经训练好的CNN模型，通过FGSM算法生成对抗样本（并可视化），对原模型进行FGSM攻击。

### 实验结果

1. 对抗样本域预测结果可视化（不同epsilon下的对抗样本）：

<img src="C:\Users\Gsc020504\AppData\Roaming\Typora\typora-user-images\image-20230528010459220.png" alt="image-20230528010459220" style="zoom:67%;" />

2. 扰动率对模型准确度的影响

   <img src="C:\Users\Gsc020504\AppData\Roaming\Typora\typora-user-images\image-20230528010536444.png" alt="image-20230528010536444" style="zoom:67%;" />

## 4 GAN/DCGAN生成对抗样本

### GAN

GAN（Generative Adversarial Network）是一种用于生成模型的机器学习算法，通过生成器和判别器之间的对抗训练来学习生成逼真的样本数据。

多次迭代训练过程如下：

- 从真实数据中随机抽取一批样本，为真实样本；
- 从随机噪声中生成一批样本，成为生成样本。
- 使用生成样本和真实样本分别训练判别器D，计算总体损失，并对参数进行反向传播和优化，使其能够区分生成样本和真实样本。
- 固定判别器D的参数后，更新生成器G的参数，使之尽可能欺骗判别器。
- 生成器和判别器逐渐改进，生成器生成的样本逐渐接近真实样本的分布，判别器的判断能力也逐渐提高。

### DCGAN

DCGAN（Deep Convolutional GAN）是GAN的一个变体，特别适用于图像生成任务。DCGAN在生成器和判别器中引入了卷积层和转置卷积层，以利用卷积神经网络的特性。相比于传统的全连接层，卷积层和转置卷积层能够更好地捕捉图像中的局部特征和空间关系。

### 实验结果

1. GAN迭代100轮

   <img src="C:\Users\Gsc020504\AppData\Roaming\Typora\typora-user-images\image-20230528010618237.png" alt="image-20230528010618237" style="zoom: 80%;" />

2. DCGAN迭代20轮

   <img src="C:\Users\Gsc020504\AppData\Roaming\Typora\typora-user-images\image-20230528010629409.png" alt="image-20230528010629409" style="zoom: 80%;" />