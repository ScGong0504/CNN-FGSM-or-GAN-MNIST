import torch
import torchvision
import os
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_dir = "gan_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

#hyperparameter
image_size = 28 * 28
hidden_size = 400
latent_size = 100
batch_size = 100
epoch_num = 100

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5,0.5)])
data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset=data, shuffle=True, batch_size=batch_size)


D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,1),
    nn.Sigmoid()
)

G = nn.Sequential(

    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)

criteria = nn.BCELoss()
d_optimizer = optim.Adam(params=D.parameters(), lr=0.0001)
g_optimizer = optim.Adam(params=G.parameters(), lr=0.0001)

D = D.to(device)
G = G.to(device)

total_step = len(data_loader)
for epoch in range(epoch_num):
    # 对数据集进行批量的迭代训练
    for i, (images, _) in enumerate(data_loader, 0):

        # 将真实图像数据进行reshape并移动到设备上
        images = images.reshape(batch_size, -1).to(device)

        #训练判别器
        # 创建真实标签和假标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = D(images)  # 在判别器上进行前向传播计算
        # 计算判别器对真实图像的损失
        d_loss_real = criteria(outputs, real_labels)

        # 生成随机噪声，并通过生成器生成假图像
        z = torch.randn(batch_size, latent_size).to(device)
        outputs = D(G(z))
        # 计算判别器D对假图像的损失
        d_loss_fake = criteria(outputs, fake_labels)
        # 计算判别器D总体的损失
        d_loss = d_loss_real + d_loss_fake

        # 对判别器的参数进行反向传播和优化
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        #训练生成器
        # 生成新的随机噪声，并通过生成器生成假图像
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        # 在判别器上进行前向传播
        outputs = D(fake_images)
        # 计算生成器对假图像的损失
        g_loss = criteria(outputs, real_labels)

        # 对生成器的参数进行反向传播和优化
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i % 200 == 0:
            print("Epoch [{} / {}],Step [{} / {}] d_loss = {:.4f}, g_loss = {:.4f}".format(epoch + 1, epoch_num,i, total_step, d_loss.item(), g_loss.item()))

    fake_images = fake_images.reshape(batch_size, 1, 28, 28)
    save_image(fake_images, os.path.join(image_dir, "gan_images-{}.png".format(epoch + 1)))


torch.save(G.state_dict(), "Generator.pth")
torch.save(D.state_dict(), "Discriminator.pth")

