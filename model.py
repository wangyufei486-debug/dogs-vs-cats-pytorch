import torch
from torch import nn


class CatsDogsCNN(nn.Module):

    def __init__(self):
        super(CatsDogsCNN, self).__init__()

        self.model = nn.Sequential(

            # 第一组：3通道 -> 16通道
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 第二组：16通道 -> 32通道
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 第三组：32通道 -> 64通道
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 展平
            nn.Flatten(),

            # 全连接层
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),

            # 最终输出猫、狗两个类别
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 测试网络
if __name__ == "__main__":

    model = CatsDogsCNN()

    # 模拟一个batch：
    # 32张RGB图片，每张128×128
    x = torch.ones((32, 3, 128, 128))

    output = model(x)

    print("输入形状：", x.shape)
    print("输出形状：", output.shape)
    print(output)