import torch
from torch import nn


class ResidualBlock(nn.Module):
    """残差块：学习到的特征 + 捷径分支。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # 通道数或空间尺寸变化时，让捷径分支同步调整
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x))


class SmallResNet(nn.Module):
    """从零训练的小型残差网络，不使用预训练权重。"""

    def __init__(self, num_classes=5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def check_model():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 分别验证保持尺寸和改变尺寸的残差块
    for in_ch, out_ch, stride in [(32, 32, 1), (32, 64, 2)]:
        block = ResidualBlock(in_ch, out_ch, stride).to(device)
        inputs = torch.randn(2, in_ch, 32, 32, device=device)
        outputs = block(inputs)
        expected = (2, out_ch, 32 // stride, 32 // stride)
        assert tuple(outputs.shape) == expected
        print(f"残差块：{tuple(inputs.shape)} -> {tuple(outputs.shape)}")

    model = SmallResNet(num_classes=5).to(device)
    images = torch.randn(2, 3, 128, 128, device=device)
    labels = torch.tensor([0, 4], device=device)

    logits = model(images)
    assert tuple(logits.shape) == (2, 5)

    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"\n运行设备：{device}")
    print(f"输入形状：{tuple(images.shape)}")
    print(f"输出形状：{tuple(logits.shape)}")
    print(f"模型参数量：{parameter_count:,}")
    print("网络前向计算和反向传播检查通过")


if __name__ == "__main__":
    check_model()