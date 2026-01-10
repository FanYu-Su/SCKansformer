import torch
import torch.nn as nn
from torchvision import transforms

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),      # 强制将任何尺寸转化为 64x64 [cite: 126]
    transforms.ToTensor(),            # 转化为 Tensor 并自动执行 1/255 归一化
])

# --- 2. 定义模型 (对应 Table 3 参数) ---


class WBCsNet(nn.Module):
    def __init__(self, num_classes=4):  # 多分类通常为 4 类
        super(WBCsNet, self).__init__()

        # Input Layer: 64x64x3
        # Layer 1: Conv2D, 16 filters, 3x3, No Padding
        # 输出形状计算: (64-3+1) = 62. 输出为 (16, 62, 62)
        self.num_classes = num_classes
        self.blocks = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=1),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))
        self.blocks_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.blocks_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=83232, out_features=128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.blocks_2(x)
        x = self.blocks_3(x)
        x = nn.Softmax(dim=1)(x)
        # 接下来继续执行剩余的 Conv 和 Pool 层...
        return x
