import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()

        self.feature_extractor = nn.Sequential( # 224
            nn.Conv2d(3, 96, kernel_size=11, stride=4), # (224 - 11)//4 + 1 = 54
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (54 - 3)//2 + 1 = 26
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # 26 + 4 - 5 + 1 = 26
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (26 - 3)//2 + 1 = 12
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # 12 + 2 - 3 + 1 = 12
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # 12
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (12 - 2)//2 + 1 = 6
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6,6))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(p=dropout),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))