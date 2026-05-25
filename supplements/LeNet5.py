import torch.nn as nn

class LeNet5(nn.Module):

    def __init__(self, n_class=10):
        super().__init__()
        
        self.features = nn.Sequential(# bs, 3, 32, 32
            nn.Conv2d(1, 6, kernel_size=5), # bs, 6, 28, 28
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2), # bs, 6, 14, 14

            nn.Conv2d(6, 16, kernel_size=5), # bs, 16, 10, 10
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2), # bs, 16, 5, 5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), # bs, n
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_class),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
    
