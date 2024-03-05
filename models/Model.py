from torch import nn


class LitNet(nn.Module):
    def __init__(self, num_labels):
        super(LitNet, self).__init__()
        self.num_labels = num_labels
        self.padding = nn.ZeroPad2d(2)
        self.Conv1 = nn.Conv2d(3, 50, kernel_size=5, stride=2, padding=2, padding_mode='zeros')
        self.norm1 = nn.BatchNorm2d(50)
        self.Conv2 = nn.Conv2d(50, 100, kernel_size=5, stride=2, padding=2, padding_mode='zeros')
        self.norm2 = nn.BatchNorm2d(100)
        self.Conv3 = nn.Conv2d(100, 100, kernel_size=3, stride=2, padding=1, padding_mode='zeros')
        self.norm3 = nn.BatchNorm2d(100)
        self.Conv4 = nn.Conv2d(100, 200, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.norm4 = nn.BatchNorm2d(200)
        self.relu = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(200 * 7 * 7 * self.num_labels, 600*self.num_labels),  # 全链接
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(600*self.num_labels),
            nn.Linear(600*self.num_labels, 400),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(400),
            nn.Dropout(p=0.3),
            nn.Linear(400, self.num_labels**2),
        )

    def forward(self, x):
        batch_size, num_images, channels, height, width = x.shape
        x = x.view(batch_size*num_images, channels, height, width)
        # 3*104*104
        x = self.padding(x)

        # 50*52*52
        x = self.Conv1(x)
        x = self.relu(x)
        # batch normalization
        x = self.norm1(x)

        # 50*26*26
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # 100*13*13
        x = self.Conv2(x)
        x = self.relu(x)
        # batch normalization
        x = self.norm2(x)
        x = nn.Dropout(p=0.3)(x)

        # 100*7*7
        x = self.Conv3(x)
        x = self.relu(x)
        # batch normalization
        x = self.norm3(x)
        x = nn.Dropout(p=0.3)(x)

        # 200*7*7
        x = self.Conv4(x)
        x = self.relu(x)
        # batch normalization
        x = self.norm4(x)
        x = nn.Dropout(p=0.3)(x)

        x = x.contiguous().view(batch_size, -1)

        x = self.classifier(x)
        x = x.view(batch_size, self.num_labels, self.num_labels)
        return x

