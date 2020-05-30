from torch import nn
from torch.functional import F


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.l1 = nn.Linear(input_dim, 1024)
        self.lrelu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(1024)
        self.l2 = nn.Linear(1024, 256)
        self.lrelu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(256)
        self.l3 = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.lrelu1(x)
        x = self.bn1(x)
        x = self.l2(x)
        x = self.lrelu2(x)
        x = self.bn2(x)
        qvals = self.l3(x)
        return qvals


class ConvDQN(nn.Module):
    def __init__(self, h, w, output_dim):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
