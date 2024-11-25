import torch

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
   
class CNN(torch.nn.Module):
    def __init__(self):
        super(torch.nn, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(128*3*3, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128*3*3)
        x = torch.nn.functional.relu(self.fc1(x))
        latent = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(latent)
        return x, latent


class ResNet(torch.nn.Module):
    # def __init__(self, block, num_blocks, num_classes=10):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = torch.nn.Linear(512 * block.expansion, num_classes)
        # self.linear = torch.nn.Linear(2048, num_classes)
        # self.linear = torch.nn.Linear(25088, num_classes)

        self.activations = {}

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # Capture activations with hooks for each layer we are interested in
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))  # After the first conv layer
        self.activations['conv1'] = out.detach()

        out = self.layer1(out)
        self.activations['layer1'] = out.detach()  # After layer1 (two BasicBlocks)

        out = self.layer2(out)
        self.activations['layer2'] = out.detach()  # After layer2

        out = self.layer3(out)
        self.activations['layer3'] = out.detach()  # After layer3

        out = self.layer4(out)
        self.activations['layer4'] = out.detach()  # After layer4

        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        latent = out
        self.activations['latent'] = latent.detach()  # Latent representation after avg pooling

        out = self.linear(latent)
        self.activations['output'] = out.detach()  # Final fully connected layer

        return out, latent

class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                                     stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion * planes,
                                kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])