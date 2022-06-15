import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from util import export_model


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_chans, n_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chans),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(n_chans, n_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chans)
        )

    def forward(self, x):
        res = x
        out = self.block(x)
        out += res
        return nn.ReLU()(out)


class SmallResNet(nn.Module):
    def __init__(self, n_classes=10, n_blocks=2):
        super(SmallResNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.resblocks = nn.Sequential(*[ResBlock(n_chans=32) for _ in range(n_blocks)])

        # Global avg pooling
        # + a fully connected layer
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, n_classes)
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.resblocks(out)
        out = self.fc(out)
        return out


def train_cifar10():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    epochs = 20
    learning_rate = 0.001

    # Transforms: augment and normalize data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = SmallResNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")

    export_model(model, torch.randn(4, 3, 32, 32).to(device), "models/cifar-cnn.pt")

    # Eval on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    train_cifar10()
