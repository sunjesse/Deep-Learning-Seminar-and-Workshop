import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def dataloader(args):
    transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
					    download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
					      shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
					   download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
					     shuffle=False, num_workers=2)

    return trainloader, testloader

def optimizer(net, args):
    if args.optimizer.lower() == "sgd":
	       return optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta1)
    elif args.optimizer.lower() == "adam":
	       return optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    return

def test(net, criterion, testloader, args):
    net.eval()
    with torch.no_grad():
        correct = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            outputs = net(inputs)
            pred = F.softmax(outputs, 1)
            if torch.sum(round(pred) == labels) == 1:
                correct += 1
        print(correct / len(testloader))

def train(net, criterion, optimizer, trainloader, args):
    for epoch in range(1, args.epoch+1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 0:
                print('[Epoch %02d, Minibatch %05d] Loss: %.5f' %
			    (epoch, i+1, running_loss/2000))
                running_loss = 0.0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Model related arguments
    # optimization related arguments
    parser.add_argument('--batch_size', default=4, type=int,
                        help='input batch size')
    parser.add_argument('--epoch', default=120, type=int,
                        help='epochs to train for')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float)
    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    trainloader, testloader = dataloader(args)
    net = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(net, args)
    train(net, criterion, optimizer, trainloader, args)
    test(net, criterion, testloader, args)
    print("Training completed!")
