from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot
from numpy import random
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        x = x.view(-1, 8)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)

class diabetesDataset(Dataset):
    """classify diabetes dataset."""

    def __init__(self, dataarr, labelarr, normal=None):

        self.Xdata = dataarr
        self.Ydata = labelarr
        a = self.Xdata[:2, :]
        if normal != None:
            self.Xdata = (self.Xdata - normal[1]) / normal[0]

    def __len__(self):
        return len(self.Ydata)

    def __getitem__(self, idx):
        sample = (torch.tensor(self.Xdata[idx, :], dtype=torch.float32), torch.tensor(self.Ydata[idx], dtype=torch.float32).view(-1,))

        return sample

def test(args, model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output > 0.5
            correct += prediction.eq(target.type_as(prediction)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=11, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)
    random.seed(2018)

    # set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('run device is {}'.format(device))

    ## build dataset ##
    # load dataset
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

    #split X, Y data
    X = dataset[:, 0:8]
    y = dataset[:, 8]

    #split training, validation data
    mask = random.rand(len(X)) < 0.8
    X_train = X[mask]
    Y_train = y[mask]
    X_val = X[~mask]
    Y_val = y[~mask]

    # calculate std and mean in training set
    colsstd = np.std(X_train, axis=0)
    colsmean = np.mean(X_train, axis=0)

    #test = diabetesDataset(X_train, Y_train, normal= (colsstd, colsmean))
    #a = test[1:3]

    # build data loader
    train_loader = DataLoader(
        diabetesDataset(X_train, Y_train, normal= (colsstd, colsmean)),
        batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(
        diabetesDataset(X_val, Y_val, normal= (colsstd, colsmean)),
                        batch_size=args.batch_size)


    ## build model
    model = Net()
    model.to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    ## train model
    t1 = time.time()
    model.train()
    best_acc = 0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

                acc = test(args, model, device, val_loader)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                print('best acc : {:.2f} %\tat {} epoch'.format(best_acc, best_epoch))
    print('run time : {:.3f} sec'.format(time.time() - t1))
    '''
    model = XGBClassifier(n_estimators=200, booster='gbtree')
    print(model)
    model.fit(X_train, Y_train)
    result = model.predict(X_val)
    print('training acc :{}\t runtime : {:.3f}'.format(sum(Y_val == result) / len(Y_val), time.time()-t1))
    print('pridiction : {}'.format(result[:10]))
    print('pridiction : {}'.format(Y_val[:10]))
    print(result.shape)
    ##plot_importance(model)
    #pyplot.show()
    '''