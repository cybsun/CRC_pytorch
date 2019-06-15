import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import csv
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from PIL import Image
from sklearn.metrics import confusion_matrix

#load in dataset
train_set = datasets.ImageFolder('/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/train_threeclass_320_0.25_sn_3',transform=transforms.Compose([
                                                            transforms.Resize(32),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # normalize inputs
                                                            ]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100, shuffle = True)

test_set = datasets.ImageFolder('/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/test_threeclass_320_0.25_sn_3' ,transform=transforms.Compose([
                                                            transforms.Resize(32),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # normalize inputs
                                                            ]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size = 100, shuffle = True)

classes = ('gland', 'other', 'tumormucosa')

#define classifier
class CNNClassifier(nn.Module):

    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=3)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=4)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(9600, 1000)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        # input is 32x32x3
        # conv1(kernel=2, filters=96) 32x32x96 -> 31x31x96

        x = F.relu(self.conv1(x))

        # conv2(kernel=3, filters=256) 31x31x256 -> 29x29x256
        # max_pool(kernel=2) 29x29x256 -> 14x14x256

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.dropout(x, training=self.training)

        # conv3(kernel=4, filters=384) 14x14x384 -> 11x11x384
        # max_pool (kernel=2) 11x11x384 -> 5x5x384

        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        # flatten 5x5x384 = 9600
        x = x.view(-1, 9600)

        # 9600 -> 1000
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        # 1000 -> 3
        x = self.fc2(x)
        return F.log_softmax(x)

# create classifier and optimizer objects
clf = CNNClassifier().cuda() if torch.cuda.is_available() else CNNClassifier()
opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)

loss_history = []
acc_history = []
a = []
b = []
c = []


def train(epoch):
    clf.train()  # set model in training mode (need this because of dropout)
    correct = 0
    train_loss = 0
    #weights = []
    #class_weights = torch.cuda.FloatTensor(weights)
    # dataset API gives us pythonic batching
    for batch_id, (data, label) in enumerate(train_loader):
        #print('len(train_data)',len(data))
        data = Variable(data).to('cuda')
        target = Variable(label).to('cuda')

        # forward pass, calculate loss and backprop!
        opt.zero_grad()
        output = clf(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        #loss_history.append(loss.data[0])
        opt.step()
        train_loss += loss.item()

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    #train_loss = np.mean(loss_history)
    train_loss /= len(train_loader)
    train_accuracy = float(correct) / float(len(train_loader.dataset))
    print('\n{:d}, {:.4f}, {:.4f}, {}/{}'.format(epoch, train_loss, train_accuracy, correct, len(train_loader.dataset)))

    a.append(epoch)
    b.append(train_loss)
    c.append(train_accuracy)

    # output to excel

    d = {'epoch': a, 'loss': b, 'accuracy': c}
    df = pd.DataFrame(d)

    writer = ExcelWriter('result_threeclass_320_4x_train_sn_3_.xlsx')
    df.to_excel(writer, 'Sheet1', index=False)

    # create chart

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    chart = workbook.add_chart({'type': 'line'})

    chart.add_series({
        'categories': ['Sheet1', 1, 0, epoch + 1, 0],
        'values': ['Sheet1', 1, 2, epoch + 1, 2],
    })

    chart.set_x_axis({'name': 'epoch', 'position_axis': 'on_tick'})
    chart.set_y_axis({'name': 'accuracy', 'major_gridlines': {'visible': False}})

    worksheet.insert_chart('D1', chart)

    writer.save()

e = []
f = []
g = []


def test(epoch):
    clf.eval()  # set model in inference mode (need this because of dropout)
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        #print('len(test_data)',len(data))
        with torch.no_grad():
            data = Variable(data).to('cuda')
            target = Variable(target).to('cuda')

            output = clf(data)
            test_loss += F.nll_loss(output, target).data[0]
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)  # loss function already averages over batch size
    accuracy = float(correct) / float(len(test_loader.dataset))
    acc_history.append(accuracy)
    print('\n{:d}, {:.4f}, {:.4f}, {}/{}'.format(epoch, test_loss, accuracy, correct, len(test_loader.dataset)))

    # output to excel

    e.append(epoch)
    f.append(test_loss.detach().cpu().numpy())
    g.append(accuracy)

    h = {'epoch': e, 'loss': f, 'accuracy': g}
    df = pd.DataFrame(h)

    writer = ExcelWriter('result_threeclass_320_4x_test_sn_3_.xlsx')
    df.to_excel(writer, 'Sheet2', index=False)

    # create chart

    workbook = writer.book
    worksheet = writer.sheets['Sheet2']

    chart = workbook.add_chart({'type': 'line'})

    chart.add_series({
        'categories': ['Sheet2', 1, 0, epoch + 1, 0],
        'values': ['Sheet2', 1, 2, epoch + 1, 2],
    })

    chart.set_x_axis({'name': 'epoch', 'position_axis': 'on_tick'})
    chart.set_y_axis({'name': 'accuracy', 'major_gridlines': {'visible': False}})

    worksheet.insert_chart('D1', chart)

    writer.save()

#run the classifier
print('Epoch, Loss, Accuracy')
for epoch in range(0, 100):
    train(epoch)
    test(epoch)

# Save the model
torch.save(clf.state_dict(), 'crc_checkpoint_threeclass_4x_sn_3.pth.tar')

# Load parameters back
clf_trained = CNNClassifier()
saved_model = torch.load('crc_checkpoint_threeclass_4x_sn_3.pth.tar')
clf_trained.load_state_dict(saved_model)
clf_trained.eval()

#get prediction
a = []
b = []
dataiter = iter(test_loader)

for i in range(len(test_loader)):

    images, labels = next(dataiter)
    output = clf_trained(images)
    pred = output.data.max(1)[1] # get the index of the max log-probability
    pred = pred.numpy()
    a.extend(pred)
    b.extend(labels)
    i += 1


# confusion matrix

y_true = b
y_pred = a

class_names = ['mucosa', 'other']


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)


