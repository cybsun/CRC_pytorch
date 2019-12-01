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
train_set = datasets.ImageFolder('/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/train_twoclass_64_f1',transform=transforms.Compose([
                                                            #transforms.Resize(64),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.ColorJitter(0.5, 0.075, 0.075, 0.075),
                                                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # normalize inputs
                                                            ]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 10, shuffle = True)

test_set = datasets.ImageFolder('/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/val_twoclass_64_f1' ,transform=transforms.Compose([
                                                            #transforms.Resize(64),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.ColorJitter(0.5, 0.075, 0.075, 0.075),
                                                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # normalize inputs
                                                            ]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size = 10, shuffle = True)

classes = ('other', 'tumorbud')



#define classifier
class CNNClassifier(nn.Module):

    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=3)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(9600, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        # input is 64x64x3
        # conv1(kernel=5, filters=96) 64x64x96 -> 60x60x96

        x = F.relu(self.conv1(x))

        # conv2(kernel=3, filters=256) 60x60x256 -> 58x58x256
        # max_pool(kernel=2) 58x58x256 -> 29x29x256

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.dropout(x, training=self.training)

        # conv3(kernel=3, filters=384) 29x29x384 -> 27x27x384
        # max_pool (kernel=2) 27x27x384 -> 13x13x384

        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        
        # conv4(kernel=3, filters=384) 13x13x384 -> 11x11x384
        # max_pool (kernel=2) 11x11x384 -> 5x5x384
        
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)

        # flatten 5x5x384 = 9600
        x = x.view(-1, 9600)

        # 9600 -> 1000
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        # 1000 -> 2
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
    weights = [0.00703, 0.0333]
    class_weights = torch.cuda.FloatTensor(weights)
    # dataset API gives us pythonic batching
    for batch_id, (data, label) in enumerate(train_loader):
        #print('len(train_data)',len(data))
        data = Variable(data).to('cuda')
        target = Variable(label).to('cuda')

        # forward pass, calculate loss and backprop!
        opt.zero_grad()
        output = clf(data)
        loss = F.nll_loss(output, target, weight=class_weights)  #weight=class_weights
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

    writer = ExcelWriter('result_twoclass_64_40x_jitter_e80_more_f1_train.xlsx')
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
            test_loss += F.nll_loss(output, target).item()  #.item() or data[0]
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)  # loss function already averages over batch size
    accuracy = float(correct) / float(len(test_loader.dataset))
    acc_history.append(accuracy)
    print('\n{:d}, {:.4f}, {:.4f}, {}/{}'.format(epoch, test_loss, accuracy, correct, len(test_loader.dataset)))

    # output to excel

    e.append(epoch)
    f.append(test_loss) #.detach().cpu().numpy()
    g.append(accuracy)

    h = {'epoch': e, 'loss': f, 'accuracy': g}
    df = pd.DataFrame(h)

    writer = ExcelWriter('result_twoclass_64_40x_jitter_e80_more_f1_test.xlsx')
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
for epoch in range(0, 80):
    train(epoch)
    test(epoch)

# Save the model
torch.save(clf.state_dict(), 'crc_checkpoint_twoclass_40x_64_jitter_e80_more_f1.pth.tar')

# Load parameters back
clf_trained = CNNClassifier()
saved_model = torch.load('crc_checkpoint_twoclass_40x_64_jitter_e80_more_f1.pth.tar')
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

class_names = ['other', 'tumorbud']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #plt.imshow(cm, cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=90)
    #plt.yticks(tick_marks, classes)

    #fmt = '.2f' if normalize else 'd'
    #thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)


# Plot normalized confusion matrix

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

#plt.savefig('cm_twoclass_4x.png', bbox_inches='tight', dpi=1000)
