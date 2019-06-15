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

# define classifier
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=3)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(13824, 1024)
        self.fc2 = nn.Linear(1024, 9)

    def forward(self, x):
        # input is 32x32x3
        # conv1(kernel=2, filters=96) 32x32x96 -> 31x31x96

        x = F.relu(self.conv1(x))

        # conv2(kernel=3, filters=256) 31x31x96 -> 29x29x256
        # max_pool(kernel=2) 29x29x256 -> 14x14x256

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # conv3(kernel=3, filters=384) 14x14x256 -> 12x12x384
        # max_pool (kernel=2) 12x12x384 -> 6x6x384

        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        # flatten 6x6x384 = 24576
        x = x.view(-1, 13824)

        # 24576 -> 1024
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        # 1024 -> 9
        x = self.fc2(x)
        return F.log_softmax(x)

# create classifier and optimizer objects
clf = CNNClassifier()
opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)

loss_history = []
acc_history = []

# load in testset
test_set = datasets.ImageFolder('/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/test_5_twoclass_32',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize inputs
                                ]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

classes = ('Background', 'Epithelial', 'Lymphocytes', 'Mucin', 'RBCs', 'SmoothMuscle', 'Stroma', 'TumorBuds', 'TumorMucosa')


# Load parameters back
clf_trained = CNNClassifier()
saved_model = torch.load('crc_tumorbudclassifier_more_e40.pth.tar', map_location='cpu')
clf_trained.load_state_dict(saved_model)
clf_trained.eval()


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# dataiter = iter(test_loader)
# images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images, nrow=30))

import time

t0 = time.time()
a = []
b = []
dataiter = iter(test_loader)
for i in range(len(dataiter)):
    images, labels = next(dataiter)
    # imshow(torchvision.utils.make_grid(images, nrow=2))
    output = clf_trained(images)
    output = np.exp(output.detach())
    #output[0,1] = 0.01*output[0,1]
    #output[0,3] = 0.01*output[0,3]
    #output[0,8] = 0.01*output[0,8]
    #if output[0, 7] > 0.5:
        #output[0, 7] = 10
    output = np.log(output)
    b.append(output)
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    pred = pred.tolist()[0]
    a.append(pred)
    i += 1

t1 = time.time()
total = t1 - t0

# print('Predicted:  \n', ','.join('%11s' % classes[pred[j]] for j in range(10)))
print('time:\n', total)

# save prediction result
with open('tumorbudclassifier_pred_e40.txt', 'w') as f:
    for i in range(len(a)):
        f.write("%s\n" % a[i])
