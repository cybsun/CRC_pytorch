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


#define classifier
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=1)
        self.conv2 = nn.Conv2d(64, 96, kernel_size=5)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=2)

        #self.conv4 = nn.Conv2d(96, 192, kernel_size=3)
        #self.conv5 = nn.Conv2d(192, 256, kernel_size=3)
        #self.conv6 = nn.Conv2d(256, 384, kernel_size=4)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(4800, 1000)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        # input is 320x320x3
        # conv1(kernel=11, filters=64) 320x320x3 -> 78x78x64
        # max_pool(kernel=3) 78x78x64 -> 26x26x64

        x = F.max_pool2d(F.relu(self.conv1(x)), 3)

        # conv2(kernel=5, filters=96) 26x26x64 -> 22x22x96
        # max_pool(kernel=2) 22x22x96 -> 11x11x96

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # conv3(kernel=2, filters=192) 11x11x96 -> 10x10x192
        # max_pool(kernel=2) 10x10x192 -> 5x5x192

        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        # conv4(kernel=3, filters=192) 36x36x96 -> 33x33x192
        # max_pool(kernel=2) 33x33x192 -> 16x16x192

        #x = F.max_pool2d(F.relu(self.conv4(x)), 2)

        # conv5(kernel=3, filters=256) 16x16x192 -> 14x14x256
        # max_pool(kernel=2) 14x14x256 -> 7x7x256

        #x = F.max_pool2d(F.relu(self.conv5(x)), 2)

        # conv6(kernel=4, filters=384) 14x14x384 -> 11x11x384
        # max_pool (kernel=2) 11x11x384 -> 5x5x384

        #x = self.dropout(F.max_pool2d(F.relu(self.conv6(x)), 2))

        # flatten 6x6x256 = 9216
        x = x.view(-1, 4800)

        # 9600 -> 1000
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        # 1000 -> 3
        x = self.fc2(x)
        return F.log_softmax(x)

# create classifier and optimizer objects
clf = CNNClassifier()
opt = optim.SGD(clf.parameters(), lr=0.001, momentum=0.5)

loss_history = []
acc_history = []

#load in testset
test_set = datasets.ImageFolder('/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/test_5_320_40x' ,transform=transforms.Compose([
                                                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # normalize inputs
                                                            ]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False)

classes = ('gland', 'other', 'tumormucosa')

# Load parameters back
clf_trained = CNNClassifier()
saved_model = torch.load('crc_checkpoint_threeclass_40x_sn_3.pth.tar', map_location='cpu')
clf_trained.load_state_dict(saved_model)
clf_trained.eval()

def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

#dataiter = iter(test_loader)
#images, labels = dataiter.next()

#print images
#imshow(torchvision.utils.make_grid(images, nrow=30))

import time

t0 = time.time()
a = []
b = []
dataiter = iter(test_loader)
for i in range(len(dataiter)):
    images, labels = next(dataiter)
    #imshow(torchvision.utils.make_grid(images, nrow=2))
    output = clf_trained(images)
    b.append(output)
    pred = output.data.max(1)[1] # get the index of the max log-probability
    pred = pred.tolist()[0]
    a.append(pred)
    i += 1

t1 = time.time()
total = t1-t0


#print('Predicted:  \n', ','.join('%11s' % classes[pred[j]] for j in range(10)))
print('time:\n', total)

#save prediction result
with open('coarseclassifier_40x_pred_sn_3.txt', 'w') as f:
    for i in range(len(a)):
        f.write("%s\n" % a[i])
