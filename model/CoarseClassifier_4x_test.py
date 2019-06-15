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
clf = CNNClassifier()
opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)

loss_history = []
acc_history = []

#load in testset
test_set = datasets.ImageFolder('/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/test_5_320_40x' ,transform=transforms.Compose([
                                                            transforms.Resize(32),
                                                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # normalize inputs
                                                            ]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False)

classes = ('gland', 'other', 'tumormucosa')

# Load parameters back
clf_trained = CNNClassifier()
saved_model = torch.load('crc_checkpoint_threeclass_4x_sn_3_.pth.tar', map_location='cpu')
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
    #output = np.exp(output.detach())
    #if output[0, 2] < 0.9:
        #output[0, 2] = 0.0001
    #if output[0, 0] < 0.98:
        #output[0, 0] = 0.0001
    #output = np.log(output)
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
with open('coarseclassifier_4x_sn_3_pred.txt', 'w') as f:
    for i in range(len(a)):
        f.write("%s\n" % a[i])

#color grid
#from matplotlib import colors
#pd = np.array([a])
#pdmap = pd.reshape((60, 60))
#pdmap = np.flipud(pdmap)
#cmap = colors.ListedColormap(['yellow','gray', [1, 0, 1]])
#map = plt.pcolormesh(pdmap, cmap=cmap)
#plt.axis('equal')
#cbar = plt.colorbar()
#cbar.ax.set_ylabel('gland  other  tumormucosa')
#plt.savefig('map_threeclass.png', dpi=1000)
