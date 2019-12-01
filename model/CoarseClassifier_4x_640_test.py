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
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=3)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(9600, 1000)
        self.fc2 = nn.Linear(1000, 3)

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
clf = CNNClassifier()
opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)

loss_history = []
acc_history = []

#load in testset
test_set = datasets.ImageFolder('/projects/academic/scottdoy/projects/CRC_pytorch_chenyu/src/data/test_WSI_TCGA_3' ,transform=transforms.Compose([
                                                            transforms.Resize(64),
                                                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # normalize inputs
                                                            ]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False)

classes = ('gland', 'other', 'tumor')

# Load parameters back
clf_trained = CNNClassifier()
saved_model = torch.load('crc_checkpoint_threeclass_4x_640_jitter_50e_f3.pth.tar', map_location='cpu')
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
    #    output[0, 2] = 0.0001
    #if output[0, 0] < 0.98:
    #    output[0, 0] = 0.0001
    #output = np.log(output)
    b.append(labels)
    pred = output.data.max(1)[1] # get the index of the max log-probability
    pred = pred.tolist()[0]
    a.append(pred)
    i += 1

t1 = time.time()
total = t1-t0



#print('Predicted:  \n', ','.join('%11s' % classes[pred[j]] for j in range(10)))
print('time:\n', total)

count = 0
#save prediction result
with open('coarseclassifier_4x_640_jitter_50e_f3_pred_TCGA_3.txt', 'w') as f:
    for i in range(len(a)):
        f.write("%s\n" % a[i])
        if a[i] == 1:
            count += 1
print('count',count)


#confusion matrix
y_true = b
y_pred = a

class_names = ['gland', 'other', 'tumor']

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
