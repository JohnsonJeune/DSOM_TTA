import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import sklearn.metrics as sm
# from torchstat import stat
import torch.nn.functional as F
from device import DEVICE

class CNN_UCI(nn.Module):
    def __init__(self):
        super(CNN_UCI, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )
        self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
        )
        self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
        self.layer6 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
        )
        self.classifier = nn.Linear(15360, 6)


    def forward(self, x):
        # print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.classifier(x)
        # x = nn.LayerNorm(x.size())(x.cpu())
        # x = x.cuda()
        # x = F.normalize(x.cuda())
        return x, features

class ResCNN_UCI(nn.Module):
    def __init__(self):
            super(ResCNN_UCI, self).__init__()
            self.Block1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
            )
            self.Block2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.shortcut2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(128),
            )
            self.Block3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.Dropout(0.5),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                # nn.Dropout(0.5),
                nn.ReLU(True)
            )
            self.shortcut3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(256),
                # nn.Dropout(0.5)
            )
            self.classifier = nn.Linear(15360, 6)


    def forward(self, x):
        h1 = self.Block1(x)
        r = self.shortcut1(x)
        h1 = h1 + r
        h2 = self.Block2(h1)
        r = self.shortcut2(h1)
        h2 = h2 + r
        h3 = self.Block3(h2)
        r = self.shortcut3(h2)
        h3 = h3 + r
        x = h3.view(h3.size(0), -1)
        features = x
        x = self.classifier(x)
        # x = nn.LayerNorm(x.size())(x.cpu())
        # x = x.cuda()
        x = F.normalize(x.to(DEVICE))
        return x, features


class CNN_UNIMIB(nn.Module):
    def __init__(self):
            super(CNN_UNIMIB, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
            )
            self.layer6 = nn.Sequential(
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
            )
            self.classifier = nn.Linear(8448, 17)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.classifier(x)
        x = nn.LayerNorm(x.size()).to(DEVICE)(x)
        x = x.to(DEVICE)
        # x = F.normalize(x.cuda())
        return x, features


class ResCNN_UNIMIB(nn.Module):
    def __init__(self):
            super(ResCNN_UNIMIB, self).__init__()
            self.Block1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
            )
            self.Block2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
            self.shortcut2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
            )
            self.Block3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True)
            )
            self.shortcut3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
            )
            self.classifier = nn.Linear(8448, 17)


    def forward(self, x):
        h1 = self.Block1(x)
        r = self.shortcut1(x)
        h1 = h1 + r
        h2 = self.Block2(h1)
        r = self.shortcut2(h1)
        h2 = h2 + r
        h3 = self.Block3(h2)
        r = self.shortcut3(h2)
        h3 = h3 + r
        x = h3.view(h3.size(0), -1)
        features = x
        x = self.classifier(x)
        x = nn.LayerNorm(x.size()).to(DEVICE)(x)
        x = x.to(DEVICE)
        # x = F.normalize(x.cuda())
        return x, features

" OPPORTUNITY "
class CNN_OPPORTUNITY(nn.Module):
    def __init__(self,  num_class=17):
        super(CNN_OPPORTUNITY, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.classifier = nn.Linear(1024, num_class)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
    
        out = out.view(out.size(0), -1)
        features = out

        out = self.classifier(out)
        
        return out, features

class ResCNN_OPPORTUNITY(nn.Module):
    def __init__(self, num_class=17):
        super(ResCNN_OPPORTUNITY, self).__init__()

        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(64),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(128),
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2)),
            nn.BatchNorm2d(512),
        )

        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.classifier = nn.Linear(1024, num_class)


    def forward(self, x):

        h1 = self.Block1(x)
        r = self.shortcut1(x)
        h1 = h1 + r
        h1 = self.pool1(h1)

        h2 = self.Block2(h1)
        r = self.shortcut2(h1)
        h2 = h2 + r
        h2 = self.pool2(h2)


        h3 = self.Block3(h2)
        r = self.shortcut3(h2)
        h3 = h3 + r
        out = self.pool3(h3)
    
        out = out.view(out.size(0), -1)
        features = out
        # print(out.shape)
        out = self.classifier(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out, features

import torch
import torch.nn as nn

class CNN_PAMAP2(nn.Module):
    def __init__(self):
        super(CNN_PAMAP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )



        self.classifier = nn.Linear(53760, 12)  # PAMAP2: 12 classes

    def forward(self, x):
        # x shape: [B, 1, 171, 36]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        feature = x
        x = self.classifier(x)
        x = nn.LayerNorm(x.size()[1:]).to(x.device)(x)
        return x, feature


import torch
import torch.nn as nn

class ResCNN_PAMAP2(nn.Module):
    def __init__(self):
        super(ResCNN_PAMAP2, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
        )

        self.Block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
        )

        self.Block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
        )


        self.classifier = nn.Linear(53760, 12) 
    def forward(self, x):
        # x shape: [B, 1, 171, 36]
        h1 = self.Block1(x) + self.shortcut1(x)
        h2 = self.Block2(h1) + self.shortcut2(h1)
        h3 = self.Block3(h2) + self.shortcut3(h2)

        x = h3.view(h3.size(0), -1)
        feature = x
        x = self.classifier(x)
        x = nn.LayerNorm(x.size()[1:]).to(x.device)(x)
        x= x.to(DEVICE)
        return x, feature

import torch
import torch.nn as nn


class CNN_USC(nn.Module):
    def __init__(self):
        super(CNN_USC, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 100, 6)
            x = self.layer1(dummy)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            flatten_dim = x.view(1, -1).size(1)
        #print(flatten_dim)
        self.classifier = nn.Linear(flatten_dim, 12)  # USC has 12 activities

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        feature = x
        x = self.classifier(x)
        x = nn.LayerNorm(x.size()[1:]).to(x.device)(x)
        return x, feature


import torch
import torch.nn as nn


class ResCNN_USC(nn.Module):
    def __init__(self):
        super(ResCNN_USC, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
        )

        self.Block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
        )

        self.Block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 100, 6)
            h1 = self.Block1(dummy) + self.shortcut1(dummy)
            h2 = self.Block2(h1) + self.shortcut2(h1)
            h3 = self.Block3(h2) + self.shortcut3(h2)
            flatten_dim = h3.view(1, -1).shape[1]

        self.classifier = nn.Linear(flatten_dim, 12)

    def forward(self, x):
        h1 = self.Block1(x) + self.shortcut1(x)
        h2 = self.Block2(h1) + self.shortcut2(h1)
        h3 = self.Block3(h2) + self.shortcut3(h2)
        x = h3.view(h3.size(0), -1)
        feature = x
        x = self.classifier(x)
        x = nn.LayerNorm(x.size()).to(DEVICE)(x)
        return x.to(DEVICE), feature



def CNN_choose(dataset = 'uci', res=False, return_feature = False):
    #print(res)
    if dataset == 'uci':
        
        if res == False:
            # print('1')
            model = CNN_UCI()
        else:
            model = ResCNN_UCI()
        return model

    if dataset == 'unimib':
        if res == False:
            model = CNN_UNIMIB()
        else:
            model = ResCNN_UNIMIB()
        return model

    if dataset == 'oppo':
        if res == False:
            model = CNN_OPPORTUNITY()
        else:
            model = ResCNN_OPPORTUNITY()
        return model
    if dataset == 'pamap2':
        if res == False:
            model = CNN_PAMAP2().to(DEVICE)
        else:
            model = ResCNN_PAMAP2()
        return model
    if dataset == 'usc':
        if res == False:
            model = CNN_USC().to(DEVICE)
        else:
            model = ResCNN_USC()
        return model
    else:
        return print('not exist this model')

        

        



def main():
    model = CNN_choose(dataset = 'oppo', res=True).to(DEVICE)
    input = torch.rand(3, 1, 30, 77).to(DEVICE)
    output = model(input)
    print(output.shape)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))

if __name__ == '__main__':
    main()