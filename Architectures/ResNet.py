import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_c, out_c, identity_downsample=None):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_c, out_c, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        #self.conv3 = nn.Conv2d(out_c, out_c*self.expansion, 1, 1, 0)
        #self.bn3 = nn.BatchNorm2d(out_c*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        #self.dropout = nn.Dropout2d(0.2)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        #x = self.dropout(x)
        x = self.relu(x)
        #x = self.conv3(x)
        #x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x = x + identity
        return self.relu(x)

class ResNet(nn.Module): #blocks = [3,4,6,3]
    def __init__(self, Block, layers, input_img_c, output_img_c):
        super(ResNet, self).__init__()
        # Conv layer
        self.conv1 = nn.Conv2d(input_img_c, 64, 7, 1, 3)                # (m, 64, x, x)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # ResNet layers
        self.in_c = 64
        self.layer1 = self.make_layer(Block, layers[0], out_c=64)       # (m, 64, x, x)
        self.layer2 = self.make_layer(Block, layers[1], out_c=128)      # (m, 128, x, x)
        self.layer3 = self.make_layer(Block, layers[2], out_c=256)      # (m, 256, x, x)
        self.layer4 = self.make_layer(Block, layers[3], out_c=512)      # (m, 512, x, x)

        self.conv2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, output_img_c, 1, 1, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return self.conv4(x)

    def make_layer(self, Block, num_res_blocks, out_c):
        identity_downsample = None
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_c, out_c, 1, 1, 0),
                                            nn.BatchNorm2d(out_c))
        
        # Conv block
        layers.append(Block(self.in_c, out_c, identity_downsample))
        self.in_c = out_c  # 64, 128, 256, 512
        
        # Identity blocks
        for i in range(num_res_blocks-1):
            layers.append(Block(self.in_c, out_c))   
        
        return nn.Sequential(*layers)


# model = ResNet(Block, [3, 6, 4, 3], 1, 1)         # ResNet34
# model = ResNet(Block, [2, 2, 2, 2], 1, 1)         # ResNet18
# x = torch.randn(1, 1, 128, 128)
# y = model(x)
# print(y.shape)


