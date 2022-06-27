import torch.nn as nn
import torch
import torchvision.transforms.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            #nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3, inplace=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:          # features = #filters
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))  # (n-1)*s + f - 2p
            self.decoder.append(
                DoubleConv(feature*2, feature))
            #self.decoder.append(nn.Dropout2d(0.3))
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
       
       # Encoder
        for encoding in self.encoder:
            x = encoding(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck layer
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for decoding in range(0, len(self.decoder), 2):
            x = self.decoder[decoding](x)
            skip_connection = skip_connections[decoding//2]

            if x.shape != skip_connection.shape:       # Maybe because of max pooling in encoder. Due to this, size from decoder will always be smaller than encoder.
                x = F.resize(x, size=skip_connection.shape[2:])      # Don't change #channels in x

            concat_skip = torch.cat((skip_connection, x), dim=1)     # dim=1 because (m, n_c, n_H, n_W)
            x = self.decoder[decoding+1](concat_skip)

        # Final layer
        return self.final_conv(x)


# model = UNet(1, 1)     
# x = torch.randn(1, 1, 128, 128)
# y = model(x)
# print(y.shape)