import torch
import torch.nn as nn

class Double_DnCNN_minus(nn.Module):
    def __init__(self, in_c, out_c, ch1, ch2):
        super(Double_DnCNN_minus, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_c, in_c*ch1[0], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_c*ch1[0], in_c*ch1[1], 5, 1, 2),
            nn.BatchNorm2d(in_c*ch1[1]),
            nn.ReLU(),
            nn.Conv2d(in_c*ch1[1], in_c*ch1[2], 3, 1, 1),
            nn.BatchNorm2d(in_c*ch1[2]),
            nn.ReLU(),
            nn.Conv2d(in_c*ch1[2], in_c*ch1[3], 5, 1, 2),
            nn.BatchNorm2d(in_c*ch1[3]),
            nn.ReLU(),
            nn.Conv2d(in_c*ch1[3], in_c*ch1[2], 3, 1, 1),
            nn.BatchNorm2d(in_c*ch1[2]),
            nn.ReLU(),
            nn.Conv2d(in_c*ch1[2], in_c*ch1[1], 5, 1, 2),
            nn.BatchNorm2d(in_c*ch1[1]),
            nn.ReLU(),           
            #nn.Conv2d(in_c*ch[1], out_c, 5, 1, 2)
        )
    
        self.b2 = nn.Sequential(
            #nn.Conv2d(in_c, in_c*ch[0], 3, 1, 1),
            #nn.ReLU(),
            nn.Conv2d(in_c*ch1[1], in_c*ch2[1], 5, 1, 2),
            nn.BatchNorm2d(in_c*ch2[1]),
            nn.ReLU(),
            nn.Conv2d(in_c*ch2[1], in_c*ch2[2], 3, 1, 1),
            nn.BatchNorm2d(in_c*ch2[2]),
            nn.ReLU(),
            nn.Conv2d(in_c*ch2[2], in_c*ch2[3], 5, 1, 2),
            nn.BatchNorm2d(in_c*ch2[3]),
            nn.ReLU(),
            nn.Conv2d(in_c*ch2[3], in_c*ch2[2], 3, 1, 1),
            nn.BatchNorm2d(in_c*ch2[2]),
            nn.ReLU(),
            nn.Conv2d(in_c*ch2[2], in_c*ch2[1], 5, 1, 2),
            nn.BatchNorm2d(in_c*ch2[1]),
            nn.ReLU(),           
            nn.Conv2d(in_c*ch2[1], out_c, 5, 1, 2)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        return torch.sigmoid(x)


# model = Double_DnCNN_minus(1, 1, [4,16,64,256], [8,32,128,512])
# inp = torch.randn(1, 1, 90, 90)
# out = model(inp)
# print(out.shape)