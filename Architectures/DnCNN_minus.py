import torch
import torch.nn as nn

class DnCNN_minus(nn.Module):
    def __init__(self, in_c, out_c, ch):
        super(DnCNN_minus, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_c, in_c*ch[0], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_c*ch[0], in_c*ch[1], 5, 1, 2),
            nn.BatchNorm2d(in_c*ch[1]),
            nn.ReLU(),
            nn.Conv2d(in_c*ch[1], in_c*ch[2], 3, 1, 1),
            nn.BatchNorm2d(in_c*ch[2]),
            nn.ReLU()            
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_c*ch[2], in_c*ch[3], 5, 1, 2),
            nn.BatchNorm2d(in_c*ch[3]),
            nn.ReLU()            
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_c*ch[3], in_c*ch[2], 3, 1, 1),
            nn.BatchNorm2d(in_c*ch[2]),
            nn.ReLU(),
            nn.Conv2d(in_c*ch[2], in_c*ch[1], 5, 1, 2),
            nn.BatchNorm2d(in_c*ch[1]),
            nn.ReLU(),           
            nn.Conv2d(in_c*ch[1], out_c, 5, 1, 2))
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return torch.sigmoid(x)


# model = DnCNN_minus(1, 1, [8,32,128,512])
# inp = torch.randn(1, 1, 28, 28)
# out = model(inp)
# print(out.shape)