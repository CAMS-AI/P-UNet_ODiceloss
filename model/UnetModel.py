import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torchsummary import summary

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)



class Grid_Unet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[ 64,128,256,512],
        # features=[32, 64, 128, 256],
    ):
        super(Grid_Unet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down Part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # UP part of UNET
        for feature in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Conv2d(feature*2,feature*2*2,1),
                nn.PixelShuffle(2)))
            # self.ups.append(
            #     nn.ConvTranspose2d(
            #         feature * 2,
            #         feature,
            #         kernel_size=2,
            #         stride=2,
            #     ))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
               # x=TF.to_pil_image(x)
                x= TF.resize(x,size=skip_connection.shape[2:])
                #x=TF.to_tensor(x)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        x=self.final_conv(x)
        return x

class T_net(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[ 64],
        # features=[32, 64, 128, 256],
    ):
        super(T_net, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        

        # Down Part of UNET
        for feature in features:
            self.downs.append(DoubleConv1(in_channels, feature))
            in_channels = feature

        # UP part of UNET
        for feature in reversed(features):
            # self.ups.append(nn.Sequential(
            #     nn.Conv2d(feature*2,feature*2*2,2,1,padding="same"),
            #     nn.PixelShuffle(2)))
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                ))
            self.ups.append(DoubleConv1(feature * 2, feature))

        self.bottleneck = DoubleConv1(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
               # x=TF.to_pil_image(x)
                x= TF.resize(x,size=skip_connection.shape[2:])
                #x=TF.to_tensor(x)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        x=self.final_conv(x)
        return x

def test():
    x=torch.randn(4,3,87,96)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # preds=model(x)
    # print(model)
    # for ix,im in model.named_parameters():
    #     print (ix,":",im.size())
    # print(preds.shape)
    # print(x.shape)

if __name__=="__main__":
    # print(torch.cuda.device_count())
    # for i in range(torch.cuda.device_count()):
    #     print(i,torch.cuda.get_device_name(i))
    test()
