import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1(nn.Module):
    def __init__(self):
        super(UNet1, self).__init__()
        self.enc1 = self.contracting_block(1, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)
        self.enc5 = self.contracting_block(512, 1024)
        self.upconv5 = self.expansive_block(1024, 512)
        self.upconv4 = self.expansive_block(512, 256)
        self.upconv3 = self.expansive_block(256, 128)
        self.upconv2 = self.expansive_block(128, 64)
        self.upconv1 = self.expansive_block(64, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        dec4 = self.upconv5(enc5)
        #print("sizes")
        #print(dec4.shape)
        #print(enc4.shape)
        dec3 = self.upconv4(dec4 + enc4)
        dec2 = self.upconv3(dec3 + enc3)
        dec1 = self.upconv2(dec2 + enc2)
        dec0 = self.upconv1(dec1 + enc1)
        # all outputs between 0 and 1
        final = torch.sigmoid(self.final_conv(dec0))
        return final

model = UNet1()