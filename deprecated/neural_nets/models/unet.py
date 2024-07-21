import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_nets.trainer import THparams

class UNet2(nn.Module):
    # UNet with adaptable depth


    def __init__(self, t_h:THparams=None):
        super().__init__()
        self.depth = t_h.depth
        self.channels = t_h.channels
        if t_h.first_kernel_size %2 ==0:
            raise ValueError("First kernel size must be uneven number")
        self.first_kernel_size = t_h.first_kernel_size

        # Create encoder layers
        self.encoder = nn.ModuleList()
        in_channels = 1
        for i in range(self.depth):
            out_channels = self.channels * (2 ** i)
            # in first layer set custom kernel size
            if i == 0:
                self.encoder.append(self.contracting_block(in_channels, out_channels,
                                                           first_kernel_size=self.first_kernel_size))
            else:
                self.encoder.append(self.contracting_block(in_channels, out_channels))
            in_channels = out_channels

        # Create decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            in_channels = self.channels * (2 ** i)
            out_channels = self.channels * (2 ** (i - 1))
            self.decoder.append(self.expansive_block(in_channels, out_channels))

        # Final expansive block and output layer
        self.final_block = self.expansive_block(self.channels, self.channels)
        self.final_conv = nn.Conv2d(self.channels, 1, kernel_size=1)

    def contracting_block(self, in_channels, out_channels, first_kernel_size=3):

        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=first_kernel_size, padding=int((first_kernel_size-1)/2)),
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
        # Encoder
        enc_outputs = []
        for encode in self.encoder:
            x = encode(x)
            enc_outputs.append(x)

        # Bottleneck (last encoder output without pooling)
        bottleneck = enc_outputs[-1]

        # Decoder
        for i, decode in enumerate(self.decoder):
            if i == 0:
                bottleneck = decode(bottleneck)
            else:
                bottleneck = decode(bottleneck + enc_outputs[-(i + 1)])

        # Final layer
        dec0 = self.final_block(bottleneck)
        final = torch.sigmoid(self.final_conv(dec0))
        return final