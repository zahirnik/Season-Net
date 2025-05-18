import torch
import torch.nn as nn
from modules import Encoder2D, Decoder2D, MaxPooling2D, OutConv2D, ConvLSTM

class UNet2DConvLSTM(nn.Module):
    """
    UNet-based spatiotemporal model with ConvLSTM temporal layers for seasonal forecast bias correction.

    Architecture:
    - UNet-style encoder-decoder for spatial features
    - ConvLSTM layers at multiple feature levels to capture temporal dynamics
    - Multi-GPU support: Encoder on cuda:0, Decoder on cuda:1
    """

    def __init__(self, in_channels, out_channels, num_filters, embd_channels, dropout, batch_size, bottelneck_size):
        super(UNet2DConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.dropout = dropout
        self.embd_channels = embd_channels
        self.batch_size = batch_size
        self.bottelneck_size = bottelneck_size

        # === Encoder (Downsampling path) ===
        self.Encoder1 = Encoder2D(in_channels, num_filters, num_filters, dropout)
        self.Pool1 = MaxPooling2D()
        self.Encoder2 = Encoder2D(num_filters, num_filters * 4, num_filters * 4, dropout)
        self.Pool2 = MaxPooling2D()
        self.Encoder5 = Encoder2D(num_filters * 4, num_filters * 16, num_filters * 16, dropout)

        # === ConvLSTM Layers ===
        self.LSTM1 = ConvLSTM(
            input_dim=num_filters,
            hidden_dim=[num_filters, num_filters],
            kernel_size=(1, 1),
            num_layers=2,
            dropout_prob=0.2,
            batch_first=True,
            bias=False,
            return_all_layers=False,
            size=1
        )

        self.LSTM2 = ConvLSTM(
            input_dim=num_filters * 4,
            hidden_dim=[num_filters * 4, num_filters * 4],
            kernel_size=(1, 1),
            num_layers=2,
            dropout_prob=0.2,
            batch_first=True,
            bias=False,
            return_all_layers=False,
            size=2
        )

        self.LSTM = ConvLSTM(
            input_dim=num_filters * 16,
            hidden_dim=[num_filters * 16],
            kernel_size=(1, 1),
            num_layers=1,
            dropout_prob=0.2,
            batch_first=True,
            bias=False,
            return_all_layers=False,
            size=4
        )

        # === Decoder (Upsampling path) ===
        self.Up2 = Decoder2D(num_filters * 16, num_filters * 4, dropout)
        self.Encoder2Up = Encoder2D(num_filters * 8, num_filters * 4, num_filters * 4, dropout)
        self.Up1 = Decoder2D(num_filters * 4, num_filters, dropout)
        self.Encoder1Up = Encoder2D(num_filters * 2, num_filters, num_filters, dropout)

        # === Final output layer ===
        self.out1 = OutConv2D(num_filters, out_channels)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W, T] - a sequence of 2D images
        Returns:
            ff: Tensor of shape [B, T, C_out, H, W]
        """

        f, b = [], []
        en1, en2 = [], []

        # === Process each time step ===
        for i in range(x.shape[-1]):
            img = x[:, :, :, :, i]  # [B, C, H, W] at time t

            # Encoder level 1
            self.Encoder1.to('cuda:0')
            feat1 = self.Encoder1(img.to('cuda:0'))
            en1.append(feat1.unsqueeze(1))  # store temporal sequence

            # Encoder level 2
            self.Pool1.to('cuda:0')
            pooled1 = self.Pool1(feat1)
            self.Encoder2.to('cuda:0')
            feat2 = self.Encoder2(pooled1)
            en2.append(feat2.unsqueeze(1))

            # Bottleneck
            self.Pool2.to('cuda:0')
            pooled2 = self.Pool2(feat2)
            self.Encoder5.to('cuda:0')
            bottleneck = self.Encoder5(pooled2)
            b.append(bottleneck.unsqueeze(1).to('cuda:1'))

        # === Stack over time ===
        stacked_1 = torch.cat(en1, dim=1).float().to('cuda:0')  # [B, T, C, H, W]
        stacked_2 = torch.cat(en2, dim=1).float().to('cuda:0')
        stacked_bottleneck = torch.cat(b, dim=1).float()  # [B, T, C, H, W]

        # === Apply ConvLSTM on each level ===
        self.LSTM1.to('cuda:0')
        self.LSTM2.to('cuda:0')
        self.LSTM.to('cuda:1')

        layer_1, _ = self.LSTM1(stacked_1)         # Shallow temporal features
        layer_2, _ = self.LSTM2(stacked_2)         # Mid-level temporal features
        layer_b, _ = self.LSTM(stacked_bottleneck) # Deep temporal features

        # === Decoder for each time step ===
        for i in range(x.shape[-1]):
            l1 = layer_1[0][:, i, :, :, :]  # [B, C, H, W]
            l2 = layer_2[0][:, i, :, :, :]
            lb = layer_b[0][:, i, :, :, :]

            self.Up2.to('cuda:1')
            up2 = self.Up2(lb.to('cuda:1'))
            conc2 = torch.cat([up2, l2.to('cuda:1')], dim=1)

            self.Encoder2Up.to('cuda:1')
            e2_up = self.Encoder2Up(conc2)

            self.Up1.to('cuda:1')
            up1 = self.Up1(e2_up)
            conc1 = torch.cat([up1, l1.to('cuda:1')], dim=1)

            self.Encoder1Up.to('cuda:1')
            e1_up = self.Encoder1Up(conc1)

            self.out1.to('cuda:1')
            out = self.out1(e1_up)  # Final output for time t

            f.append(out)

        # === Stack final outputs ===
        ff = torch.cat(f, dim=1).float()  # [B, T, C_out, H, W]
        return ff.cpu()
