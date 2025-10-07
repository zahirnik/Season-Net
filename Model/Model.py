import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from Modules import Encoder2D, Decoder2D, MaxPooling2D, ConvLSTM, OutConv2D

class UNetConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels=19,
        output_channels=1,
        features=[64, 128, 256],
        dropout=0.3,
    ):
        super().__init__()

        # Encoder 0
        self.enc0 = Encoder2D(input_channels, features[0], features[0], dropout)
        self.pool0 = MaxPooling2D()
        self.lstm0 = ConvLSTM(features[0], [features[0], features[0]], (3, 3), 2, 0, 1, True, False, False)

        # Encoder 1
        self.enc1 = Encoder2D(features[0], features[1], features[1], dropout)
        self.pool1 = MaxPooling2D()
        self.lstm1 = ConvLSTM(features[1], [features[1], features[1]], (3, 3), 2, 0, 2, True, False, False)

        # Bottleneck
        self.bottleneck = Encoder2D(features[1], features[2], features[2], dropout)
        self.bottleneck_lstm = ConvLSTM(features[2], [features[2], features[2]], (3, 3), 2, 0, 4, True, False, False)

        # Decoder
        self.up1 = Decoder2D(features[2], features[1], dropout)
        self.dec1_1 = Encoder2D(features[1]*2, features[1], features[1], dropout)
        self.dec1_2 = Encoder2D(features[1], features[1], features[1], dropout)
        self.dec1_lstm = ConvLSTM(features[1], [features[1], features[1]], (3, 3), 2, 0, 2, True, False, False)

        self.up0 = Decoder2D(features[1], features[0], dropout)
        self.dec0_1 = Encoder2D(features[0]*2, features[0], features[0], dropout)
        self.dec0_2 = Encoder2D(features[0], features[0], features[0], dropout)
        self.dec0_lstm = ConvLSTM(features[0], [features[0], features[0]], (3, 3), 2, 0, 1, True, False, False)

        self.out_conv = OutConv2D(features[0], output_channels, kernel_size=1, activation='prelu')

    def _checkpointed_layer(self, layer, x, B, T, H, W):
        x = x.contiguous().view(B * T, x.shape[1], H, W)
        x = cp.checkpoint(layer, x, use_reentrant=True)
        x = x.view(B, -1, T, H, W)
        return x

    def _pool_spatial_all_timesteps(self, pool, x):
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        x_pooled = pool(x_reshaped)
        H_p, W_p = x_pooled.shape[-2], x_pooled.shape[-1]
        x_pooled = x_pooled.view(B, T, C, H_p, W_p).permute(0, 2, 1, 3, 4)
        return x_pooled

    def _unet_forward(self, x):
        B, C, T, H, W = x.shape

        # --- ENCODER 0 ---
        x0 = self._checkpointed_layer(self.enc0, x, B, T, H, W)
        x0_lstm_in = x0.permute(0, 2, 1, 3, 4).contiguous()
        x0_skip, _ = self.lstm0(x0_lstm_in)
        x0_skip = x0_skip[0].permute(0, 2, 1, 3, 4).contiguous()

        # --- POOLING FOR ENCODER 1 ---
        x1 = self._pool_spatial_all_timesteps(self.pool0, x0_skip)

        # --- ENCODER 1 ---
        x1 = self._checkpointed_layer(self.enc1, x1, B, T, H//2, W//2)
        x1_lstm_in = x1.permute(0, 2, 1, 3, 4).contiguous()
        x1_skip, _ = self.lstm1(x1_lstm_in)
        x1_skip = x1_skip[0].permute(0, 2, 1, 3, 4).contiguous()

        # --- POOLING FOR BOTTLENECK ---
        x2 = self._pool_spatial_all_timesteps(self.pool1, x1_skip)

        # --- BOTTLENECK ---
        x2 = self._checkpointed_layer(self.bottleneck, x2, B, T, H//4, W//4)
        x2_lstm_in = x2.permute(0, 2, 1, 3, 4).contiguous()
        x2, _ = self.bottleneck_lstm(x2_lstm_in)
        x2 = x2[0].permute(0, 2, 1, 3, 4).contiguous()

        # --- DECODER 1 ---
        up1 = self.up1(x2.contiguous().view(B * T, -1, H//4, W//4)).view(B, -1, T, H//2, W//2)
        up1 = torch.cat([up1, x1_skip], dim=1)
        up1 = self._checkpointed_layer(self.dec1_1, up1, B, T, H//2, W//2)
        up1 = self._checkpointed_layer(self.dec1_2, up1, B, T, H//2, W//2)
        up1_lstm_in = up1.permute(0, 2, 1, 3, 4).contiguous()
        up1, _ = self.dec1_lstm(up1_lstm_in)
        up1 = up1[0].permute(0, 2, 1, 3, 4).contiguous()

        # --- DECODER 0 ---
        up0 = self.up0(up1.contiguous().view(B * T, -1, H//2, W//2)).view(B, -1, T, H, W)
        up0 = torch.cat([up0, x0_skip], dim=1)
        up0 = self._checkpointed_layer(self.dec0_1, up0, B, T, H, W)
        up0 = self._checkpointed_layer(self.dec0_2, up0, B, T, H, W)
        up0_lstm_in = up0.permute(0, 2, 1, 3, 4).contiguous()
        up0, _ = self.dec0_lstm(up0_lstm_in)
        up0 = up0[0].permute(0, 2, 1, 3, 4).contiguous()

        # --- OUTPUT CONV ---
        B, C, T, H, W = up0.shape
        out = self.out_conv(up0.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))
        out = out.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)
        out = out.squeeze(1)  # [B, T, H, W]
        return out

    def forward(self, x):
        return self._unet_forward(x)
