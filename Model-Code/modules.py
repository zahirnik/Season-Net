# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math


import torch
import torch.nn as nn

class SpatialAttention0(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention0, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling across channels
        attention = torch.cat([avg_out, max_out], dim=1)  # Concatenate along channel dim
        attention = self.conv(attention)
        return x * self.sigmoid(attention)  # Apply attention map

def Encoder2D(in_channels: int, middle_channels: int, out_channels: int,
              dropout: float = 0.4, kernel_size: int = 3, stride: int = 1, padding: int = 1,
              activation: str = 'prelu') -> nn.Sequential:

    activation_fn = nn.PReLU() if activation == 'prelu' else nn.ReLU()

    layers = [
        nn.Conv2d(in_channels, middle_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(middle_channels),
        activation_fn,
        nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),

        nn.Conv2d(middle_channels, middle_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(middle_channels),
        activation_fn,
        nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),

        SpatialAttention0()  # Spatial Attention Block
    ]

    return nn.Sequential(*layers)


import torch.nn as nn

class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling across channels
        attention = torch.cat([avg_out, max_out], dim=1)  # Concatenate along channel dim
        attention = self.conv(attention)
        return x * self.sigmoid(attention)  # Apply attention map


def Decoder2D(in_channels: int, out_channels: int, dropout: float = 0.4,
              activation: str = 'prelu') -> nn.Sequential:

    activation_fn = nn.PReLU() if activation == 'prelu' else nn.ReLU()

    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0),
        nn.BatchNorm2d(out_channels),
        activation_fn,
        nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),

        SpatialAttention1()  # Spatial Attention Block to refine upsampled features
    ]

    return nn.Sequential(*layers)





    return nn.Sequential(*layers)

def MaxPooling2D(kernel_size: int = 2, stride: int = 2, padding: int = 0) -> nn.MaxPool2d:
    """
    Creates a max pooling layer for downsampling the feature maps.

    Parameters:
    kernel_size (int): The size of the window to take a max over. Default is 2.
    stride (int): The stride of the window. Default is 2.
    padding (int): Implicit zero padding to be added on both sides. Default is 0.

    Returns:
    nn.MaxPool2d: A max pooling layer.
    """

    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    #return  nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)


def OutConv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0,activation: str = 'prelu'):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
              nn.BatchNorm2d(out_channels),
              nn.PReLU() if activation == 'prelu' else nn.ReLU(),
    ]
    # Add Batch Normalization and Activation layers if needed
    return nn.Sequential(*layers)

class HadamardProduct(nn.Module):
    def __init__(self, shape):
        super(HadamardProduct, self).__init__()
        self.weights = nn.Parameter(torch.rand(shape)).cuda()

    def forward(self, x):
        return x*self.weights




import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================== #
#    ConvLSTM Cell         #
# ======================== #
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, dropout_prob, size):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.dropout_prob = dropout_prob
        self.size = size

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        self.layer_norm = nn.LayerNorm(
            [4 * hidden_dim, int(64 / self.size), int(64 / self.size)]
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        # Apply layer normalization
        combined_conv_norm = self.layer_norm(combined_conv)

        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv_norm, self.hidden_dim, dim=1)

        # Compute gates
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Update cell state
        c_next = f * c_cur + i * g

        # Compute next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

# ======================== #
#    ConvLSTM Model        #
# ======================== #
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, dropout_prob, size,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.dropout_prob = dropout_prob
        self.size = size

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          dropout_prob=self.dropout_prob,
                                          size=self.size))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                h = h + cur_layer_input[:, t, :, :, :]
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output = layer_output_list[-1:]
            last_state = last_state_list[-1:]

        return layer_output, last_state

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# ======================== #
# ConvLSTM without Attention
# ======================== #
class ConvLSTM_F(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, dropout_prob, size,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM_F, self).__init__()

        self.convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, dropout_prob, size,
                                 batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)

    def forward(self, input_tensor):
        layer_output, last_state = self.convlstm(input_tensor)

        # Directly return ConvLSTM output without attention
        return layer_output, last_state

