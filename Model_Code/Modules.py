import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# === Spatial Self-Attention Layer ===
class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class Encoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        kernel_size: int = 7,
        padding: int = 3,
        activation: str = 'prelu'
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.act1 = nn.PReLU() if activation == 'prelu' else nn.ReLU()
        self.drop1 = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.act2 = nn.PReLU() if activation == 'prelu' else nn.ReLU()
        self.drop2 = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.conv3 = nn.Conv2d(middle_channels, out_channels, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.PReLU() if activation == 'prelu' else nn.ReLU()
        self.drop3 = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.attn = SpatialAttention1(kernel_size=7)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        identity = self.residual(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        out = self.drop3(self.act3(self.bn3(self.conv3(out))))
        out = self.attn(out)
        out += identity
        return out


class Decoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        activation: str = 'prelu',
        kernel_size: int = 7,
        padding: int = 3
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.act_up = nn.PReLU() if activation == 'prelu' else nn.ReLU()
        self.drop_up = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.PReLU() if activation == 'prelu' else nn.ReLU()
        self.drop1 = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.PReLU() if activation == 'prelu' else nn.ReLU()
        self.drop2 = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.attn = SpatialAttention1(kernel_size=7)

    def forward(self, x):
        identity = x
        out = self.drop_up(self.act_up(self.bn_up(self.up(x))))
        out = self.drop1(self.act1(self.bn1(self.conv1(out))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        out = self.attn(out)
        if out.shape == identity.shape:
            out += identity
        return out


def MaxPooling2D(kernel_size: int = 2, stride: int = 2, padding: int = 0):
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)


def OutConv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0, activation: str = 'prelu'):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.PReLU() if activation == 'prelu' else nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)








# ==== ConvLSTMCell with Residual ====
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
        # self.group_norm = nn.GroupNorm(num_groups=4, num_channels=4 * hidden_dim)


        self.shortcut = nn.Conv2d(self.input_dim, self.hidden_dim, 1) if self.input_dim != self.hidden_dim else nn.Identity()

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        combined_conv_norm = self.layer_norm(combined_conv)
        # combined_conv_norm = self.group_norm(combined_conv)


        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv_norm, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        h_next = h_next + self.shortcut(input_tensor)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

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
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias,
                dropout_prob=self.dropout_prob,
                size=self.size
            ))
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
