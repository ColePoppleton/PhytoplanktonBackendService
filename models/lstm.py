import torch
import torch.nn as nn
from .base import BaseModel


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(BaseModel):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            layers.append(ConvLSTMCell(cur_input_dim, self.hidden_dim, kernel_size))
        self.cell_list = nn.ModuleList(layers)
        self.final_conv = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(self, x, state=None):
        batch_size, seq_len, _, height, width = x.size()
        if state is None:
            state = self._init_hidden(batch_size, (height, width))

        cur_layer_input = x
        for i in range(self.num_layers):
            h, c = state[i]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[i](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            cur_layer_input = torch.stack(output_inner, dim=1)
            state[i] = (h, c)

        return self.final_conv(cur_layer_input[:, -1, :, :, :]), state

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append((
                torch.zeros(batch_size, self.hidden_dim, *image_size, device=self.final_conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, *image_size, device=self.final_conv.weight.device)
            ))
        return init_states