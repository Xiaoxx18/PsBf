import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Adaptive aggregate block
class AFFB(nn.Module):
    def __init__(self, input_size, output_size):
        super(AFFB, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.conv1 = nn.Conv2d(input_size, output_size, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        # input shape: batch, seq_length, N
        output = input
        output = torch.mean(output, dim=[2, 3], keepdim=False)
        output = self.relu(self.linear1(output))
        output = self.sigmoid(self.linear2(output))
        output = output.unsqueeze(dim=2).unsqueeze(dim=3)
        output = output * input
        output = self.conv1(output)

        return output


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class ProjRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='LSTM', dropout=0, bidirectional=False):
        super(ProjRNN, self).__init__()

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                         bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):

        # input shape: batch, seq, dim
        output = input
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(output)
        proj_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return proj_output

# dual-path RNN
class DPRNN(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, num_layers=6, bidirectional=False):
        super(DPRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])


        for i in range(num_layers):
            self.row_rnn.append(ProjRNN(input_size, hidden_size, rnn_type, dropout, bidirectional=True))  # intra-segment RNN is always noncausal
            self.col_rnn.append(ProjRNN(input_size, hidden_size, rnn_type, dropout, bidirectional=bidirectional))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.col_norm.append(cLN(input_size, eps=1e-8))

        self.affb = AFFB(input_size*(len(self.row_rnn)+1), input_size)
            

    def forward(self, input):

        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        
        batch_size, _, dim1, dim2 = input.shape
        output = input
        skip = []
        skip.append(output)


        for i in range(len(self.row_rnn)):

            row_input = output.permute(0,3,2,1).contiguous().view(batch_size*dim2, dim1, -1)  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0,3,2,1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output
            
            col_input = output.permute(0,2,3,1).contiguous().view(batch_size*dim1, dim2, -1)  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = self.col_norm[i](col_output.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0,3,1,2).contiguous()  # B, N, dim1, dim2
            output = output + col_output

            skip.append(output)

        skip = torch.cat(skip, dim=1)
        skip = self.affb(skip)
            
        return output + skip
