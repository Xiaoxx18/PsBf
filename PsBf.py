import torch
import torch.nn as nn
import numpy as np
from DPRNN import *
from utils import *


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels*r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r
    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels//self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels//self.r, H, -1))
        return out


class RSU5(nn.Module):
    def __init__(self, in_channels, out_channels, width=64):
        super(RSU5, self).__init__()
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.chomp_f = Chomp_F(1)

        # input layer
        self.rebnconvin = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1, 1))
        self.rebnnormin = nn.LayerNorm(257)
        self.rebnpreluin = nn.PReLU(self.out_channels)


        self.rebnconv1 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.width, kernel_size=(2, 3), stride=(1, 2))
        self.rebnnorm1 = nn.LayerNorm(129)
        self.rebnprelu1 = nn.PReLU(self.width)



        self.rebnconv2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(2, 3), stride=(1, 2))
        self.rebnnorm2 = nn.LayerNorm(65)
        self.rebnprelu2 = nn.PReLU(self.width)



        self.rebnconv3 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(2, 3), stride=(1, 2))
        self.rebnnorm3 = nn.LayerNorm(33)
        self.rebnprelu3 = nn.PReLU(self.width)


        self.rebnconv4 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(2, 3), stride=(1, 2))
        self.rebnnorm4 = nn.LayerNorm(17)
        self.rebnprelu4 = nn.PReLU(self.width)


        self.rebnconv5 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(2, 3), stride=(1, 2))
        self.rebnnorm5 = nn.LayerNorm(9)
        self.rebnprelu5 = nn.PReLU(self.width)

        self.rebnconv6 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(2, 3), stride=(1, 2))
        self.rebnnorm6 = nn.LayerNorm(5)
        self.rebnprelu6 = nn.PReLU(self.width)


        self.rebnconv6d = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(2, 3), r=2)
        self.rebnnorm6d = nn.LayerNorm(9)
        self.rebnprelu6d = nn.PReLU(self.width)


        self.rebnconv5d = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(2, 3), r=2)
        self.rebnnorm5d = nn.LayerNorm(17)
        self.rebnprelu5d = nn.PReLU(self.width)


        self.rebnconv4d = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(2, 3), r=2)
        self.rebnnorm4d = nn.LayerNorm(33)
        self.rebnprelu4d = nn.PReLU(self.width)


        self.rebnconv3d = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(2, 3), r=2)
        self.rebnnorm3d = nn.LayerNorm(65)
        self.rebnprelu3d = nn.PReLU(self.width)


        self.rebnconv2d = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(2, 3), r=2)
        self.rebnnorm2d = nn.LayerNorm(129)
        self.rebnprelu2d = nn.PReLU(self.width)


        self.rebnconv1d = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.out_channels, kernel_size=(2, 3), r=2)
        self.rebnnorm1d = nn.LayerNorm(257)
        self.rebnprelu1d = nn.PReLU(self.out_channels)

    def forward(self, x):


        enc_list = []
        hx = self.rebnpreluin(self.rebnnormin(self.rebnconvin(x)))

        out = self.rebnprelu1(self.rebnnorm1(self.rebnconv1(self.pad(hx))))
        enc_list.append(out)

        out = self.rebnprelu2(self.rebnnorm2(self.rebnconv2(self.pad(out))))
        enc_list.append(out)

        out = self.rebnprelu3(self.rebnnorm3(self.rebnconv3(self.pad(out))))
        enc_list.append(out)

        out = self.rebnprelu4(self.rebnnorm4(self.rebnconv4(self.pad(out))))
        enc_list.append(out)

        out = self.rebnprelu5(self.rebnnorm5(self.rebnconv5(self.pad(out))))
        enc_list.append(out)

        out = self.rebnprelu6(self.rebnnorm6(self.rebnconv6(self.pad(out))))
        enc_list.append(out)

        out = torch.cat([out, enc_list[-1]], dim=1)
        out = self.rebnprelu6d(self.rebnnorm6d(self.chomp_f(self.rebnconv6d(self.pad(out)))))

        out = torch.cat([out, enc_list[-2]], dim=1)
        out = self.rebnprelu5d(self.rebnnorm5d(self.chomp_f(self.rebnconv5d(self.pad(out)))))

        out = torch.cat([out, enc_list[-3]], dim=1)
        out = self.rebnprelu4d(self.rebnnorm4d(self.chomp_f(self.rebnconv4d(self.pad(out)))))

        out = torch.cat([out, enc_list[-4]], dim=1)
        out = self.rebnprelu3d(self.rebnnorm3d(self.chomp_f(self.rebnconv3d(self.pad(out)))))

        out = torch.cat([out, enc_list[-5]], dim=1)
        out = self.rebnprelu2d(self.rebnnorm2d(self.chomp_f(self.rebnconv2d(self.pad(out)))))

        out = torch.cat([out, enc_list[-6]], dim=1)
        out = self.rebnprelu1d(self.rebnnorm1d(self.chomp_f(self.rebnconv1d(self.pad(out)))))
        return hx + out


class Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(Chomp_F, self).__init__()
        self.chomp_f = chomp_f
    def forward(self, x):
        return x[:, :, :, :-self.chomp_f]


class GRU_BF(nn.Module):
    def __init__(self, embed_dim, M, hid_node):
        super(GRU_BF, self).__init__()
        self.embed_dim = embed_dim
        self.M = M
        self.hid_node = hid_node
        # Components
        self.rnn1 = nn.GRU(input_size=embed_dim, hidden_size=hid_node, batch_first=True)
        self.rnn2 = nn.GRU(input_size=hid_node, hidden_size=hid_node, batch_first=True)
        self.w_dnn = nn.Sequential(
            nn.Linear(hid_node, hid_node),
            nn.PReLU(),
            nn.Linear(hid_node, 2*M)
        )

    def forward(self, embed_x):
        """
        formulate the bf operation
        :param embed_x: (B, C, T, F)
        :return: (B, T, F, M, 2)
        """
        # norm
        B, _, T, F = embed_x.shape
        x = embed_x.permute(0,3,2,1).contiguous()
        x = x.view(B*F, T, -1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x.view(B, F, T, -1).transpose(1, 2).contiguous()
        bf_w = self.w_dnn(x).view(B, T, F, self.M, 2)
        return bf_w


class U2SA(nn.Module):
    def __init__(self, width=64, use_cuda=1, num_spks=2):
        super(U2SA, self).__init__()

        self.in_channels = 12
        self.out_channels = 2
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.chomp_f = Chomp_F(1)
        self.width = width
        self.stft = STFT(use_cuda=use_cuda)
        self.istft = ISTFT(use_cuda=use_cuda)
        self.num_spks = num_spks

        # input layer
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))
        self.inp_norm = nn.LayerNorm(257)
        self.inp_prelu = nn.PReLU(self.width)

        self.stage1 = RSU5(64, 64, 32)
        #
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm1 = nn.LayerNorm(129)
        self.enc_prelu1 = nn.PReLU(self.width)

        self.seq_model = DPRNN('GRU', 64, 64, num_layers=4, bidirectional=False)

        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(257)
        self.dec_prelu1 = nn.PReLU(self.width)

        self.stage_out = RSU5(64, 64, 32)

        # output layer
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.in_channels, kernel_size=(1, 1))

        self.norm1 = nn.LayerNorm(self.in_channels)
        self.norm2 = nn.LayerNorm(self.in_channels)
        self.bf = GRU_BF(embed_dim=24, M=6, hid_node=128)

    def forward(self, input):
        

        output = input
        batch_size, channel, _ = output.shape
        output = output.view(batch_size*channel, -1).contiguous()
        
  
        output = self.stft(output)
        _, _, dim1, dim2 = output.shape

        inpt = output.view(batch_size, channel, _, dim1, dim2).permute(0, 3, 4, 1, 2).contiguous() # [batch_size, T, F, M, 2]

        output = output.view(batch_size, 2*channel, dim1, dim2)

        enc_list = []

        output = self.inp_prelu(self.inp_norm(self.inp_conv((output))))
        output = self.stage1(output)

        output = self.enc_prelu1(self.enc_norm1(self.enc_conv1(self.pad1(output))))
        enc_list.append(output)
        output = output.permute(0,1,3,2).contiguous()

        output = self.seq_model(output)

        output = output.permute(0,1,3,2).contiguous()

        output = torch.cat([output, enc_list[-1]], dim=1)

        output = self.dec_prelu1(self.dec_norm1(self.chomp_f(self.dec_conv1(self.pad1(output)))))
        output = self.stage_out(output)

        output = self.out_conv(output)  # [batch_size, 2*M, T, F]

        bf_w = output.view(batch_size, channel, -1, dim1, dim2).permute(0,3,4,1,2) # [batch_size, T, F, M, 2]

        bf_w_r, bf_w_i = bf_w[..., 0], bf_w[..., -1]
        esti_x_r, esti_x_i = (bf_w_r * inpt[..., 0] - bf_w_i * inpt[..., -1]), (bf_w_r * inpt[..., -1] + bf_w_i * inpt[..., 0])

        output_speech = torch.stack((esti_x_r, esti_x_i), dim=4)
        output_noise = inpt - output_speech

        output_speech = output_speech.view(batch_size, dim1, dim2, -1).contiguous()
        output_speech = self.norm1(output_speech)
        output_noise = output_noise.view(batch_size, dim1, dim2, -1).contiguous()
        output_noise = self.norm2(output_noise)

        output = torch.cat([output_speech, output_noise], dim=3)

        output = output.permute(0, 3, 1, 2)
        bf_w = self.bf(output)

        bf_w_r, bf_w_i = bf_w[..., 0], bf_w[..., -1]

        esti_x_r, esti_x_i = (bf_w_r * inpt[..., 0] - bf_w_i * inpt[..., -1]).sum(dim=-1), \
                             (bf_w_r * inpt[..., -1] + bf_w_i * inpt[..., 0]).sum(dim=-1)

        output = torch.stack((esti_x_r, esti_x_i), dim=1)
        output = self.istft(output, input)

        return output
    

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls()
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss = None, cv_loss = None):
        package = {
            'state_dict':model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch':epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

# from thop import profile
# x = torch.rand([1,6,64000])
# net = U2SA()
# print(net(x).shape)
# macs, params = profile(net, inputs=(x, ))
# print(macs/1000000000)
# print(params/1000000)
