import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, Conv2d


def conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.ReLU(inplace=True)

        )


def soft_deconv(in_planes, out_planes, upsample_mode='bilinear'):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode=upsample_mode),
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True)
    )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def predict_flow(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


class PhyCM(nn.Module):
    def __init__(self, input_channels=4, output_channels=2, batch_norm=True, upsample_mode='bilinear'):
        super(PhyCM, self).__init__()

        self.batch_norm = batch_norm
        self.upsample_mode = upsample_mode
        self.input_channels = input_channels
        self.conv1 = conv(self.batch_norm, input_channels, 64, kernel_size=3, stride=2)
        self.conv2 = conv(self.batch_norm, 64, 128, kernel_size=3, stride=2)
        self.conv3 = conv(self.batch_norm, 128, 256, kernel_size=3, stride=2)
        self.conv3_1 = conv(self.batch_norm, 256, 256, kernel_size=3)
        self.conv4 = conv(self.batch_norm, 256, 512, kernel_size=3, stride=2)
        self.conv4_1 = conv(self.batch_norm, 512, 512, kernel_size=3)
        self.conv5 = conv(self.batch_norm, 512, 1024, stride=2)
        self.conv5_1 = conv(self.batch_norm, 1024, 1024)

        if upsample_mode == 'deconv':
            self.deconv4 = deconv(1024, 256)
            self.deconv3 = deconv(768, 128)
            self.deconv2 = deconv(384, 64)
            self.deconv1 = deconv(192, 32)
            self.deconv0 = deconv(96, 16)
        else:
            self.deconv4 = soft_deconv(1024, 256, upsample_mode=upsample_mode)
            self.deconv3 = soft_deconv(768, 128, upsample_mode=upsample_mode)
            self.deconv2 = soft_deconv(384, 64, upsample_mode=upsample_mode)
            self.deconv1 = soft_deconv(192, 32, upsample_mode=upsample_mode)
            self.deconv0 = soft_deconv(96, 16, upsample_mode=upsample_mode)

        self.predict_flow0 = predict_flow(16 + input_channels, output_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.02 / n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))

        out_deconv4 = self.deconv4(out_conv5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        flow0 = self.predict_flow0(concat0)

        return flow0


class DenseGridGen(nn.Module):

    def __init__(self, transpose=True):
        super(DenseGridGen, self).__init__()
        self.transpose = transpose
        self.register_buffer('grid', torch.Tensor())

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1, 2).transpose(2, 3)
        g0 = torch.linspace(-1, 1, x.size(2)
                            ).unsqueeze(0).repeat(x.size(1), 1)
        g1 = torch.linspace(-1, 1, x.size(1)
                            ).unsqueeze(1).repeat(1, x.size(2))
        grid = torch.cat([g0.unsqueeze(-1), g1.unsqueeze(-1)], -1)
        self.grid.resize_(grid.size()).copy_(grid)
        bgrid = Variable(self.grid)
        bgrid = bgrid.unsqueeze(0).expand(x.size(0), *bgrid.size())

        return bgrid - x


class BilinearWarpingScheme(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super(BilinearWarpingScheme, self).__init__()
        self.grid = DenseGridGen()
        self.padding_mode = padding_mode

    def forward(self, im, w):
        return F.grid_sample(im, self.grid(w), padding_mode=self.padding_mode, mode='bilinear')

class SST_Phy(nn.Module):
    def __init__(self, input_channels=4, output_channels=64):
        super(SST_Phy, self).__init__()
        self.PhyCM = PhyCM(input_channels=input_channels)
        self.wrap = BilinearWarpingScheme()
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=output_channels,
                      kernel_size=4,
                      stride=4),
            nn.GroupNorm(16, output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[4], x.shape[4])
        w = self.PhyCM(x)
        im = self.wrap(x[:, -1].unsqueeze(1), w)
        output = self.downconv(im)
        return output

class CrossAttention(torch.nn.Module):
    def __init__(self, batch_size=16, input_channels=64, width=16, height=16):
        super(CrossAttention, self).__init__()

        self.batch_size = batch_size
        self.input_channels = input_channels
        self.width = width
        self.height = height
        self.conv_query_st = torch.nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1)
        self.conv_key_st = torch.nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1)
        self.conv_value_st = torch.nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1)
        self.conv_query_phy = torch.nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1)
        self.conv_key_phy = torch.nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1)
        self.conv_value_phy = torch.nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1)
        self.conv33_64_16 = Sequential(
            Conv2d(in_channels=self.input_channels * 4, out_channels=self.input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm([self.input_channels, self.width, self.height])  # 输入特征图的形状为[64, 16, 16], 进行归一化
        )
        self.conv55_64_16= Sequential(
            Conv2d(in_channels=self.input_channels * 4, out_channels=self.input_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LayerNorm([self.input_channels, self.width, self.height])
        )
        self.conv77_64_16 = Sequential(
            Conv2d(in_channels=self.input_channels * 4, out_channels=self.input_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.LayerNorm([self.input_channels, self.width, self.height])
        )
        self.conv11_64_16 = Sequential(
            Conv2d(in_channels=self.input_channels * 4, out_channels=self.input_channels, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([self.input_channels, self.width, self.height])
        )



    def forward(self, st, phy):
        query_phy = self.conv_query_phy(phy)
        key_phy = self.conv_key_phy(phy)
        value_phy = self.conv_value_phy(phy)

        attention_map_phy = F.softmax(torch.matmul(query_phy, key_phy), dim=-1)
        output_phy = F.softmax(torch.matmul(attention_map_phy, value_phy), dim=-1) + phy

        query_st = self.conv_query_st(st)
        key_st = self.conv_key_st(st)
        value_st = self.conv_value_st(st)

        attention_map_st = F.softmax(torch.matmul(query_st, key_st), dim=-1)
        output_st = F.softmax(torch.matmul(attention_map_st, value_st), dim=-1) + st
        attention_map_st_phy = F.softmax(torch.matmul(query_phy, key_st), dim=-1)
        output_st_phy = F.softmax(torch.matmul(attention_map_st_phy, value_st), dim=-1)
        attention_map_phy_st = F.softmax(torch.matmul(query_st, key_phy), dim=-1)
        output_phy_st = F.softmax(torch.matmul(attention_map_phy_st, value_phy), dim=-1)
        concat = torch.cat([output_st, output_phy, output_st_phy, output_phy_st], dim=1)

        x1 = self.conv33_64_16(concat)
        x2 = self.conv55_64_16(concat)
        x3 = self.conv77_64_16(concat)

        output_mti = torch.sigmoid((x1 + x2 + x3)/3)
        output_att = self.conv11_64_16(concat)
        output_Phy_St = output_mti + output_att

        return output_Phy_St

class Attention_Cell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention_Cell, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        f = self.conv1(x)
        a = self.conv2(f)
        a = self.softmax(a.view(batch_size, 1, -1)).view(batch_size, 1, height, width)
        y = torch.mul(f, a)
        return y


class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential()
        self.F.add_module('conv1',
                          nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size,
                                    stride=(1, 1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(7, F_hidden_dim))
        self.F.add_module('conv2',
                          nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels=self.input_dim,
                                  kernel_size=(3, 3),
                                  padding=(1, 1), bias=self.bias)

    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)
        next_hidden = hidden_tilde + K * (x - hidden_tilde)
        return next_hidden

class PCMcell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PCMcell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []
        self.device = device
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)
        for j, cell in enumerate(self.cell_list):
            if j == 0:
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], self.H[j])
        return self.H, self.H

    def initHidden(self, batch_size):
        self.H = []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, H):
        self.H = H
class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        super(ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)
        for j, cell in enumerate(self.cell_list):
            if j == 0:
                self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))

        return (self.H, self.C), self.H

    def initHidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
            self.C.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride == 2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1,
                               output_padding=output_padding),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class encoder_E(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(encoder_E, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=2)
        self.c2 = dcgan_conv(nf, nf, stride=1)
        self.c3 = dcgan_conv(nf, 2 * nf, stride=2)

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_D(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(2 * nf, nf, stride=2)
        self.upc2 = dcgan_upconv(nf, nf, stride=1)
        self.upc3 = nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=(3, 3), stride=2, padding=1,
                                       output_padding=1)

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1)
        self.c2 = dcgan_conv(nf, nf, stride=1)

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2

class decoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1)
        self.upc2 = dcgan_upconv(nf, nc, stride=1)

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2

# PANN
class PANN(torch.nn.Module):
    def __init__(self, phycell, convcell, device):
        super(PANN, self).__init__()
        self.encoder_E = encoder_E() 
        self.encoder_Ephy = encoder_specific()
        self.encoder_Escm = encoder_specific()
        self.decoder_Dphy = decoder_specific()
        self.decoder_Dscm = decoder_specific()
        self.decoder_D = decoder_D()
        # cuda
        self.encoder_E = self.encoder_E.to(device)
        self.encoder_Ephy = self.encoder_Ephy.to(device)
        self.encoder_Escm = self.encoder_Escm.to(device)
        self.decoder_Dphy = self.decoder_Dphy.to(device)
        self.decoder_Dscm = self.decoder_Dscm.to(device)
        self.decoder_D = self.decoder_D.to(device)
        self.phycell = phycell.to(device)
        self.convcell = convcell.to(device)
        self.crossAtt = CrossAttention().to(device)
        self.attention = Attention_Cell(64, 64).to(device)

    def forward(self, input, first_timestep=False, decoding=False, past=None):
        input = self.encoder_E(input)
        input_conv = self.encoder_Escm(input)
        hidden2, output2 = self.convcell(input_conv, first_timestep)

        if decoding: 
            output1 = self.phycell(past[:, -4:, :, :, :])
        else:
            output1 = output2[-1]

        decoder_Dphy = self.decoder_Dphy(output1)
        decoder_Dscm = self.decoder_Dscm(output2[-1])
        fusion_result = self.crossAtt(decoder_Dphy, decoder_Dscm)
        decoder_Dphy = decoder_Dphy + fusion_result
        decoder_Dscm = decoder_Dscm + fusion_result
        decoded_both_attention = self.attention(decoder_Dphy + decoder_Dscm)
        decoder_Dphy = decoder_Dphy + decoded_both_attention
        decoder_Dscm = decoder_Dscm + decoded_both_attention
        concat = decoder_Dphy + decoder_Dscm
        output_sst = torch.sigmoid(self.decoder_D(concat))
        out_phys = torch.sigmoid(self.decoder_D(decoder_Dphy))
        out_conv = torch.sigmoid(self.decoder_D(decoder_Dscm))

        past_clone = past.clone()

        if decoding:
            for index in range(0, 9):
                past_clone[:, index, :, :, :] = past_clone[:, index + 1, :, :, :]
            past_clone[:, -1, :, :, :] = output_sst

        return out_phys, output1, output_sst, out_phys, out_conv, past_clone
    
