## torch lib
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f

class ConvLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.zeros(state_size).to(input_.device),
                torch.zeros(state_size).to(input_.device)
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        return hidden, cell

class TransformNet(nn.Module):

    def __init__(self, opts, nc_in, nc_out):
        super(TransformNet, self).__init__()
        self.blocks = opts.blocks
        self.epoch = 0
        nf = opts.nf
        use_bias = (opts.norm == "IN")
        
        ## convolution layers
        self.conv1a = ConvLayer(3 + 3, nf * 1, kernel_size=7, stride=1, bias=use_bias, norm=opts.norm) ## input: P_t, O_t-1
        self.conv1b = ConvLayer(3 + 3, nf * 1, kernel_size=7, stride=1, bias=use_bias, norm=opts.norm) ## input: I_t, I_t-1
        self.conv2a = ConvLayer(nf * 1, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.conv2b = ConvLayer(nf * 1, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.conv3  = ConvLayer(nf * 4, nf * 4, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        
        # Residual blocks
        self.ResBlocks = nn.ModuleList()
        for b in range(self.blocks):
            self.ResBlocks.append(ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm))

        ## LSTM
        self.convlstm = ConvLSTM(input_size=nf * 4, hidden_size = nf * 4, kernel_size=3)
        
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(nf * 4, nf * 2, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)
        self.deconv2 = UpsampleConvLayer(nf * 4, nf * 1, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)
        self.deconv3 = ConvLayer(nf * 2, nc_out, kernel_size=7, stride=1) ## output one channel mask
        
        # Non-linearities
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, X, prev_state):
        
        Xa = X[:,:6,:,:] ## P_t, O_t-1
        Xb = X[:,6:,:,:] ## P_t, P_t-1

        E1a = self.relu(self.conv1a(Xa))
        E1b = self.relu(self.conv1b(Xb))

        E2a = self.relu(self.conv2a(E1a))
        E2b = self.relu(self.conv2b(E1b))

        E3 = self.relu(self.conv3(torch.cat((E2a, E2b), 1)))
        
        RB = E3
        for b in range(self.blocks):
            RB = self.ResBlocks[b](RB)
        
        state = self.convlstm(RB, prev_state)

        D2 = self.relu(self.deconv1(state[0]))
        C2 = torch.cat((D2, E2a), 1)
        D1 = self.relu(self.deconv2(C2))
        C1 = torch.cat((D1, E1a), 1)
        Y = self.deconv3(C1)

        Y = self.tanh(Y)
        
        return Y, state


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=None, bias=True):
        super(ConvLayer, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.norm in ["BN" or "IN"]:
            out = self.norm_layer(out)

        return out


class UpsampleConvLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, norm=None, bias=True):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):

        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)

        if self.norm in ["BN" or "IN"]:
            out = self.norm_layer(out)

        return out


class ResidualBlock(nn.Module):
    
    def __init__(self, channels, norm=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1  = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)
        self.conv2  = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)

        self.relu   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        
        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + input

        return out