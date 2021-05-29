import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler


# Class components
class DownSample(nn.Module):
    """
    SingleConv1D module + MaxPool
    The output dimension = input dimension // down_ratio
    """

    def __init__(self, input_chan_num, output_chan_num, down_ratio, kernel_size=9, norm_layer=nn.InstanceNorm1d,
                 leaky_slope=0.2, dropout_p=0):
        """
        Construct a downsampling block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor
            output_chan_num (int) -- the number of channels of the output tensor
            down_ratio (int)      -- the kernel size and stride of the MaxPool1d layer
            kernel_size (int)     -- the kernel size of the DoubleConv1D block
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer
        """
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            SingleConv1D(input_chan_num, output_chan_num, kernel_size, norm_layer, leaky_slope),
            nn.MaxPool1d(down_ratio),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.down_sample(x)


class UpSample(nn.Module):
    """
    ConvTranspose1d + SingleConv1D
    The output dimension = input dimension * ratio
    """

    def __init__(self, input_chan_num, output_chan_num, up_ratio, kernel_size=9, norm_layer=nn.BatchNorm1d,
                 leaky_slope=0.2, dropout_p=0, attention=True):
        """
        Construct a upsampling block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor (the tensor from get from the last layer, not the tensor from the skip-connection mechanism)
            output_chan_num (int) -- the number of channels of the output tensor
            up_ratio (int)        -- the kernel size and stride of the ConvTranspose1d layer
            kernel_size (int)     -- the kernel size of the DoubleConv1D block
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer
            activation (bool)     -- need activation or not
        """
        super(UpSample, self).__init__()
        self.attention = attention
        self.up_sample = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.ConvTranspose1d(input_chan_num, input_chan_num, kernel_size=up_ratio, stride=up_ratio),
            SingleConv1D(input_chan_num, output_chan_num, kernel_size, norm_layer, leaky_slope)
        )
        self.up_sample_no_relu = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.ConvTranspose1d(input_chan_num, input_chan_num, kernel_size=up_ratio, stride=up_ratio),
            nn.Conv1d(input_chan_num, output_chan_num, kernel_size=kernel_size, padding=kernel_size // 2)
        )

    def forward(self, x):
        if self.attention:
            return self.up_sample(x)
        else:
            return self.up_sample_no_relu(x)


class OutputConv(nn.Module):
    """
    Output convolution layer
    """

    def __init__(self, input_chan_num, output_chan_num):
        """
        Construct the output convolution layer
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor
            output_chan_num (int) -- the number of channels of the output omics data
        """
        super(OutputConv, self).__init__()
        self.output_conv = nn.Sequential(
            nn.Conv1d(input_chan_num, output_chan_num, kernel_size=1),
        )

    def forward(self, x):
        return self.output_conv(x)


class SingleConv1D(nn.Module):
    """
    Convolution1D => Norm1D => LeakyReLU
    The omics data dimension keep the same during this process
    """

    def __init__(self, input_chan_num, output_chan_num, kernel_size=9, norm_layer=nn.InstanceNorm1d, leaky_slope=0.2):
        """
        Construct a single convolution block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor
            output_chan_num (int) -- the number of channels of the output tensor
            kernel_size (int)     -- the kernel size of the convolution layer
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
        """
        super(SingleConv1D, self).__init__()

        # Only if the norm method is instance norm we use bias for the corresponding conv layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        self.single_conv_1d = nn.Sequential(
            nn.Conv1d(input_chan_num, output_chan_num, kernel_size=kernel_size, padding=kernel_size // 2,
                      bias=use_bias),
            norm_layer(output_chan_num),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        )

    def forward(self, x):
        return self.single_conv_1d(x)


class FCBlock(nn.Module):
    """
    Linear => Norm1D => LeakyReLU
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, activation=True, normalization=True, activation_name='LeakyReLU'):
        """
        Construct a fully-connected block
        Parameters:
            input_dim (int)         -- the dimension of the input tensor
            output_dim (int)        -- the dimension of the output tensor
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            activation (bool)       -- need activation or not
            normalization (bool)    -- need normalization or not
            activation_name (str)   -- name of the activation function used in the FC block
        """
        super(FCBlock, self).__init__()
        # Linear
        self.fc_block = [nn.Linear(input_dim, output_dim)]
        # Norm
        if normalization:
            # FC block doesn't support InstanceNorm1d
            if isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm1d:
                norm_layer = nn.BatchNorm1d
            self.fc_block.append(norm_layer(output_dim))
        # Dropout
        if 0 < dropout_p <= 1:
            self.fc_block.append(nn.Dropout(p=dropout_p))
        # LeakyReLU
        if activation:
            if activation_name.lower() == 'leakyrelu':
                self.fc_block.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            elif activation_name.lower() == 'tanh':
                self.fc_block.append(nn.Tanh())
            else:
                raise NotImplementedError('Activation function [%s] is not implemented' % activation_name)

        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        y = self.fc_block(x)
        return y


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, dim):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.dim = dim

    def forward(self, x):
        return x.view(x.size(0), self.channel, self.dim)


class Identity(nn.Module):
    def forward(self, x):
        return x


# Class for VAE
# ConvVae
class ConvVaeABC(nn.Module):
    """
        Defines a one dimensional convolution variational autoencoder for multi-omics dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9, leaky_slope=0.2,
                 dropout_p=0, ratio_1B=16, ratio_2B=16, ratio_1A=4, ratio_2A=4, ratio_1C=2, ratio_2C=2, ratio_3=16,
                 latent_dim=256):
        """
            Construct a one dimensional convolution variational autoencoder for multi-omics dataset
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(ConvVaeABC, self).__init__()

        A_dim = omics_dims[0]
        B_dim = omics_dims[1]
        C_dim = omics_dims[2]

        hidden_dim_1 = (B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A + C_dim // ratio_1C // ratio_2C) // ratio_3 * filter_num * 4
        hidden_dim_2 = (hidden_dim_1 // (filter_num * 4) + 1) * filter_num * 4
        self.narrow_B = hidden_dim_2 // (4 * filter_num) * ratio_3 * (B_dim // ratio_1B // ratio_2B) // (
                B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A + C_dim // ratio_1C // ratio_2C)
        self.narrow_A = hidden_dim_2 // (4 * filter_num) * ratio_3 * (A_dim // ratio_1A // ratio_2A) // (
                B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A + C_dim // ratio_1C // ratio_2C)
        self.narrow_C = hidden_dim_2 // (4 * filter_num) * ratio_3 * (C_dim // ratio_1C // ratio_2C) // (
                B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A + C_dim // ratio_1C // ratio_2C)
        self.B_dim = B_dim
        self.A_dim = A_dim
        self.C_dim = C_dim

        # ENCODER
        # B 1 -> 8
        self.down_sample_1B = DownSample(1, filter_num, down_ratio=ratio_1B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 8 -> 16
        self.down_sample_2B = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # A 1 -> 8
        self.down_sample_1A = DownSample(1, filter_num, down_ratio=ratio_1A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A 8 -> 16
        self.down_sample_2A = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # C 1 -> 8
        self.down_sample_1C = DownSample(1, filter_num, down_ratio=ratio_1C, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # C 8 -> 16
        self.down_sample_2C = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2C, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)

        # 16 -> 32
        self.down_sample_3 = DownSample(filter_num * 2, filter_num * 4, down_ratio=ratio_3, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # Flatten
        self.flatten = Flatten()
        # FC to mean
        self.fc_mean = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                               dropout_p=0, activation=False, normalization=False)
        # FC to log_var
        self.fc_log_var = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                  dropout_p=0, activation=False, normalization=False)

        # DECODER
        # FC from z
        self.fc_z = FCBlock(latent_dim, hidden_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=True)
        # Unflatten
        self.unflatten = Unflatten(filter_num * 4, hidden_dim_2 // (4 * filter_num))
        # 32 -> 16
        self.up_sample_1 = UpSample(filter_num * 4, filter_num * 2, up_ratio=ratio_3, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 16 -> 8
        self.up_sample_2B = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # B 8 -> 1
        self.up_sample_3B = UpSample(filter_num, filter_num, up_ratio=ratio_1B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B Output
        self.output_conv_B = OutputConv(filter_num, 1)

        # A 16 -> 8
        self.up_sample_2A = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # A 8 -> 1
        self.up_sample_3A = UpSample(filter_num, filter_num, up_ratio=ratio_1A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A Output
        self.output_conv_A = OutputConv(filter_num, 1)

        # C 16 -> 8
        self.up_sample_2C = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2C, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # C 8 -> 1
        self.up_sample_3C = UpSample(filter_num, filter_num, up_ratio=ratio_1C, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # C Output
        self.output_conv_C = OutputConv(filter_num, 1)

    def encode(self, x):
        level_2_B = self.down_sample_1B(x[1])
        level_2_A = self.down_sample_1A(x[0])
        level_2_C = self.down_sample_1C(x[2])

        level_3_B = self.down_sample_2B(level_2_B)
        level_3_A = self.down_sample_2A(level_2_A)
        level_3_C = self.down_sample_2C(level_2_C)

        level_3 = torch.cat((level_3_B, level_3_A, level_3_C), 2)

        level_4 = self.down_sample_3(level_3)
        level_4_flatten = self.flatten(level_4)

        latent_mean = self.fc_mean(level_4_flatten)
        latent_log_var = self.fc_log_var(level_4_flatten)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.fc_z(z)
        level_1_unflatten = self.unflatten(level_1)

        level_2 = self.up_sample_1(level_1_unflatten)
        level_2_B = level_2.narrow(2, 0, self.narrow_B)
        level_2_A = level_2.narrow(2, self.narrow_B, self.narrow_A)
        level_2_C = level_2.narrow(2, self.narrow_B+self.narrow_A, self.narrow_C)

        level_3_B = self.up_sample_2B(level_2_B)
        level_3_A = self.up_sample_2A(level_2_A)
        level_3_C = self.up_sample_2C(level_2_C)

        level_4_B = self.up_sample_3B(level_3_B)
        level_4_A = self.up_sample_3A(level_3_A)
        level_4_C = self.up_sample_3C(level_3_C)

        output_B = self.output_conv_B(level_4_B)
        output_A = self.output_conv_A(level_4_A)
        output_C = self.output_conv_C(level_4_C)

        recon_B = output_B[:, :, 0:self.B_dim]
        recon_A = output_A[:, :, 0:self.A_dim]
        recon_C = output_C[:, :, 0:self.C_dim]

        return [recon_A, recon_B, recon_C]

    def get_last_encode_layer(self):
        return self.fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class ConvVaeAB(nn.Module):
    """
        Defines a one dimensional convolution variational autoencoder for multi-omics dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9, leaky_slope=0.2,
                 dropout_p=0, ratio_1B=16, ratio_2B=16, ratio_1A=4, ratio_2A=4, ratio_3=16, latent_dim=256):
        """
            Construct a one dimensional convolution variational autoencoder for multi-omics dataset
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(ConvVaeAB, self).__init__()

        A_dim = omics_dims[0]
        B_dim = omics_dims[1]

        hidden_dim_1 = (B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A) // ratio_3 * filter_num * 4
        hidden_dim_2 = (hidden_dim_1 // (filter_num * 4) + 2) * filter_num * 4
        self.narrow_B = hidden_dim_2 // (4 * filter_num) * ratio_3 * (B_dim // ratio_1B // ratio_2B) // (
                B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A)
        self.narrow_A = hidden_dim_2 // (4 * filter_num) * ratio_3 - self.narrow_B
        self.B_dim = B_dim
        self.A_dim = A_dim

        # ENCODER
        # B 1 -> 8
        self.down_sample_1B = DownSample(1, filter_num, down_ratio=ratio_1B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 8 -> 16
        self.down_sample_2B = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # A 1 -> 8
        self.down_sample_1A = DownSample(1, filter_num, down_ratio=ratio_1A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A 8 -> 16
        self.down_sample_2A = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 16 -> 32
        self.down_sample_3 = DownSample(filter_num * 2, filter_num * 4, down_ratio=ratio_3, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # Flatten
        self.flatten = Flatten()
        # FC to mean
        self.fc_mean = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                               dropout_p=0, activation=False, normalization=False)
        # FC to log_var
        self.fc_log_var = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                  dropout_p=0, activation=False, normalization=False)

        # DECODER
        # FC from z
        self.fc_z = FCBlock(latent_dim, hidden_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=True)
        # Unflatten
        self.unflatten = Unflatten(filter_num * 4, hidden_dim_2 // (4 * filter_num))
        # 32 -> 16
        self.up_sample_1 = UpSample(filter_num * 4, filter_num * 2, up_ratio=ratio_3, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 16 -> 8
        self.up_sample_2B = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # B 8 -> 1
        self.up_sample_3B = UpSample(filter_num, filter_num, up_ratio=ratio_1B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B Output
        self.output_conv_B = OutputConv(filter_num, 1)

        # A 16 -> 8
        self.up_sample_2A = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # A 8 -> 1
        self.up_sample_3A = UpSample(filter_num, filter_num, up_ratio=ratio_1A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A Output
        self.output_conv_A = OutputConv(filter_num, 1)

    def encode(self, x):
        level_2_B = self.down_sample_1B(x[1])
        level_2_A = self.down_sample_1A(x[0])
        level_3_B = self.down_sample_2B(level_2_B)
        level_3_A = self.down_sample_2A(level_2_A)
        level_3 = torch.cat((level_3_B, level_3_A), 2)

        level_4 = self.down_sample_3(level_3)
        level_4_flatten = self.flatten(level_4)

        latent_mean = self.fc_mean(level_4_flatten)
        latent_log_var = self.fc_log_var(level_4_flatten)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.fc_z(z)
        level_1_unflatten = self.unflatten(level_1)

        level_2 = self.up_sample_1(level_1_unflatten)
        level_2_B = level_2.narrow(2, 0, self.narrow_B)
        level_2_A = level_2.narrow(2, self.narrow_B, self.narrow_A)

        level_3_B = self.up_sample_2B(level_2_B)
        level_3_A = self.up_sample_2A(level_2_A)

        level_4_B = self.up_sample_3B(level_3_B)
        level_4_A = self.up_sample_3A(level_3_A)

        output_B = self.output_conv_B(level_4_B)
        output_A = self.output_conv_A(level_4_A)

        recon_B = output_B[:, :, 0:self.B_dim]
        recon_A = output_A[:, :, 0:self.A_dim]

        return [recon_A, recon_B]

    def get_last_encode_layer(self):
        return self.fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class ConvVaeB(nn.Module):
    """
        Defines a one dimensional convolution variational autoencoder for DNA methylation dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9, leaky_slope=0.2,
                 dropout_p=0, ratio_1B=16, ratio_2B=16, ratio_3=16, latent_dim=256):
        """
            Construct a one dimensional convolution variational autoencoder for DNA methylation dataset
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(ConvVaeB, self).__init__()

        B_dim = omics_dims[1]

        hidden_dim_1 = B_dim // ratio_1B // ratio_2B // ratio_3 * filter_num * 4
        hidden_dim_2 = (hidden_dim_1 // (filter_num * 4) + 1) * filter_num * 4

        self.B_dim = B_dim

        # ENCODER
        # B 1 -> 8
        self.down_sample_1B = DownSample(1, filter_num, down_ratio=ratio_1B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 8 -> 16
        self.down_sample_2B = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 16 -> 32
        self.down_sample_3 = DownSample(filter_num * 2, filter_num * 4, down_ratio=ratio_3, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # Flatten
        self.flatten = Flatten()
        # FC to mean
        self.fc_mean = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                               dropout_p=0, activation=False, normalization=False)
        # FC to log_var
        self.fc_log_var = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                  dropout_p=0, activation=False, normalization=False)

        # DECODER
        # FC from z
        self.fc_z = FCBlock(latent_dim, hidden_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=True)
        # Unflatten
        self.unflatten = Unflatten(filter_num * 4, hidden_dim_2 // (4 * filter_num))
        # 32 -> 16
        self.up_sample_1 = UpSample(filter_num * 4, filter_num * 2, up_ratio=ratio_3, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 16 -> 8
        self.up_sample_2B = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # B 8 -> 1
        self.up_sample_3B = UpSample(filter_num, filter_num, up_ratio=ratio_1B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B Output
        self.output_conv_B = OutputConv(filter_num, 1)

    def encode(self, x):
        level_2_B = self.down_sample_1B(x[1])
        level_3_B = self.down_sample_2B(level_2_B)

        level_4 = self.down_sample_3(level_3_B)
        level_4_flatten = self.flatten(level_4)

        latent_mean = self.fc_mean(level_4_flatten)
        latent_log_var = self.fc_log_var(level_4_flatten)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.fc_z(z)
        level_1_unflatten = self.unflatten(level_1)
        level_2 = self.up_sample_1(level_1_unflatten)
        level_3_B = self.up_sample_2B(level_2)
        level_4_B = self.up_sample_3B(level_3_B)
        output_B = self.output_conv_B(level_4_B)
        recon_B = output_B[:, :, 0:self.B_dim]

        return [None, recon_B]

    def get_last_encode_layer(self):
        return self.fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class ConvVaeA(nn.Module):
    """
        Defines a one dimensional convolution variational autoencoder for gene expression dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9, leaky_slope=0.2,
                 dropout_p=0, ratio_1A=4, ratio_2A=4, ratio_3=16, latent_dim=256):
        """
            Construct a one dimensional convolution variational autoencoder for multi-omics dataset
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(ConvVaeA, self).__init__()

        A_dim = omics_dims[0]

        hidden_dim_1 = A_dim // ratio_1A // ratio_2A // ratio_3 * filter_num * 4
        hidden_dim_2 = (hidden_dim_1 // (filter_num * 4) + 1) * filter_num * 4
        self.A_dim = A_dim

        # ENCODER
        # A 1 -> 8
        self.down_sample_1A = DownSample(1, filter_num, down_ratio=ratio_1A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A 8 -> 16
        self.down_sample_2A = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 16 -> 32
        self.down_sample_3 = DownSample(filter_num * 2, filter_num * 4, down_ratio=ratio_3, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # Flatten
        self.flatten = Flatten()
        # FC to mean
        self.fc_mean = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                               dropout_p=0, activation=False, normalization=False)
        # FC to log_var
        self.fc_log_var = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                  dropout_p=0, activation=False, normalization=False)

        # DECODER
        # FC from z
        self.fc_z = FCBlock(latent_dim, hidden_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=True)
        # Unflatten
        self.unflatten = Unflatten(filter_num * 4, hidden_dim_2 // (4 * filter_num))
        # 32 -> 16
        self.up_sample_1 = UpSample(filter_num * 4, filter_num * 2, up_ratio=ratio_3, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)

        # A 16 -> 8
        self.up_sample_2A = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # A 8 -> 1
        self.up_sample_3A = UpSample(filter_num, filter_num, up_ratio=ratio_1A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A Output
        self.output_conv_A = OutputConv(filter_num, 1)

    def encode(self, x):
        level_2_A = self.down_sample_1A(x[0])
        level_3_A = self.down_sample_2A(level_2_A)
        level_4 = self.down_sample_3(level_3_A)
        level_4_flatten = self.flatten(level_4)

        latent_mean = self.fc_mean(level_4_flatten)
        latent_log_var = self.fc_log_var(level_4_flatten)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.fc_z(z)
        level_1_unflatten = self.unflatten(level_1)

        level_2 = self.up_sample_1(level_1_unflatten)

        level_3_A = self.up_sample_2A(level_2)

        level_4_A = self.up_sample_3A(level_3_A)

        output_A = self.output_conv_A(level_4_A)

        recon_A = output_A[:, :, 0:self.A_dim]

        return [recon_A]

    def get_last_encode_layer(self):
        return self.fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class ConvVaeC(nn.Module):
    """
        Defines a one dimensional convolution variational autoencoder for miRNA expression dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9, leaky_slope=0.2,
                 dropout_p=0, ratio_1C=2, ratio_2C=2, ratio_3=16, latent_dim=256):
        """
            Construct a one dimensional convolution variational autoencoder for multi-omics dataset
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(ConvVaeC, self).__init__()

        C_dim = omics_dims[2]

        hidden_dim_1 = (C_dim // ratio_1C // ratio_2C) // ratio_3 * filter_num * 4
        hidden_dim_2 = (hidden_dim_1 // (filter_num * 4) + 1) * filter_num * 4
        self.C_dim = C_dim

        # ENCODER
        # C 1 -> 8
        self.down_sample_1C = DownSample(1, filter_num, down_ratio=ratio_1C, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # C 8 -> 16
        self.down_sample_2C = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2C, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)

        # 16 -> 32
        self.down_sample_3 = DownSample(filter_num * 2, filter_num * 4, down_ratio=ratio_3, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # Flatten
        self.flatten = Flatten()
        # FC to mean
        self.fc_mean = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                               dropout_p=0, activation=False, normalization=False)
        # FC to log_var
        self.fc_log_var = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                  dropout_p=0, activation=False, normalization=False)

        # DECODER
        # FC from z
        self.fc_z = FCBlock(latent_dim, hidden_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=True)
        # Unflatten
        self.unflatten = Unflatten(filter_num * 4, hidden_dim_2 // (4 * filter_num))
        # 32 -> 16
        self.up_sample_1 = UpSample(filter_num * 4, filter_num * 2, up_ratio=ratio_3, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)

        # C 16 -> 8
        self.up_sample_2C = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2C, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # C 8 -> 1
        self.up_sample_3C = UpSample(filter_num, filter_num, up_ratio=ratio_1C, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # C Output
        self.output_conv_C = OutputConv(filter_num, 1)

    def encode(self, x):
        level_2_C = self.down_sample_1C(x[2])

        level_3_C = self.down_sample_2C(level_2_C)

        level_4 = self.down_sample_3(level_3_C)
        level_4_flatten = self.flatten(level_4)

        latent_mean = self.fc_mean(level_4_flatten)
        latent_log_var = self.fc_log_var(level_4_flatten)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.fc_z(z)
        level_1_unflatten = self.unflatten(level_1)

        level_2 = self.up_sample_1(level_1_unflatten)

        level_3_C = self.up_sample_2C(level_2)

        level_4_C = self.up_sample_3C(level_3_C)

        output_C = self.output_conv_C(level_4_C)

        recon_C = output_C[:, :, 0:self.C_dim]

        return [None, None, recon_C]

    def get_last_encode_layer(self):
        return self.fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


# FcSepVae
class FcSepVaeABC(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_1A=2048, dim_2A=1024, dim_1C=1024, dim_2C=1024, dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcSepVaeABC, self).__init__()

        self.A_dim = omics_dims[0]
        self.B_dim_list = omics_dims[1]
        self.C_dim = omics_dims[2]
        self.dim_1B = dim_1B
        self.dim_2B = dim_2B
        self.dim_2A = dim_2A
        self.dim_2C = dim_2C

        # ENCODER
        # Layer 1
        self.encode_fc_1B_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1B_list.append(
                FCBlock(self.B_dim_list[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B*23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2C = FCBlock(dim_1C, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A+dim_2C, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B+dim_2A+dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 4
        self.decode_fc_4B_list = nn.ModuleList()
        for i in range(0, 23):
            self.decode_fc_4B_list.append(
                FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                        activation=False, normalization=False))
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B_list = []
        for i in range(0, 23):
            level_2_B_list.append(self.encode_fc_1B_list[i](x[1][i]))
        level_2_B = torch.cat(level_2_B_list, 1)
        level_2_A = self.encode_fc_1A(x[0])
        level_2_C = self.encode_fc_1C(x[2])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3_C = self.encode_fc_2C(level_2_C)
        level_3 = torch.cat((level_3_B, level_3_A, level_3_C), 1)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)
        level_2_C = level_2.narrow(1, self.dim_2B+self.dim_2A, self.dim_2C)

        level_3_B = self.decode_fc_3B(level_2_B)
        level_3_B_list = []
        for i in range(0, 23):
            level_3_B_list.append(level_3_B.narrow(1, self.dim_1B*i, self.dim_1B))
        level_3_A = self.decode_fc_3A(level_2_A)
        level_3_C = self.decode_fc_3C(level_2_C)

        recon_B_list = []
        for i in range(0, 23):
            recon_B_list.append(self.decode_fc_4B_list[i](level_3_B_list[i]))
        recon_A = self.decode_fc_4A(level_3_A)
        recon_C = self.decode_fc_4C(level_3_C)

        return [recon_A, recon_B_list, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcSepVaeAB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_1A=2048, dim_2A=1024, dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcSepVaeAB, self).__init__()

        self.A_dim = omics_dims[0]
        self.B_dim_list = omics_dims[1]
        self.dim_1B = dim_1B
        self.dim_2B = dim_2B
        self.dim_2A = dim_2A

        # ENCODER
        # Layer 1
        self.encode_fc_1B_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1B_list.append(
                FCBlock(self.B_dim_list[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B*23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B+dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 4
        self.decode_fc_4B_list = nn.ModuleList()
        for i in range(0, 23):
            self.decode_fc_4B_list.append(
                FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                        activation=False, normalization=False))
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B_list = []
        for i in range(0, 23):
            level_2_B_list.append(self.encode_fc_1B_list[i](x[1][i]))
        level_2_B = torch.cat(level_2_B_list, 1)
        level_2_A = self.encode_fc_1A(x[0])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3 = torch.cat((level_3_B, level_3_A), 1)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)

        level_3_B = self.decode_fc_3B(level_2_B)
        level_3_B_list = []
        for i in range(0, 23):
            level_3_B_list.append(level_3_B.narrow(1, self.dim_1B*i, self.dim_1B))
        level_3_A = self.decode_fc_3A(level_2_A)

        recon_B_list = []
        for i in range(0, 23):
            recon_B_list.append(self.decode_fc_4B_list[i](level_3_B_list[i]))
        recon_A = self.decode_fc_4A(level_3_A)

        return [recon_A, recon_B_list]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcSepVaeB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for DNA methylation dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcSepVaeB, self).__init__()

        self.B_dim_list = omics_dims[1]
        self.dim_1B = dim_1B

        # ENCODER
        # Layer 1
        self.encode_fc_1B_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1B_list.append(
                FCBlock(self.B_dim_list[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B*23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 4
        self.decode_fc_4B_list = nn.ModuleList()
        for i in range(0, 23):
            self.decode_fc_4B_list.append(
                FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                        activation=False, normalization=False))

    def encode(self, x):
        level_2_B_list = []
        for i in range(0, 23):
            level_2_B_list.append(self.encode_fc_1B_list[i](x[1][i]))
        level_2_B = torch.cat(level_2_B_list, 1)

        level_3_B = self.encode_fc_2B(level_2_B)

        level_4 = self.encode_fc_3(level_3_B)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)

        level_3_B = self.decode_fc_3B(level_2)
        level_3_B_list = []
        for i in range(0, 23):
            level_3_B_list.append(level_3_B.narrow(1, self.dim_1B*i, self.dim_1B))

        recon_B_list = []
        for i in range(0, 23):
            recon_B_list.append(self.decode_fc_4B_list[i](level_3_B_list[i]))

        return [None, recon_B_list]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


# FcVae
class FcVaeABC(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input not separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=384, dim_2B=256,
                 dim_1A=384, dim_2A=256, dim_1C=384, dim_2C=256, dim_3=256, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeABC, self).__init__()

        self.A_dim = omics_dims[0]
        self.B_dim = omics_dims[1]
        self.C_dim = omics_dims[2]
        self.dim_1B = dim_1B
        self.dim_2B = dim_2B
        self.dim_2A = dim_2A
        self.dim_2C = dim_2C

        # ENCODER
        # Layer 1
        self.encode_fc_1B = FCBlock(self.B_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2C = FCBlock(dim_1C, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A+dim_2C, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B+dim_2A+dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4B = FCBlock(dim_1B, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B = self.encode_fc_1B(x[1])
        level_2_A = self.encode_fc_1A(x[0])
        level_2_C = self.encode_fc_1C(x[2])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3_C = self.encode_fc_2C(level_2_C)
        level_3 = torch.cat((level_3_B, level_3_A, level_3_C), 1)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)
        level_2_C = level_2.narrow(1, self.dim_2B+self.dim_2A, self.dim_2C)

        level_3_B = self.decode_fc_3B(level_2_B)
        level_3_A = self.decode_fc_3A(level_2_A)
        level_3_C = self.decode_fc_3C(level_2_C)

        recon_B = self.decode_fc_4B(level_3_B)
        recon_A = self.decode_fc_4A(level_3_A)
        recon_C = self.decode_fc_4C(level_3_C)

        return [recon_A, recon_B, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcVaeAB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input not separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=384, dim_2B=256,
                 dim_1A=384, dim_2A=256, dim_3=256, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeAB, self).__init__()

        self.A_dim = omics_dims[0]
        self.B_dim = omics_dims[1]
        self.dim_1B = dim_1B
        self.dim_2B = dim_2B
        self.dim_2A = dim_2A

        # ENCODER
        # Layer 1
        self.encode_fc_1B = FCBlock(self.B_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B+dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4B = FCBlock(dim_1B, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B = self.encode_fc_1B(x[1])
        level_2_A = self.encode_fc_1A(x[0])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3 = torch.cat((level_3_B, level_3_A), 1)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)

        level_3_B = self.decode_fc_3B(level_2_B)
        level_3_A = self.decode_fc_3A(level_2_A)

        recon_B = self.decode_fc_4B(level_3_B)
        recon_A = self.decode_fc_4A(level_3_A)

        return [recon_A, recon_B]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcVaeB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for DNA methylation dataset
        DNA methylation input not separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=512, dim_2B=256,
                 dim_3=256, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeB, self).__init__()

        self.B_dim = omics_dims[1]

        # ENCODER
        # Layer 1
        self.encode_fc_1B = FCBlock(self.B_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                      dropout_p=0, activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                         dropout_p=0, activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                   dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4B = FCBlock(dim_1B, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B = self.encode_fc_1B(x[1])

        level_3 = self.encode_fc_2B(level_2_B)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)

        level_3_B = self.decode_fc_3B(level_2)

        recon_B = self.decode_fc_4B(level_3_B)

        return [None, recon_B]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcVaeA(nn.Module):
    """
        Defines a fully-connected variational autoencoder for gene expression dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1A=1024, dim_2A=1024,
                 dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeA, self).__init__()

        self.A_dim = omics_dims[0]

        # ENCODER
        # Layer 1
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2A, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_A = self.encode_fc_1A(x[0])

        level_3_A = self.encode_fc_2A(level_2_A)

        level_4 = self.encode_fc_3(level_3_A)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)

        level_3_A = self.decode_fc_3A(level_2)

        recon_A = self.decode_fc_4A(level_3_A)

        return [recon_A]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcVaeC(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1C=1024, dim_2C=1024, dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeC, self).__init__()

        self.C_dim = omics_dims[2]
        self.dim_2C = dim_2C

        # ENCODER
        # Layer 1
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2C = FCBlock(dim_1C, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2C, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_C = self.encode_fc_1C(x[2])

        level_3_C = self.encode_fc_2C(level_2_C)

        level_4 = self.encode_fc_3(level_3_C)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)

        level_3_C = self.decode_fc_3C(level_2)

        recon_C = self.decode_fc_4C(level_3_C)

        return [None, None, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


# Class for downstream task
class MultiFcClassifier(nn.Module):
    """
    Defines a multi-layer fully-connected classifier
    """
    def __init__(self, class_num=2, latent_dim=256, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 class_dim_1=128, class_dim_2=64, layer_num=3):
        """
        Construct a multi-layer fully-connected classifier
        Parameters:
            class_num (int)         -- the number of class
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(MultiFcClassifier, self).__init__()

        self.input_fc = FCBlock(latent_dim, class_dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                activation=True)

        # create a list to store fc blocks
        mul_fc_block = []
        # the block number of the multi-layer fully-connected block should be at least 3
        block_layer_num = max(layer_num, 3)
        input_dim = class_dim_1
        dropout_flag = True
        for num in range(0, block_layer_num-2):
            mul_fc_block += [FCBlock(input_dim, class_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout_p=dropout_flag*dropout_p, activation=True)]
            input_dim = class_dim_2
            # dropout for every other layer
            dropout_flag = not dropout_flag
        self.mul_fc = nn.Sequential(*mul_fc_block)

        # the output fully-connected layer of the classifier
        self.output_fc = FCBlock(class_dim_2, class_num, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                 activation=False, normalization=False)

    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y


class MultiFcRegression(nn.Module):
    """
    Defines a multi-layer fully-connected regression net
    """
    def __init__(self, latent_dim=256, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, down_dim_1=128,
                 down_dim_2=64, layer_num=3):
        """
        Construct a one dimensional multi-layer regression net
        Parameters:
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(MultiFcRegression, self).__init__()

        self.input_fc = FCBlock(latent_dim, down_dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                activation=True)

        # create a list to store fc blocks
        mul_fc_block = []
        # the block number of the multi-layer fully-connected block should be at least 3
        block_layer_num = max(layer_num, 3)
        input_dim = down_dim_1
        dropout_flag = True
        for num in range(0, block_layer_num-2):
            mul_fc_block += [FCBlock(input_dim, down_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope,
                             dropout_p=dropout_flag*dropout_p, activation=True)]
            input_dim = down_dim_2
            # dropout for every other layer
            dropout_flag = not dropout_flag
        self.mul_fc = nn.Sequential(*mul_fc_block)

        # the output fully-connected layer of the classifier
        self.output_fc = FCBlock(down_dim_2, 1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                 activation=False, normalization=False)

    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y


class MultiFcSurvival(nn.Module):
    """
    Defines a multi-layer fully-connected survival predictor
    """
    def __init__(self, time_num=256, latent_dim=128, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 down_dim_1=512, down_dim_2=256, layer_num=3):
        """
        Construct a multi-layer fully-connected survival predictor
        Parameters:
            time_num (int)          -- the number of time intervals in the model
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(MultiFcSurvival, self).__init__()

        self.input_fc = FCBlock(latent_dim, down_dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                activation=True, activation_name='Tanh')

        # create a list to store fc blocks
        mul_fc_block = []
        # the block number of the multi-layer fully-connected block should be at least 3
        block_layer_num = max(layer_num, 3)
        input_dim = down_dim_1
        dropout_flag = True
        for num in range(0, block_layer_num-2):
            mul_fc_block += [FCBlock(input_dim, down_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout_p=dropout_p, activation=True, activation_name='Tanh')]
            input_dim = down_dim_2
            # dropout for every other layer
            dropout_flag = not dropout_flag
        self.mul_fc = nn.Sequential(*mul_fc_block)

        # the output fully-connected layer of the classifier
        # the output dimension should be the number of time intervals
        self.output_fc = FCBlock(down_dim_2, time_num, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                 activation=False, normalization=False)

    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y


class MultiFcMultitask(nn.Module):
    """
        Defines a multi-layer fully-connected multitask downstream network
    """

    def __init__(self, class_num=2, time_num=256, latent_dim=128, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 layer_num=3):
        """
        Construct a multi-layer fully-connected multitask downstream network
        Parameters:
            class_num (int)         -- the number of class
            time_num (int)          -- the number of time intervals in the model
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the downstream networks, >=3
        """
        super(MultiFcMultitask, self).__init__()
        norm_layer_none = lambda x: Identity()
        self.survival = MultiFcSurvival(time_num, latent_dim, norm_layer=norm_layer_none, leaky_slope=leaky_slope, dropout_p=0.5, layer_num=layer_num)
        self.classifier = MultiFcClassifier(class_num, latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=leaky_slope, dropout_p=0.2, layer_num=layer_num)
        self.regression = MultiFcRegression(latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=leaky_slope, dropout_p=0.01, layer_num=layer_num)

    def forward(self, x):
        y_out_sur = self.survival(x)
        y_out_cla = self.classifier(x)
        y_out_reg = self.regression(x)
        return y_out_sur, y_out_cla, y_out_reg


class MultiFcAlltask(nn.Module):
    """
        Defines a multi-layer fully-connected multitask downstream network (all tasks)
    """

    def __init__(self, class_num, time_num=256, task_num=7, latent_dim=128, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 layer_num=3):
        """
        Construct a multi-layer fully-connected multitask downstream network (all tasks)
        Parameters:
            class_num (list)        -- the list of class numbers
            time_num (int)          -- the number of time intervals in the model
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
            task_num (int)          -- the number of downstream tasks
        """
        super(MultiFcAlltask, self).__init__()
        norm_layer_none = lambda x: Identity()
        self.survival = MultiFcSurvival(time_num, latent_dim, norm_layer=norm_layer_none, leaky_slope=leaky_slope, dropout_p=0.5, layer_num=layer_num)
        self.classifiers = nn.ModuleList([MultiFcClassifier(class_num[i], latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=leaky_slope, dropout_p=0.2, layer_num=layer_num) for i in range(task_num-2)])
        self.regression = MultiFcRegression(latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=leaky_slope, dropout_p=0.01, layer_num=layer_num)
        self.task_num = task_num

    def forward(self, x):
        y_out_sur = self.survival(x)
        y_out_cla = []
        for i in range(self.task_num - 2):
            y_out_cla.append(self.classifiers[i](x))
        y_out_reg = self.regression(x)
        return y_out_sur, y_out_cla, y_out_reg


# Class for the OmiEmbed combined network
class OmiEmbed(nn.Module):
    """
    Defines the OmiEmbed combined network
    """
    def __init__(self, net_VAE, net_down, omics_dims, omics_mode='multi_omics', norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9,
                 leaky_slope=0.2, dropout_p=0, latent_dim=128, class_num=2, time_num=256, task_num=7):
        """
            Construct the OmiEmbed combined network
            Parameters:
                net_VAE (str)           -- the backbone of the VAE, default: conv_1d
                net_down (str)          -- the backbone of the downstream task network, default: multi_FC_classifier
                omics_dims (list)       -- the list of input omics dimensions
                omics_mode (str)        -- omics types would like to use in the model
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
                class_num (int/list)    -- the number of classes
                time_num (int)          -- the number of time intervals
                task_num (int)          -- the number of downstream tasks
        """
        super(OmiEmbed, self).__init__()

        self.vae = None
        if net_VAE == 'conv_1d':
            if omics_mode == 'abc':
                self.vae = ConvVaeABC(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'ab':
                self.vae = ConvVaeAB(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'b':
                self.vae = ConvVaeB(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'a':
                self.vae = ConvVaeA(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'c':
                self.vae = ConvVaeC(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p, latent_dim=latent_dim)

        elif net_VAE == 'fc_sep':
            if omics_mode == 'abc':
                self.vae = FcSepVaeABC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'ab':
                self.vae = FcSepVaeAB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'b':
                self.vae = FcSepVaeB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'a':
                self.vae = FcVaeA(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'c':
                self.vae = FcVaeC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)

        elif net_VAE == 'fc':
            if omics_mode == 'abc':
                self.vae = FcVaeABC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'ab':
                self.vae = FcVaeAB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'b':
                self.vae = FcVaeB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'a':
                self.vae = FcVaeA(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'c':
                self.vae = FcVaeC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        else:
            raise NotImplementedError('VAE model name [%s] is not recognized' % net_VAE)

        self.net_down = net_down
        self.down = None
        if net_down == 'multi_FC_classifier':
            self.down = MultiFcClassifier(class_num, latent_dim, norm_layer, leaky_slope, dropout_p)
        elif net_down == 'multi_FC_regression':
            self.down = MultiFcRegression(latent_dim, norm_layer, leaky_slope, dropout_p)
        elif net_down == 'multi_FC_survival':
            self.down = MultiFcSurvival(time_num, latent_dim, norm_layer, leaky_slope, dropout_p)
        elif net_down == 'multi_FC_multitask':
            self.down = MultiFcMultitask(class_num, time_num, latent_dim, norm_layer, leaky_slope, dropout_p)
        elif net_down == 'multi_FC_alltask':
            self.down = MultiFcAlltask(class_num, time_num, task_num, latent_dim, norm_layer, leaky_slope, dropout_p)
        else:
            raise NotImplementedError('Downstream model name [%s] is not recognized' % net_down)

    def get_last_encode_layer(self):
        return self.vae.get_last_encode_layer()

    def forward(self, x):
        z, recon_x, mean, log_var = self.vae(x)
        if self.net_down == 'multi_FC_multitask' or self.net_down == 'multi_FC_alltask':
            y_out_sur, y_out_cla, y_out_reg = self.down(mean)
            return z, recon_x, mean, log_var, y_out_sur, y_out_cla, y_out_reg
        else:
            y_out = self.down(mean)
            return z, recon_x, mean, log_var, y_out


def get_norm_layer(norm_type='batch'):
    """
    Return a normalization layer
    Parameters:
        norm_type (str) -- the type of normalization applied to the model, default to use batch normalization, options: [batch | instance | none ]
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization method [%s] is not found' % norm_type)
    return norm_layer


def define_net(net_VAE, net_down, omics_dims, omics_mode='multi_omics', norm_type='batch', filter_num=8, kernel_size=9,
               leaky_slope=0.2, dropout_p=0, latent_dim=256, class_num=2, time_num=256, task_num=7, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create the OmiEmbed network

    Parameters:
        net_VAE (str)           -- the backbone of the VAE, default: conv_1d
        net_down (str)          -- the backbone of the downstream task network, default: multi_FC_classifier
        omics_dims (list)       -- the list of input omics dimensions
        omics_mode (str)        -- omics types would like to use in the model
        norm_type (str)         -- the name of normalization layers used in the network, default: batch
        filter_num (int)        -- the number of filters in the first convolution layer in the VAE
        kernel_size (int)       -- the kernel size of convolution layers
        leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
        latent_dim (int)        -- the dimensionality of the latent space
        class_num (int)         -- the number of classes
        time_num (int)          -- the number of time intervals
        task_num (int)          -- the number of downstream tasks
        init_type (str)         -- the name of our initialization method
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
        gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1

    Returns the OmiEmbed network

    The network has been initialized by <init_net>.
    """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    net = OmiEmbed(net_VAE, net_down, omics_dims, omics_mode, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                   latent_dim, class_num, time_num, task_num)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_VAE(net_VAE, omics_dims, omics_mode='multi_omics', norm_type='batch', filter_num=8, kernel_size=9, leaky_slope=0.2, dropout_p=0,
               latent_dim=256, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create the VAE network

    Parameters:
        net_VAE (str)           -- the backbone of the VAE, default: conv_1d
        omics_dims (list)       -- the list of input omics dimensions
        omics_mode (str)        -- omics types would like to use in the model
        norm_type (str)         -- the name of normalization layers used in the network, default: batch
        filter_num (int)        -- the number of filters in the first convolution layer in the VAE
        kernel_size (int)       -- the kernel size of convolution layers
        leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
        latent_dim (int)        -- the dimensionality of the latent space
        init_type (str)         -- the name of our initialization method
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
        gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1

    Returns a VAE

    The default backbone of the VAE is one dimensional convolutional layer.

    The generator has been initialized by <init_net>.
    """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    if net_VAE == 'conv_1d':
        if omics_mode == 'abc':
            net = ConvVaeABC(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                             latent_dim=latent_dim)
        elif omics_mode == 'ab':
            net = ConvVaeAB(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                            latent_dim=latent_dim)
        elif omics_mode == 'b':
            net = ConvVaeB(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                           latent_dim=latent_dim)
        elif omics_mode == 'a':
            net = ConvVaeA(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                           latent_dim=latent_dim)
        elif omics_mode == 'c':
            net = ConvVaeC(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                           latent_dim=latent_dim)

    elif net_VAE == 'fc_sep':
        if omics_mode == 'abc':
            net = FcSepVaeABC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'ab':
            net = FcSepVaeAB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'b':
            net = FcSepVaeB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'a':
            net = FcVaeA(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'c':
            net = FcVaeC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)

    elif net_VAE == 'fc':
        if omics_mode == 'abc':
            net = FcVaeABC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'ab':
            net = FcVaeAB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'b':
            net = FcVaeB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'a':
            net = FcVaeA(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'c':
            net = FcVaeC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
    else:
        raise NotImplementedError('VAE model name [%s] is not recognized' % net_VAE)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_down(net_down, norm_type='batch', leaky_slope=0.2, dropout_p=0, latent_dim=256, class_num=2, time_num=256,
                task_num=7, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
        Create the downstream task network

        Parameters:
            net_down (str)          -- the backbone of the downstream task network, default: multi_FC_classifier
            norm_type (str)         -- the name of normalization layers used in the network, default: batch
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            class_num (int)         -- the number of class
            time_num (int)          -- the number of time intervals
            task_num (int)          -- the number of downstream tasks
            init_type (str)         -- the name of our initialization method
            init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
            gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1

        Returns a downstream task network

        The default downstream task network is a multi-layer fully-connected classifier.

        The generator has been initialized by <init_net>.
        """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    if net_down == 'multi_FC_classifier':
        net = MultiFcClassifier(class_num, latent_dim, norm_layer, leaky_slope, dropout_p)
    elif net_down == 'multi_FC_regression':
        net = MultiFcRegression(latent_dim, norm_layer, leaky_slope, dropout_p)
    elif net_down == 'multi_FC_survival':
        net = MultiFcSurvival(time_num, latent_dim, norm_layer, leaky_slope, dropout_p)
    elif net_down == 'multi_FC_multitask':
        net = MultiFcMultitask(class_num, time_num, latent_dim, norm_layer, leaky_slope, dropout_p)
    elif net_down == 'multi_FC_alltask':
        net = MultiFcAlltask(class_num, time_num, task_num, latent_dim, norm_layer, leaky_slope, dropout_p)
    else:
        raise NotImplementedError('Downstream model name [%s] is not recognized' % net_down)

    return init_net(net, init_type, init_gain, gpu_ids)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Initialize a network:
    1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights
    Parameters:
        net (nn.Module)    -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # multi-GPUs
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (nn.Module)    -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier_normal | xavier_uniform | kaiming_normal | kaiming_uniform | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
    """
    # define the initialization function
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('Initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_scheduler(optimizer, param):
    """
    Return a learning rate scheduler

    Parameters:
        optimizer (opt class)     -- the optimizer of the network
        param (params class)      -- param.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <param.niter> epochs and linearly decay the rate to zero
    over the next <param.niter_decay> epochs.

    """
    if param.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_lambda = 1.0 - max(0, epoch + param.epoch_count - param.epoch_num + param.epoch_num_decay) / float(param.epoch_num_decay + 1)
            return lr_lambda
        # lr_scheduler is imported from torch.optim
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif param.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=param.decay_step_size, gamma=0.1)
    elif param.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif param.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=param.epoch_num, eta_min=0)
    else:
        return NotImplementedError('Learning rate policy [%s] is not found', param.lr_policy)
    return scheduler
