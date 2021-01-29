import torch
from .basic_model import BasicModel
from . import networks
from . import losses


class VaeBasicModel(BasicModel):
    """
    This is the basic VAE model class, called by all other VAE son classes.
    """

    def __init__(self, param):
        """
        Initialize the VAE basic class.
        """
        BasicModel.__init__(self, param)
        # specify the training losses you want to print out.
        if param.omics_mode == 'abc':
            self.loss_names = ['recon_A', 'recon_B', 'recon_C', 'kl']
        if param.omics_mode == 'ab':
            self.loss_names = ['recon_A', 'recon_B', 'kl']
        elif param.omics_mode == 'b':
            self.loss_names = ['recon_B', 'kl']
        elif param.omics_mode == 'a':
            self.loss_names = ['recon_A', 'kl']
        elif param.omics_mode == 'c':
            self.loss_names = ['recon_C', 'kl']
        # specify the models you want to save to the disk and load.
        self.model_names = ['Embed', 'Down']

        # input tensor
        self.input_omics = []
        self.data_index = None  # The indexes of input data

        # output tensor
        self.z = None
        self.recon_omics = None
        self.mean = None
        self.log_var = None

        # define the network
        self.netEmbed = networks.define_VAE(param.net_VAE, param.omics_dims, param.omics_mode,
                                            param.norm_type, param.filter_num, param.conv_k_size, param.leaky_slope,
                                            param.dropout_p, param.latent_space_dim, param.init_type, param.init_gain,
                                            self.gpu_ids)

        # define the reconstruction loss
        self.lossFuncRecon = losses.get_loss_func(param.recon_loss, param.reduction)

        self.loss_recon_A = None
        self.loss_recon_B = None
        self.loss_recon_C = None
        self.loss_recon = None
        self.loss_kl = None

        if self.isTrain:
            # Set the optimizer
            # netEmbed and netDown can set to different initial learning rate
            self.optimizer_Embed = torch.optim.Adam(self.netEmbed.parameters(), lr=param.lr, betas=(param.beta1, 0.999), weight_decay=param.weight_decay)
            # optimizer list was already defined in BaseModel
            self.optimizers.append(self.optimizer_Embed)

            self.optimizer_Down = None

    def set_input(self, input_dict):
        """
        Unpack input data from the output dictionary of the dataloader

        Parameters:
            input_dict (dict): include the data tensor and its index.
        """
        self.input_omics = []
        for i in range(0, 3):
            if i == 1 and self.param.ch_separate:
                input_B = []
                for ch in range(0, 23):
                    input_B.append(input_dict['input_omics'][1][ch].to(self.device))
                self.input_omics.append(input_B)
            else:
                self.input_omics.append(input_dict['input_omics'][i].to(self.device))

        self.data_index = input_dict['index']

    def forward(self):
        # Get the output tensor
        self.z, self.recon_omics, self.mean, self.log_var = self.netEmbed(self.input_omics)
        # define the latent
        if self.phase == 'p1' or self.phase == 'p3':
            self.latent = self.mean
        elif self.phase == 'p2':
            self.latent = self.mean.detach()

    def cal_losses(self):
        """Calculate losses"""
        # Calculate the reconstruction loss for A
        if self.param.omics_mode == 'a' or self.param.omics_mode == 'ab' or self.param.omics_mode == 'abc':
            self.loss_recon_A = self.lossFuncRecon(self.recon_omics[0], self.input_omics[0])
        else:
            self.loss_recon_A = 0
        # Calculate the reconstruction loss for B
        if self.param.omics_mode == 'b' or self.param.omics_mode == 'ab' or self.param.omics_mode == 'abc':
            if self.param.ch_separate:
                recon_omics_B = torch.cat(self.recon_omics[1], -1)
                input_omics_B = torch.cat(self.input_omics[1], -1)
                self.loss_recon_B = self.lossFuncRecon(recon_omics_B, input_omics_B)
            else:
                self.loss_recon_B = self.lossFuncRecon(self.recon_omics[1], self.input_omics[1])
        else:
            self.loss_recon_B = 0
        # Calculate the reconstruction loss for C
        if self.param.omics_mode == 'c' or self.param.omics_mode == 'abc':
            self.loss_recon_C = self.lossFuncRecon(self.recon_omics[2], self.input_omics[2])
        else:
            self.loss_recon_C = 0
        # Overall reconstruction loss
        if self.param.reduction == 'sum':
            self.loss_recon = self.loss_recon_A + self.loss_recon_B + self.loss_recon_C
        elif self.param.reduction == 'mean':
            self.loss_recon = (self.loss_recon_A + self.loss_recon_B + self.loss_recon_C) / self.param.omics_num
        # Calculate the kl loss
        self.loss_kl = losses.kl_loss(self.mean, self.log_var, self.param.reduction)
        # Calculate the overall vae loss (embedding loss)
        # LOSS EMBED
        self.loss_embed = self.loss_recon + self.param.k_kl * self.loss_kl

    def update(self):
        if self.phase == 'p1':
            self.forward()
            self.optimizer_Embed.zero_grad()                # Set gradients to zero
            self.cal_losses()                               # Calculate losses
            self.loss_embed.backward()                      # Backpropagation
            self.optimizer_Embed.step()                     # Update weights
        elif self.phase == 'p2':
            self.forward()
            self.optimizer_Down.zero_grad()                 # Set gradients to zero
            self.cal_losses()                               # Calculate losses
            self.loss_down.backward()                       # Backpropagation
            self.optimizer_Down.step()                      # Update weights
        elif self.phase == 'p3':
            self.forward()
            self.optimizer_Embed.zero_grad()                # Set gradients to zero
            self.optimizer_Down.zero_grad()
            self.cal_losses()                               # Calculate losses
            self.loss_All.backward()                        # Backpropagation
            self.optimizer_Embed.step()                     # Update weights
            self.optimizer_Down.step()
