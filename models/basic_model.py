import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from . import networks
from collections import OrderedDict


class BasicModel(ABC):
    """
    This class is an abstract base class for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                          Initialize the class, first call BasicModel.__init__(self, param)
        -- <modify_commandline_parameters>:     Add model-specific parameters, and rewrite default values for existing parameters
        -- <set_input>:                         Unpack input data from the output dictionary of the dataloader
        -- <forward>:                           Get the reconstructed omics data and results for the downstream task
        -- <update>:                            Calculate losses, gradients and update network parameters
    """

    def __init__(self, param):
        """
        Initialize the BaseModel class
        """
        self.param = param
        self.gpu_ids = param.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(param.checkpoints_dir, param.experiment_name)  # save all the checkpoints to save_dir, and this is where to load the models
        self.load_net_dir = os.path.join(param.checkpoints_dir, param.experiment_to_load)  # load pretrained networks from certain experiment folder
        self.isTrain = param.isTrain
        self.phase = 'p1'
        self.epoch = 1
        self.iter = 0

        # Improve the performance if the dimensionality and shape of the input data keep the same
        torch.backends.cudnn.benchmark = True

        self.plateau_metric = 0  # used for learning rate policy 'plateau'

        self.loss_names = []
        self.model_names = []
        self.metric_names = []
        self.optimizers = []
        self.schedulers = []

        self.latent = None
        self.loss_embed = None
        self.loss_down = None
        self.loss_All = None

    @staticmethod
    def modify_commandline_parameters(parser, is_train):
        """
        Add model-specific parameters, and rewrite default values for existing parameters.

        Parameters:
            parser              -- original parameter parser
            is_train (bool)     -- whether it is currently training phase or test phase. Use this flag to add or change training-specific or test-specific parameters.

        Returns:
            The modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input_dict):
        """
        Unpack input data from the output dictionary of the dataloader

        Parameters:
            input_dict (dict): include the data tensor and its label
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Run forward pass
        """
        pass

    @abstractmethod
    def cal_losses(self):
        """
        Calculate losses
        """
        pass

    @abstractmethod
    def update(self):
        """
        Calculate losses, gradients and update network weights; called in every training iteration
        """
        pass

    def setup(self, param):
        """
        Load and print networks, create schedulers
        """
        if self.isTrain:
            self.print_networks(param)
            # For every optimizer we have a scheduler
            self.schedulers = [networks.get_scheduler(optimizer, param) for optimizer in self.optimizers]

        # Loading the networks
        if not self.isTrain or param.continue_train:
            self.load_networks(param.epoch_to_load)

    def update_learning_rate(self):
        """
        Update learning rates for all the networks
        Called at the end of each epoch
        """
        lr = self.optimizers[0].param_groups[0]['lr']

        for scheduler in self.schedulers:
            if self.param.lr_policy == 'plateau':
                scheduler.step(self.plateau_metric)
            else:
                scheduler.step()

        return lr


    def print_networks(self, param):
        """
        Print the total number of parameters in the network and network architecture if detail is true
        Save the networks information to the disk
        """
        message = '\n----------------------Networks Information----------------------'
        for model_name in self.model_names:
            if isinstance(model_name, str):
                net = getattr(self, 'net' + model_name)
                num_params = 0
                for parameter in net.parameters():
                    num_params += parameter.numel()
                if param.detail:
                    message += '\n' + str(net)
                message += '\n[Network {:s}] Total number of parameters : {:.3f} M'.format(model_name, num_params / 1e6)
        message += '\n----------------------------------------------------------------\n'

        # Save the networks information to the disk
        net_info_filename = os.path.join(param.checkpoints_dir, param.experiment_name, 'net_info.txt')
        with open(net_info_filename, 'w') as log_file:
            log_file.write(message)

        print(message)

    def save_networks(self, epoch):
        """
        Save all the networks to the disk.

        Parameters:
            epoch (str) -- current epoch
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '{:s}_net_{:s}.pth'.format(epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                # Use the str to get the attribute aka the network (self.netG / self.netD)
                net = getattr(self, 'net' + name)
                # If we use multi GPUs and apply the data parallel
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """
        Load networks at specified epoch from the disk.

        Parameters:
            epoch (str) -- Which epoch to load
        """
        for model_name in self.model_names:
            if isinstance(model_name, str):
                load_filename = '{:s}_net_{:s}.pth'.format(epoch, model_name)
                load_path = os.path.join(self.load_net_dir, load_filename)
                # Use the str to get the attribute aka the network (self.netG / self.netD)
                net = getattr(self, 'net' + model_name)
                # If we use multi GPUs and apply the data parallel
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('Loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def set_train(self):
        """
        Set train mode for networks
        """
        for model_name in self.model_names:
            if isinstance(model_name, str):
                # Use the str to get the attribute aka the network (self.netXXX)
                net = getattr(self, 'net' + model_name)
                net.train()

    def set_eval(self):
        """
        Set eval mode for networks
        """
        for model_name in self.model_names:
            if isinstance(model_name, str):
                # Use the str to get the attribute aka the network (self.netG / self.netD)
                net = getattr(self, 'net' + model_name)
                net.eval()

    def test(self):
        """
        Forward in testing to get the output tensors
        """
        with torch.no_grad():
            self.forward()
            self.cal_losses()

    def init_output_dict(self):
        """
        initialize a dictionary for downstream task output
        """
        output_dict = OrderedDict()
        output_names = []
        if self.param.downstream_task == 'classification':
            output_names = ['index', 'y_true', 'y_pred', 'y_prob']
        elif self.param.downstream_task == 'regression':
            output_names = ['index', 'y_true', 'y_pred']
        elif self.param.downstream_task == 'survival':
            output_names = ['index', 'y_true_E', 'y_true_T', 'survival', 'risk', 'y_out']
        elif self.param.downstream_task == 'multitask' or self.param.downstream_task == 'alltask':
            output_names = ['index', 'y_true_E', 'y_true_T', 'survival', 'risk', 'y_out_sur', 'y_true_cla', 'y_pred_cla',
                            'y_prob_cla', 'y_true_reg', 'y_pred_reg']
        for name in output_names:
            output_dict[name] = None

        return output_dict

    def update_output_dict(self, output_dict):
        """
        output_dict (OrderedDict)  -- the output dictionary to be updated
        """
        down_output = self.get_down_output()
        output_names = []
        if self.param.downstream_task == 'classification':
            output_names = ['index', 'y_true', 'y_pred', 'y_prob']
        elif self.param.downstream_task == 'regression':
            output_names = ['index', 'y_true', 'y_pred']
        elif self.param.downstream_task == 'survival':
            output_names = ['index', 'y_true_E', 'y_true_T', 'survival', 'risk', 'y_out']
        elif self.param.downstream_task == 'multitask' or self.param.downstream_task == 'alltask':
            output_names = ['index', 'y_true_E', 'y_true_T', 'survival', 'risk', 'y_out_sur', 'y_true_cla',
                            'y_pred_cla', 'y_prob_cla', 'y_true_reg', 'y_pred_reg']

        for name in output_names:
            if output_dict[name] is None:
                output_dict[name] = down_output[name]
            else:
                if self.param.downstream_task == 'alltask' and name in ['y_true_cla', 'y_pred_cla', 'y_prob_cla']:
                    for i in range(self.param.task_num-2):
                        output_dict[name][i] = torch.cat((output_dict[name][i], down_output[name][i]))
                else:
                    output_dict[name] = torch.cat((output_dict[name], down_output[name]))

    def init_losses_dict(self):
        """
        initialize a losses dictionary
        """
        losses_dict = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses_dict[name] = []
        return losses_dict

    def update_losses_dict(self, losses_dict, actual_batch_size):
        """
        losses_dict (OrderedDict)   -- the losses dictionary to be updated
        actual_batch_size (int)     -- actual batch size for loss normalization
        """
        for name in self.loss_names:
            if isinstance(name, str):
                if self.param.reduction == 'sum':
                    losses_dict[name].append(float(getattr(self, 'loss_' + name))/actual_batch_size)
                elif self.param.reduction == 'mean':
                    losses_dict[name].append(float(getattr(self, 'loss_' + name)))

    def init_metrics_dict(self):
        """
        initialize a metrics dictionary
        """
        metrics_dict = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str):
                metrics_dict[name] = None
        return metrics_dict

    def update_metrics_dict(self, metrics_dict):
        """
        metrics_dict (OrderedDict)  -- the metrics dictionary to be updated
        """
        for name in self.metric_names:
            if isinstance(name, str):
                metrics_dict[name] = getattr(self, 'metric_' + name)

    def init_log_dict(self):
        """
        initialize losses and metrics dictionary
        """
        output_dict = self.init_output_dict()
        losses_dict = self.init_losses_dict()
        metrics_dict = self.init_metrics_dict()
        return output_dict, losses_dict, metrics_dict

    def update_log_dict(self, output_dict, losses_dict, metrics_dict, actual_batch_size):
        """
        output_dict (OrderedDict)  -- the output dictionary to be updated
        losses_dict (OrderedDict)  -- the losses dictionary to be updated
        metrics_dict (OrderedDict)  -- the metrics dictionary to be updated
        actual_batch_size (int)     -- actual batch size for loss normalization
        """
        self.update_output_dict(output_dict)
        self.calculate_current_metrics(output_dict)
        self.update_losses_dict(losses_dict, actual_batch_size)
        self.update_metrics_dict(metrics_dict)

    def init_latent_dict(self):
        """
        initialize and return an empty latent space array and an empty index array
        """
        latent_dict = OrderedDict()
        latent_dict['index'] = np.zeros(shape=[0])
        latent_dict['latent'] = np.zeros(shape=[0, self.param.latent_space_dim])
        return latent_dict

    def update_latent_dict(self, latent_dict):
        """
        update the latent dict
        latent_dict (OrderedDict)
        """
        with torch.no_grad():
            current_latent_array = self.latent.cpu().numpy()
            latent_dict['latent'] = np.concatenate((latent_dict['latent'], current_latent_array))
            current_index_array = self.data_index.cpu().numpy()
            latent_dict['index'] = np.concatenate((latent_dict['index'], current_index_array))
            return latent_dict
