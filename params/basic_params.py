import time
import argparse
import torch
import os
import models
from util import util


class BasicParams:
    """
    This class define the console parameters
    """

    def __init__(self):
        """
        Reset the class. Indicates the class hasn't been initialized
        """
        self.initialized = False
        self.isTrain = True
        self.isTest = True

    def initialize(self, parser):
        """
        Define the common console parameters
        """
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='which GPU would like to use: e.g. 0 or 0,1, -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='models, settings and intermediate results are saved in folder in this directory')
        parser.add_argument('--experiment_name', type=str, default='test',
                            help='name of the folder in the checkpoint directory')

        # Dataset parameters
        parser.add_argument('--omics_mode', type=str, default='a',
                            help='omics types would like to use in the model, options: [abc | ab | a | b | c]')
        parser.add_argument('--data_root', type=str, default='./data',
                            help='path to input data')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='input data batch size')
        parser.add_argument('--num_threads', default=0, type=int,
                            help='number of threads for loading data')
        parser.add_argument('--set_pin_memory', action='store_true',
                            help='set pin_memory in the dataloader to increase data loading performance')
        parser.add_argument('--not_stratified', action='store_true',
                            help='do not apply the stratified mode in train/test split if set true')
        parser.add_argument('--use_sample_list', action='store_true',
                            help='provide a subset sample list of the dataset, store in the path data_root/sample_list.tsv, if False use all the samples')
        parser.add_argument('--use_feature_lists', action='store_true',
                            help='provide feature lists of the input omics data, e.g. data_root/feature_list_A.tsv, if False use all the features')
        parser.add_argument('--detect_na', action='store_true',
                            help='detect missing value markers during data loading, stay False can improve the loading performance')
        parser.add_argument('--file_format', type=str, default='tsv',
                            help='file format of the omics data, options: [tsv | csv | hdf]')

        # Model parameters
        parser.add_argument('--model', type=str, default='vae_classifier',
                            help='chooses which model want to use, options: [vae_classifier | vae_regression | vae_survival | vae_multitask]')
        parser.add_argument('--net_VAE', type=str, default='fc_sep',
                            help='specify the backbone of the VAE, default is the one dimensional CNN, options: [conv_1d | fc_sep | fc]')
        parser.add_argument('--net_down', type=str, default='multi_FC_classifier',
                            help='specify the backbone of the downstream task network, default is the multi-layer FC classifier, options: [multi_FC_classifier | multi_FC_regression | multi_FC_survival | multi_FC_multitask]')
        parser.add_argument('--norm_type', type=str, default='batch',
                            help='the type of normalization applied to the model, default to use batch normalization, options: [batch | instance | none ]')
        parser.add_argument('--filter_num', type=int, default=8,
                            help='number of filters in the last convolution layer in the generator')
        parser.add_argument('--conv_k_size', type=int, default=9,
                            help='the kernel size of convolution layer, default kernel size is 9, the kernel is one dimensional.')
        parser.add_argument('--dropout_p', type=float, default=0.2,
                            help='probability of an element to be zeroed in a dropout layer, default is 0 which means no dropout.')
        parser.add_argument('--leaky_slope', type=float, default=0.2,
                            help='the negative slope of the Leaky ReLU activation function')
        parser.add_argument('--latent_space_dim', type=int, default=128,
                            help='the dimensionality of the latent space')
        parser.add_argument('--seed', type=int, default=42,
                            help='random seed')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='choose the method of network initialization, options: [normal | xavier_normal | xavier_uniform | kaiming_normal | kaiming_uniform | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal initialization methods')

        # Loss parameters
        parser.add_argument('--recon_loss', type=str, default='BCE',
                            help='chooses the reconstruction loss function, options: [BCE | MSE | L1]')
        parser.add_argument('--reduction', type=str, default='mean',
                            help='chooses the reduction to apply to the loss function, options: [sum | mean]')
        parser.add_argument('--k_kl', type=float, default=0.01,
                            help='weight for the kl loss')
        parser.add_argument('--k_embed', type=float, default=0.001,
                            help='weight for the embedding loss')

        # Other parameters
        parser.add_argument('--deterministic', action='store_true',
                            help='make the model deterministic for reproduction if set true')
        parser.add_argument('--detail', action='store_true',
                            help='print more detailed information if set true')
        parser.add_argument('--epoch_to_load', type=str, default='latest',
                            help='the epoch number to load, set latest to load latest cached model')
        parser.add_argument('--experiment_to_load', type=str, default='test',
                            help='the experiment to load')

        self.initialized = True  # set the initialized to True after we define the parameters of the project
        return parser

    def get_params(self):
        """
        Initialize our parser with basic parameters once.
        Add additional model-specific parameters.
        """
        if not self.initialized:  # check if this object has been initialized
            # if not create a new parser object
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            #  use our method to initialize the parser with the predefined arguments
            parser = self.initialize(parser)

        # get the basic parameters
        param, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = param.model
        model_param_setter = models.get_param_setter(model_name)
        parser = model_param_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_params(self, param):
        """
        Print welcome words and command line parameters.
        Save the command line parameters in a txt file to the disk
        """
        message = ''
        message += '\nWelcome to OmiEmbed\nby Xiaoyu Zhang x.zhang18@imperial.ac.uk\n\n'
        message += '-----------------------Running Parameters-----------------------\n'
        for key, value in sorted(vars(param).items()):
            comment = ''
            default = self.parser.get_default(key)
            if value != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>18}: {:<15}{}\n'.format(str(key), str(value), comment)
        message += '----------------------------------------------------------------\n'
        print(message)

        # Save the running parameters setting in the disk
        experiment_dir = os.path.join(param.checkpoints_dir, param.experiment_name)
        util.mkdir(experiment_dir)
        file_name = os.path.join(experiment_dir, 'cmd_parameters.txt')
        with open(file_name, 'w') as param_file:
            now = time.strftime('%c')
            param_file.write('{:s}\n'.format(now))
            param_file.write(message)
            param_file.write('\n')

    def parse(self):
        """
        Parse the parameters of our project. Set up GPU device. Print the welcome words and list parameters in the console.
        """
        param = self.get_params()  # get the parameters to the object param
        param.isTrain = self.isTrain
        param.isTest = self.isTest

        # Print welcome words and command line parameters
        self.print_params(param)

        # Set the internal parameters
        # epoch_num: the total epoch number
        if self.isTrain:
            param.epoch_num = param.epoch_num_p1 + param.epoch_num_p2 + param.epoch_num_p3
        # downstream_task: for the classification task a labels.tsv file is needed, for the regression task a values.tsv file is needed
        if param.model == 'vae_classifier':
            param.downstream_task = 'classification'
        elif param.model == 'vae_regression':
            param.downstream_task = 'regression'
        elif param.model == 'vae_survival':
            param.downstream_task = 'survival'
        elif param.model == 'vae_multitask' or param.model == 'vae_multitask_gn':
            param.downstream_task = 'multitask'
        elif param.model == 'vae_alltask' or param.model == 'vae_alltask_gn':
            param.downstream_task = 'alltask'
        else:
            raise NotImplementedError('Model name [%s] is not recognized' % param.model)
        # add_channel: add one extra dimension of channel for the input data, used for convolution layer
        # ch_separate: separate the DNA methylation matrix base on the chromosome
        if param.net_VAE == 'conv_1d':
            param.add_channel = True
            param.ch_separate = False
        elif param.net_VAE == 'fc_sep':
            param.add_channel = False
            param.ch_separate = True
        elif param.net_VAE == 'fc':
            param.add_channel = False
            param.ch_separate = False
        else:
            raise NotImplementedError('VAE model name [%s] is not recognized' % param.net_VAE)
        # omics_num: the number of omics types
        param.omics_num = len(param.omics_mode)

        # Set up GPU
        str_gpu_ids = param.gpu_ids.split(',')
        param.gpu_ids = []
        for str_gpu_id in str_gpu_ids:
            int_gpu_id = int(str_gpu_id)
            if int_gpu_id >= 0:
                param.gpu_ids.append(int_gpu_id)
        if len(param.gpu_ids) > 0:
            torch.cuda.set_device(param.gpu_ids[0])

        self.param = param
        return self.param
