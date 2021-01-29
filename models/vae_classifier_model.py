import torch
from .vae_basic_model import VaeBasicModel
from . import networks
from . import losses
from torch.nn import functional as F


class VaeClassifierModel(VaeBasicModel):
    """
    This class implements the VAE classifier model, using the VAE framework with the classification downstream task.
    """

    @staticmethod
    def modify_commandline_parameters(parser, is_train=True):
        # changing the default values of parameters to match the vae regression model
        parser.add_argument('--class_num', type=int, default=0,
                            help='the number of classes for the classification task')
        return parser

    def __init__(self, param):
        """
        Initialize the VAE_classifier class.
        """
        VaeBasicModel.__init__(self, param)
        # specify the training losses you want to print out.
        self.loss_names.append('classifier')
        # specify the metrics you want to print out.
        self.metric_names = ['accuracy']
        # input tensor
        self.label = None
        # output tensor
        self.y_out = None
        # define the network
        self.netDown = networks.define_down(param.net_down, param.norm_type, param.leaky_slope, param.dropout_p,
                                            param.latent_space_dim, param.class_num, None, None, param.init_type,
                                            param.init_gain, self.gpu_ids)
        # define the classification loss
        self.lossFuncClass = losses.get_loss_func('CE', param.reduction)
        self.loss_classifier = None
        self.metric_accuracy = None

        if self.isTrain:
            # Set the optimizer
            self.optimizer_Down = torch.optim.Adam(self.netDown.parameters(), lr=param.lr, betas=(param.beta1, 0.999), weight_decay=param.weight_decay)
            # optimizer list was already defined in BaseModel
            self.optimizers.append(self.optimizer_Down)

    def set_input(self, input_dict):
        """
        Unpack input data from the output dictionary of the dataloader

        Parameters:
            input_dict (dict): include the data tensor and its index.
        """
        VaeBasicModel.set_input(self, input_dict)
        self.label = input_dict['label'].to(self.device)

    def forward(self):
        VaeBasicModel.forward(self)
        # Get the output tensor
        self.y_out = self.netDown(self.latent)

    def cal_losses(self):
        """Calculate losses"""
        VaeBasicModel.cal_losses(self)
        # Calculate the classification loss (downstream loss)
        self.loss_classifier = self.lossFuncClass(self.y_out, self.label)
        # LOSS DOWN
        self.loss_down = self.loss_classifier

        self.loss_All = self.param.k_embed * self.loss_embed + self.loss_down

    def update(self):
        VaeBasicModel.update(self)

    def get_down_output(self):
        """
        Get output from downstream task
        """
        with torch.no_grad():
            y_prob = F.softmax(self.y_out, dim=1)
            _, y_pred = torch.max(y_prob, 1)

            index = self.data_index
            y_true = self.label

            return {'index': index, 'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob}

    def calculate_current_metrics(self, output_dict):
        """
        Calculate current metrics
        """
        self.metric_accuracy = (output_dict['y_true'] == output_dict['y_pred']).sum().item() / len(output_dict['y_true'])
