import torch
from sklearn import metrics
from .vae_basic_model import VaeBasicModel
from . import networks
from . import losses


class VaeRegressionModel(VaeBasicModel):
    """
    This class implements the VAE regression model, using the VAE framework with the regression downstream task.
    """

    @staticmethod
    def modify_commandline_parameters(parser, is_train=True):
        # changing the default values of parameters to match the vae regression model
        parser.set_defaults(net_down='multi_FC_regression', not_stratified=True)
        parser.add_argument('--regression_scale', type=int, default=1,
                            help='normalization scale for y in regression task')
        parser.add_argument('--dist_loss', type=str, default='L1',
                            help='choose the distance loss for regression task, options: [MSE | L1]')
        return parser

    def __init__(self, param):
        """
        Initialize the VAE_regression class.
        """
        VaeBasicModel.__init__(self, param)
        # specify the training losses you want to print out.
        self.loss_names.append('distance')
        # specify the metrics you want to print out.
        self.metric_names = ['rmse']
        # input tensor
        self.value = None
        # output tensor
        self.y_out = None
        # define the network
        self.netDown = networks.define_down(param.net_down, param.norm_type, param.leaky_slope, param.dropout_p,
                                            param.latent_space_dim, None, None, None, param.init_type,
                                            param.init_gain, self.gpu_ids)
        # define the distance loss
        self.lossFuncDist = losses.get_loss_func(param.dist_loss, param.reduction)
        self.loss_distance = None
        self.metric_rmse = None

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
        self.value = input_dict['value'].to(self.device)

    def forward(self):
        VaeBasicModel.forward(self)
        # Get the output tensor
        self.y_out = self.netDown(self.latent)

    def cal_losses(self):
        """Calculate losses"""
        VaeBasicModel.cal_losses(self)
        # Calculate the regression distance loss (downstream loss)
        self.loss_distance = self.lossFuncDist(self.y_out.squeeze().type(torch.float32), (self.value / self.param.regression_scale).type(torch.float32))
        # LOSS DOWN
        self.loss_down = self.loss_distance

        self.loss_All = self.param.k_embed * self.loss_embed + self.loss_down

    def update(self):
        VaeBasicModel.update(self)

    def get_down_output(self):
        """
        Get output from downstream task
        """
        with torch.no_grad():
            index = self.data_index
            y_true = self.value
            y_pred = self.y_out * self.param.regression_scale

            return {'index': index, 'y_true': y_true, 'y_pred': y_pred}

    def calculate_current_metrics(self, output_dict):
        """
        Calculate current metrics
        """
        y_true = output_dict['y_true'].cpu().numpy()
        y_pred = output_dict['y_pred'].cpu().detach().numpy()

        self.metric_rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)

