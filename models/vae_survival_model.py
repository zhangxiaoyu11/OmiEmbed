import torch
from .vae_basic_model import VaeBasicModel
from . import networks
from . import losses


class VaeSurvivalModel(VaeBasicModel):
    """
    This class implements the VAE survival model, using the VAE framework with the survival prediction downstream task.
    """

    @staticmethod
    def modify_commandline_parameters(parser, is_train=True):
        # changing the default values of parameters to match the vae survival prediction model
        parser.set_defaults(net_down='multi_FC_survival')
        parser.add_argument('--survival_loss', type=str, default='MTLR', help='choose the survival loss')
        parser.add_argument('--survival_T_max', type=float, default=-1, help='maximum T value for survival prediction task')
        parser.add_argument('--time_num', type=int, default=256, help='number of time intervals in the survival model')
        parser.add_argument('--stratify_label', action='store_true', help='load extra label for stratified dataset separation')
        return parser

    def __init__(self, param):
        """
        Initialize the VAE_survival class.
        """
        VaeBasicModel.__init__(self, param)
        # specify the training losses you want to print out.
        self.loss_names.append('survival')
        # specify the metrics you want to print out.
        self.metric_names = []
        # input tensor
        self.survival_T = None
        self.survival_E = None
        self.y_true = None
        # output tensor
        self.y_out = None
        # define the network
        self.netDown = networks.define_down(param.net_down, param.norm_type, param.leaky_slope, param.dropout_p,
                                            param.latent_space_dim, None, param.time_num, None, param.init_type,
                                            param.init_gain, self.gpu_ids)
        self.loss_survival = None

        if param.survival_loss == 'MTLR':
            self.tri_matrix_1 = self.get_tri_matrix(dimension_type=1)
            self.tri_matrix_2 = self.get_tri_matrix(dimension_type=2)

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
        self.survival_T = input_dict['survival_T'].to(self.device)
        self.survival_E = input_dict['survival_E'].to(self.device)
        self.y_true = input_dict['y_true'].to(self.device)

    def forward(self):
        VaeBasicModel.forward(self)
        # Get the output tensor
        self.y_out = self.netDown(self.latent)

    def cal_losses(self):
        """Calculate losses"""
        VaeBasicModel.cal_losses(self)
        # Calculate the survival loss (downstream loss)
        if self.param.survival_loss == 'MTLR':
            self.loss_survival = losses.MTLR_survival_loss(self.y_out, self.y_true, self.survival_E, self.tri_matrix_1, self.param.reduction)
        # LOSS DOWN
        self.loss_down = self.loss_survival

        self.loss_All = self.param.k_embed * self.loss_embed + self.loss_down

    def update(self):
        VaeBasicModel.update(self)

    def get_down_output(self):
        """
        Get output from downstream task
        """
        with torch.no_grad():
            index = self.data_index
            y_true_E = self.survival_E
            y_true_T = self.survival_T
            y_out = self.y_out

            predict = self.predict_risk()
            # density = predict['density']
            survival = predict['survival']
            # hazard = predict['hazard']
            risk = predict['risk']

            return {'index': index, 'y_true_E': y_true_E, 'y_true_T': y_true_T, 'survival': survival, 'risk': risk, 'y_out': y_out}

    def calculate_current_metrics(self, output_dict):
        """
        Calculate current metrics
        """
        pass

    def get_tri_matrix(self, dimension_type=1):
        """
        Get tensor of the triangular matrix
        """
        if dimension_type == 1:
            ones_matrix = torch.ones(self.param.time_num, self.param.time_num + 1, device=self.device)
        else:
            ones_matrix = torch.ones(self.param.time_num + 1, self.param.time_num + 1, device=self.device)
        tri_matrix = torch.tril(ones_matrix)
        return tri_matrix

    def predict_risk(self):
        """
        Predict the density, survival and hazard function, as well as the risk score
        """
        if self.param.survival_loss == 'MTLR':
            phi = torch.exp(torch.mm(self.y_out, self.tri_matrix_1))
            div = torch.repeat_interleave(torch.sum(phi, 1).reshape(-1, 1), phi.shape[1], dim=1)

        density = phi / div
        survival = torch.mm(density, self.tri_matrix_2)
        hazard = density[:, :-1] / survival[:, 1:]

        cumulative_hazard = torch.cumsum(hazard, dim=1)
        risk = torch.sum(cumulative_hazard, 1)

        return {'density': density, 'survival': survival, 'hazard': hazard, 'risk': risk}
