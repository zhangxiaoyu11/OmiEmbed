import torch
from .vae_basic_model import VaeBasicModel
from . import networks
from . import losses
from torch.nn import functional as F
from sklearn import metrics


class VaeAlltaskModel(VaeBasicModel):
    """
    This class implements the VAE multitasking model with all downstream tasks (5 classifiers + 1 regressor + 1 survival predictor), using the VAE framework with the multiple downstream tasks.
    """
    @staticmethod
    def modify_commandline_parameters(parser, is_train=True):
        # Downstream task network
        parser.set_defaults(net_down='multi_FC_alltask')
        # Survival prediction related
        parser.add_argument('--survival_loss', type=str, default='MTLR', help='choose the survival loss')
        parser.add_argument('--survival_T_max', type=float, default=-1, help='maximum T value for survival prediction task')
        parser.add_argument('--time_num', type=int, default=256, help='number of time intervals in the survival model')
        # Classification related
        parser.add_argument('--class_num', type=int, default=0, help='the number of classes for the classification task')
        # Regression related
        parser.add_argument('--regression_scale', type=int, default=1, help='normalization scale for y in regression task')
        parser.add_argument('--dist_loss', type=str, default='L1', help='choose the distance loss for regression task, options: [MSE | L1]')
        # Loss combined
        parser.add_argument('--k_survival', type=float, default=1,
                            help='weight for the survival loss')
        parser.add_argument('--k_classifier', type=float, default=1,
                            help='weight for the classifier loss')
        parser.add_argument('--k_regression', type=float, default=1,
                            help='weight for the regression loss')
        # Number of tasks
        parser.add_argument('--task_num', type=int, default=7,
                            help='the number of downstream tasks')
        return parser

    def __init__(self, param):
        """
        Initialize the VAE_multitask class.
        """
        VaeBasicModel.__init__(self, param)
        # specify the training losses you want to print out.
        self.loss_names.extend(['survival', 'classifier_1', 'classifier_2', 'classifier_3', 'classifier_4', 'classifier_5', 'regression'])
        # specify the metrics you want to print out.
        self.metric_names = ['accuracy_1', 'accuracy_2', 'accuracy_3', 'accuracy_4', 'accuracy_5', 'rmse']
        # input tensor
        self.survival_T = None
        self.survival_E = None
        self.y_true = None
        self.label = None
        self.value = None
        # output tensor
        self.y_out_sur = None
        self.y_out_cla = None
        self.y_out_reg = None
        # define the network
        self.netDown = networks.define_down(param.net_down, param.norm_type, param.leaky_slope, param.dropout_p,
                                            param.latent_space_dim, param.class_num, param.time_num, param.task_num, param.init_type,
                                            param.init_gain, self.gpu_ids)
        # define the classification loss
        self.lossFuncClass = losses.get_loss_func('CE', param.reduction)
        # define the regression distance loss
        self.lossFuncDist = losses.get_loss_func(param.dist_loss, param.reduction)
        self.loss_survival = None
        self.loss_classifier_1 = None
        self.loss_classifier_2 = None
        self.loss_classifier_3 = None
        self.loss_classifier_4 = None
        self.loss_classifier_5 = None
        self.loss_regression = None
        self.metric_accuracy_1 = None
        self.metric_accuracy_2 = None
        self.metric_accuracy_3 = None
        self.metric_accuracy_4 = None
        self.metric_accuracy_5 = None
        self.metric_rmse = None

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
        self.label = []
        for i in range(self.param.task_num-2):
            self.label.append(input_dict['label'][i].to(self.device))
        self.value = input_dict['value'].to(self.device)

    def forward(self):
        # Get the output tensor
        VaeBasicModel.forward(self)
        self.y_out_sur, self.y_out_cla, self.y_out_reg = self.netDown(self.latent)

    def cal_losses(self):
        """Calculate losses"""
        VaeBasicModel.cal_losses(self)
        # Calculate the survival loss
        if self.param.survival_loss == 'MTLR':
            self.loss_survival = losses.MTLR_survival_loss(self.y_out_sur, self.y_true, self.survival_E, self.tri_matrix_1, self.param.reduction)
        # Calculate the classification loss
        self.loss_classifier_1 = self.lossFuncClass(self.y_out_cla[0], self.label[0])
        self.loss_classifier_2 = self.lossFuncClass(self.y_out_cla[1], self.label[1])
        self.loss_classifier_3 = self.lossFuncClass(self.y_out_cla[2], self.label[2])
        self.loss_classifier_4 = self.lossFuncClass(self.y_out_cla[3], self.label[3])
        self.loss_classifier_5 = self.lossFuncClass(self.y_out_cla[4], self.label[4])
        # Calculate the regression loss
        self.loss_regression = self.lossFuncDist(self.y_out_reg.squeeze().type(torch.float32), (self.value / self.param.regression_scale).type(torch.float32))
        # LOSS DOWN
        self.loss_down = self.param.k_survival * self.loss_survival + self.param.k_classifier * self.loss_classifier_1 + self.param.k_classifier * self.loss_classifier_2 + self.param.k_classifier * self.loss_classifier_3 + self.param.k_classifier * self.loss_classifier_4 + self.param.k_classifier * self.loss_classifier_5 + self.param.k_regression * self.loss_regression

        self.loss_All = self.param.k_embed * self.loss_embed + self.loss_down

    def update(self):
        VaeBasicModel.update(self)

    def get_down_output(self):
        """
        Get output from downstream task
        """
        with torch.no_grad():
            index = self.data_index
            # Survival
            y_true_E = self.survival_E
            y_true_T = self.survival_T
            y_out_sur = self.y_out_sur
            predict = self.predict_risk()
            # density = predict['density']
            survival = predict['survival']
            # hazard = predict['hazard']
            risk = predict['risk']

            # Classification
            y_prob_cla = []
            y_pred_cla = []
            y_true_cla = []
            for i in range(self.param.task_num-2):
                y_prob_cla.append(F.softmax(self.y_out_cla[i], dim=1))
                _, y_pred_cla_i = torch.max(y_prob_cla[i], 1)
                y_pred_cla.append(y_pred_cla_i)
                y_true_cla.append(self.label[i])

            # Regression
            y_true_reg = self.value
            y_pred_reg = self.y_out_reg * self.param.regression_scale

            return {'index': index, 'y_true_E': y_true_E, 'y_true_T': y_true_T, 'survival': survival, 'risk': risk, 'y_out_sur': y_out_sur, 'y_true_cla': y_true_cla, 'y_pred_cla': y_pred_cla, 'y_prob_cla': y_prob_cla, 'y_true_reg': y_true_reg, 'y_pred_reg': y_pred_reg}

    def calculate_current_metrics(self, output_dict):
        """
        Calculate current metrics
        """
        self.metric_accuracy_1 = (output_dict['y_true_cla'][0] == output_dict['y_pred_cla'][0]).sum().item() / len(output_dict['y_true_cla'][0])
        self.metric_accuracy_2 = (output_dict['y_true_cla'][1] == output_dict['y_pred_cla'][1]).sum().item() / len(output_dict['y_true_cla'][1])
        self.metric_accuracy_3 = (output_dict['y_true_cla'][2] == output_dict['y_pred_cla'][2]).sum().item() / len(output_dict['y_true_cla'][2])
        self.metric_accuracy_4 = (output_dict['y_true_cla'][3] == output_dict['y_pred_cla'][3]).sum().item() / len(output_dict['y_true_cla'][3])
        self.metric_accuracy_5 = (output_dict['y_true_cla'][4] == output_dict['y_pred_cla'][4]).sum().item() / len(output_dict['y_true_cla'][4])

        y_true_reg = output_dict['y_true_reg'].cpu().numpy()
        y_pred_reg = output_dict['y_pred_reg'].cpu().detach().numpy()
        self.metric_rmse = metrics.mean_squared_error(y_true_reg, y_pred_reg, squared=False)

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
            phi = torch.exp(torch.mm(self.y_out_sur, self.tri_matrix_1))
            div = torch.repeat_interleave(torch.sum(phi, 1).reshape(-1, 1), phi.shape[1], dim=1)

        density = phi / div
        survival = torch.mm(density, self.tri_matrix_2)
        hazard = density[:, :-1] / survival[:, 1:]

        cumulative_hazard = torch.cumsum(hazard, dim=1)
        risk = torch.sum(cumulative_hazard, 1)

        return {'density': density, 'survival': survival, 'hazard': hazard, 'risk': risk}
