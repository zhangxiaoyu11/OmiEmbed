import torch
import torch.nn as nn
from .basic_model import BasicModel
from . import networks
from . import losses
from torch.nn import functional as F
from sklearn import metrics


class VaeAlltaskGNModel(BasicModel):
    """
    This class implements the VAE multitasking model with GradNorm (all tasks), using the VAE framework with the multiple downstream tasks.
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
        # GradNorm ralated
        parser.add_argument('--alpha', type=float, default=1.5, help='the additional hyperparameter for GradNorm')
        parser.add_argument('--lr_gn', type=float, default=1e-3, help='the learning rate for GradNorm')
        parser.add_argument('--k_survival', type=float, default=1.0, help='initial weight for the survival loss')
        parser.add_argument('--k_classifier', type=float, default=1.0, help='initial weight for the classifier loss')
        parser.add_argument('--k_regression', type=float, default=1.0, help='initial weight for the regression loss')
        # Number of tasks
        parser.add_argument('--task_num', type=int, default=7, help='the number of downstream tasks')
        return parser

    def __init__(self, param):
        """
        Initialize the VAE_multitask class.
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
        self.loss_names.extend(['survival', 'classifier_1', 'classifier_2', 'classifier_3', 'classifier_4', 'classifier_5', 'regression', 'gradient', 'w_sur', 'w_cla_1', 'w_cla_2', 'w_cla_3', 'w_cla_4', 'w_cla_5', 'w_reg'])
        # specify the models you want to save to the disk and load.
        self.model_names = ['All']

        # input tensor
        self.input_omics = []
        self.data_index = None  # The indexes of input data
        self.survival_T = None
        self.survival_E = None
        self.y_true = None
        self.label = None
        self.value = None

        # output tensor
        self.z = None
        self.recon_omics = None
        self.mean = None
        self.log_var = None
        self.y_out_sur = None
        self.y_out_cla = None
        self.y_out_reg = None

        # specify the metrics you want to print out.
        self.metric_names = ['accuracy_1', 'accuracy_2', 'accuracy_3', 'accuracy_4', 'accuracy_5', 'rmse']

        # define the network
        self.netAll = networks.define_net(param.net_VAE, param.net_down, param.omics_dims, param.omics_mode,
                                          param.norm_type, param.filter_num, param.conv_k_size, param.leaky_slope,
                                          param.dropout_p, param.latent_space_dim, param.class_num, param.time_num, param.task_num,
                                          param.init_type, param.init_gain, self.gpu_ids)

        # define the reconstruction loss
        self.lossFuncRecon = losses.get_loss_func(param.recon_loss, param.reduction)
        # define the classification loss
        self.lossFuncClass = losses.get_loss_func('CE', param.reduction)
        # define the regression distance loss
        self.lossFuncDist = losses.get_loss_func(param.dist_loss, param.reduction)
        self.loss_recon_A = None
        self.loss_recon_B = None
        self.loss_recon_C = None
        self.loss_recon = None
        self.loss_kl = None
        self.loss_survival = None
        self.loss_classifier_1 = None
        self.loss_classifier_2 = None
        self.loss_classifier_3 = None
        self.loss_classifier_4 = None
        self.loss_classifier_5 = None
        self.loss_regression = None
        self.loss_gradient = 0

        self.loss_w_sur = None
        self.loss_w_cla_1 = None
        self.loss_w_cla_2 = None
        self.loss_w_cla_3 = None
        self.loss_w_cla_4 = None
        self.loss_w_cla_5 = None
        self.loss_w_reg = None

        self.task_losses = None
        self.weighted_losses = None
        self.initial_losses = None

        self.metric_accuracy_1 = None
        self.metric_accuracy_2 = None
        self.metric_accuracy_3 = None
        self.metric_accuracy_4 = None
        self.metric_accuracy_5 = None
        self.metric_rmse = None

        if param.survival_loss == 'MTLR':
            self.tri_matrix_1 = self.get_tri_matrix(dimension_type=1)
            self.tri_matrix_2 = self.get_tri_matrix(dimension_type=2)

        # Weights of multiple downstream tasks
        self.loss_weights = nn.Parameter(torch.ones(param.task_num, requires_grad=True, device=self.device))

        if self.isTrain:
            # Set the optimizer
            self.optimizer_All = torch.optim.Adam([{'params': self.netAll.parameters(), 'lr': param.lr, 'betas': (param.beta1, 0.999), 'weight_decay': param.weight_decay},
                                                  {'params': self.loss_weights, 'lr': param.lr_gn}])
            self.optimizers.append(self.optimizer_All)

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
        self.survival_T = input_dict['survival_T'].to(self.device)
        self.survival_E = input_dict['survival_E'].to(self.device)
        self.y_true = input_dict['y_true'].to(self.device)
        self.label = []
        for i in range(self.param.task_num - 2):
            self.label.append(input_dict['label'][i].to(self.device))
        self.value = input_dict['value'].to(self.device)

    def forward(self):
        # Get the output tensor
        self.z, self.recon_omics, self.mean, self.log_var, self.y_out_sur, self.y_out_cla, self.y_out_reg = self.netAll(self.input_omics)
        # define the latent
        self.latent = self.mean

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
        # Calculate the weighted downstream losses
        # Add initial weights
        self.task_losses = torch.stack([self.param.k_survival * self.loss_survival, self.param.k_classifier * self.loss_classifier_1, self.param.k_classifier * self.loss_classifier_2, self.param.k_classifier * self.loss_classifier_3, self.param.k_classifier * self.loss_classifier_4, self.param.k_classifier * self.loss_classifier_5, self.param.k_regression * self.loss_regression])
        self.weighted_losses = self.loss_weights * self.task_losses

        # LOSS DOWN
        self.loss_down = self.weighted_losses.sum()

        self.loss_All = self.param.k_embed * self.loss_embed + self.loss_down

        # Log the loss weights
        self.loss_w_sur = self.loss_weights[0] * self.param.k_survival
        self.loss_w_cla_1 = self.loss_weights[1] * self.param.k_classifier
        self.loss_w_cla_2 = self.loss_weights[2] * self.param.k_classifier
        self.loss_w_cla_3 = self.loss_weights[3] * self.param.k_classifier
        self.loss_w_cla_4 = self.loss_weights[4] * self.param.k_classifier
        self.loss_w_cla_5 = self.loss_weights[5] * self.param.k_classifier
        self.loss_w_reg = self.loss_weights[6] * self.param.k_regression

    def update(self):
        if self.phase == 'p1':
            self.forward()
            self.optimizer_All.zero_grad()  # Set gradients to zero
            self.cal_losses()  # Calculate losses
            self.loss_embed.backward()  # Backpropagation
            self.optimizer_All.step()  # Update weights
        elif self.phase == 'p2':
            self.forward()
            self.optimizer_All.zero_grad()  # Set gradients to zero
            self.cal_losses()  # Calculate losses
            self.loss_down.backward()  # Backpropagation
            self.optimizer_All.step()  # Update weights
        elif self.phase == 'p3':
            self.forward()
            self.cal_losses()  # Calculate losses
            self.optimizer_All.zero_grad()  # Set gradients to zero

            # Calculate the GradNorm gradients
            if isinstance(self.netAll, torch.nn.DataParallel):
                W = list(self.netAll.module.get_last_encode_layer().parameters())
            else:
                W = list(self.netAll.get_last_encode_layer().parameters())
            grad_norms = []
            for weight, loss in zip(self.loss_weights, self.task_losses):
                grad = torch.autograd.grad(loss, W, retain_graph=True)
                grad_norms.append(torch.norm(weight * grad[0]))
            grad_norms = torch.stack(grad_norms)

            if self.iter == 0:
                self.initial_losses = self.task_losses.detach()

            # Calculate the constant targets
            with torch.no_grad():
                # loss ratios
                loss_ratios = self.task_losses / self.initial_losses
                # inverse training rate
                inverse_train_rates = loss_ratios / loss_ratios.mean()
                constant_terms = grad_norms.mean() * (inverse_train_rates ** self.param.alpha)

            # Calculate the gradient loss
            self.loss_gradient = (grad_norms - constant_terms).abs().sum()
            # Set the gradients of weights
            loss_weights_grad = torch.autograd.grad(self.loss_gradient, self.loss_weights)[0]

            self.loss_All.backward()

            self.loss_weights.grad = loss_weights_grad

            self.optimizer_All.step()  # Update weights

            # Re-normalize the losses weights
            with torch.no_grad():
                normalize_coeff = len(self.loss_weights) / self.loss_weights.sum()
                self.loss_weights.data = self.loss_weights.data * normalize_coeff

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
            for i in range(self.param.task_num - 2):
                y_prob_cla.append(F.softmax(self.y_out_cla[i], dim=1))
                _, y_pred_cla_i = torch.max(y_prob_cla[i], 1)
                y_pred_cla.append(y_pred_cla_i)
                y_true_cla.append(self.label[i])

            # Regression
            y_true_reg = self.value
            y_pred_reg = self.y_out_reg * self.param.regression_scale

            return {'index': index, 'y_true_E': y_true_E, 'y_true_T': y_true_T, 'survival': survival, 'risk': risk,
                    'y_out_sur': y_out_sur, 'y_true_cla': y_true_cla, 'y_pred_cla': y_pred_cla,
                    'y_prob_cla': y_prob_cla, 'y_true_reg': y_true_reg, 'y_pred_reg': y_pred_reg}

    def calculate_current_metrics(self, output_dict):
        """
        Calculate current metrics
        """
        self.metric_accuracy_1 = (output_dict['y_true_cla'][0] == output_dict['y_pred_cla'][0]).sum().item() / len(
            output_dict['y_true_cla'][0])
        self.metric_accuracy_2 = (output_dict['y_true_cla'][1] == output_dict['y_pred_cla'][1]).sum().item() / len(
            output_dict['y_true_cla'][1])
        self.metric_accuracy_3 = (output_dict['y_true_cla'][2] == output_dict['y_pred_cla'][2]).sum().item() / len(
            output_dict['y_true_cla'][2])
        self.metric_accuracy_4 = (output_dict['y_true_cla'][3] == output_dict['y_pred_cla'][3]).sum().item() / len(
            output_dict['y_true_cla'][3])
        self.metric_accuracy_5 = (output_dict['y_true_cla'][4] == output_dict['y_pred_cla'][4]).sum().item() / len(
            output_dict['y_true_cla'][4])

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
