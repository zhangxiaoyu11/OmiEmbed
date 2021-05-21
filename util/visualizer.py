import os
import time
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import label_binarize
from util import util
from util import metrics
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """
    This class print/save logging information
    """

    def __init__(self, param):
        """
        Initialize the Visualizer class
        """
        self.param = param
        self.output_path = os.path.join(param.checkpoints_dir, param.experiment_name)
        tb_dir = os.path.join(self.output_path, 'tb_log')
        util.mkdir(tb_dir)

        if param.isTrain:
            # Create a logging file to store training losses
            self.train_log_filename = os.path.join(self.output_path, 'train_log.txt')
            with open(self.train_log_filename, 'a') as log_file:
                now = time.strftime('%c')
                log_file.write('----------------------- Training Log ({:s}) -----------------------\n'.format(now))

            self.train_summary_filename = os.path.join(self.output_path, 'train_summary.txt')
            with open(self.train_summary_filename, 'a') as log_file:
                now = time.strftime('%c')
                log_file.write('----------------------- Training Summary ({:s}) -----------------------\n'.format(now))

            # Create log folder for TensorBoard
            tb_train_dir = os.path.join(self.output_path, 'tb_log', 'train')
            util.mkdir(tb_train_dir)
            util.clear_dir(tb_train_dir)

            # Create TensorBoard writer
            self.train_writer = SummaryWriter(log_dir=tb_train_dir)

        if param.isTest:
            # Create a logging file to store testing metrics
            self.test_log_filename = os.path.join(self.output_path, 'test_log.txt')
            with open(self.test_log_filename, 'a') as log_file:
                now = time.strftime('%c')
                log_file.write('----------------------- Testing Log ({:s}) -----------------------\n'.format(now))

            self.test_summary_filename = os.path.join(self.output_path, 'test_summary.txt')
            with open(self.test_summary_filename, 'a') as log_file:
                now = time.strftime('%c')
                log_file.write('----------------------- Testing Summary ({:s}) -----------------------\n'.format(now))

            # Create log folder for TensorBoard
            tb_test_dir = os.path.join(self.output_path, 'tb_log', 'test')
            util.mkdir(tb_test_dir)
            util.clear_dir(tb_test_dir)

            # Create TensorBoard writer
            self.test_writer = SummaryWriter(log_dir=tb_test_dir)

    def print_train_log(self, epoch, iteration, losses_dict, metrics_dict, load_time, comp_time, batch_size, dataset_size, with_time=True):
        """
        print train log on console and save the message to the disk

        Parameters:
            epoch (int)                     -- current epoch
            iteration (int)                 -- current training iteration during this epoch
            losses_dict (OrderedDict)       -- training losses stored in the ordered dict
            metrics_dict (OrderedDict)      -- metrics stored in the ordered dict
            load_time (float)               -- data loading time per data point (normalized by batch_size)
            comp_time (float)               -- computational time per data point (normalized by batch_size)
            batch_size (int)                -- batch size of training
            dataset_size (int)              -- size of the training dataset
            with_time (bool)                -- print the running time or not
        """
        data_point_covered = min((iteration + 1) * batch_size, dataset_size)
        if with_time:
            message = '[TRAIN] [Epoch: {:3d}   Iter: {:4d}   Load_t: {:.3f}   Comp_t: {:.3f}]   '.format(epoch, data_point_covered, load_time, comp_time)
        else:
            message = '[TRAIN] [Epoch: {:3d}   Iter: {:4d}]\n'.format(epoch, data_point_covered)
        for name, loss in losses_dict.items():
            message += '{:s}: {:.3f}   '.format(name, loss[-1])
        for name, metric in metrics_dict.items():
            message += '{:s}: {:.3f}   '.format(name, metric)

        print(message)  # print the message

        with open(self.train_log_filename, 'a') as log_file:
            log_file.write(message + '\n')  # save the message

    def print_train_summary(self, epoch, losses_dict, output_dict, train_time, current_lr):
        """
        print the summary of this training epoch

        Parameters:
            epoch (int)                             -- epoch number of this training model
            losses_dict (OrderedDict)               -- the losses dictionary
            output_dict (OrderedDict)               -- the downstream output dictionary
            train_time (float)                      -- time used for training this epoch
            current_lr (float)                      -- the learning rate of this epoch
        """
        write_message = '{:s}\t'.format(str(epoch))
        print_message = '[TRAIN] [Epoch: {:3d}]\n'.format(int(epoch))

        for name, loss in losses_dict.items():
            write_message += '{:.6f}\t'.format(np.mean(loss))
            print_message += name + ': {:.3f}   '.format(np.mean(loss))
            self.train_writer.add_scalar('loss_'+name, np.mean(loss), epoch)

        metrics_dict = self.get_epoch_metrics(output_dict)
        for name, metric in metrics_dict.items():
            write_message += '{:.6f}\t'.format(metric)
            print_message += name + ': {:.3f}   '.format(metric)
            self.train_writer.add_scalar('metric_'+name, metric, epoch)

        train_time_msg = 'Training time used: {:.3f}s'.format(train_time)
        print_message += '\n' + train_time_msg
        with open(self.train_log_filename, 'a') as log_file:
            log_file.write(train_time_msg + '\n')

        current_lr_msg = 'Learning rate for this epoch: {:.7f}'.format(current_lr)
        print_message += '\n' + current_lr_msg
        self.train_writer.add_scalar('lr', current_lr, epoch)

        with open(self.train_summary_filename, 'a') as log_file:
            log_file.write(write_message + '\n')

        print(print_message)

    def print_test_log(self, epoch, iteration, losses_dict, metrics_dict, batch_size, dataset_size):
        """
        print performance metrics of this iteration on console and save the message to the disk

        Parameters:
            epoch (int)                     -- epoch number of this testing model
            iteration (int)                 -- current testing iteration during this epoch
            losses_dict (OrderedDict)       -- training losses stored in the ordered dict
            metrics_dict (OrderedDict)      -- metrics stored in the ordered dict
            batch_size (int)                -- batch size of testing
            dataset_size (int)              -- size of the testing dataset
        """
        data_point_covered = min((iteration + 1) * batch_size, dataset_size)
        message = '[TEST] [Epoch: {:3d}   Iter: {:4d}]   '.format(int(epoch), data_point_covered)
        for name, loss in losses_dict.items():
            message += '{:s}: {:.3f}   '.format(name, loss[-1])
        for name, metric in metrics_dict.items():
            message += '{:s}: {:.3f}   '.format(name, metric)

        print(message)

        with open(self.test_log_filename, 'a') as log_file:
            log_file.write(message + '\n')

    def print_test_summary(self, epoch, losses_dict, output_dict, test_time):
        """
        print the summary of this testing epoch

        Parameters:
            epoch (int)                             -- epoch number of this testing model
            losses_dict (OrderedDict)               -- the losses dictionary
            output_dict (OrderedDict)               -- the downstream output dictionary
            test_time (float)                       -- time used for testing this epoch
        """
        write_message = '{:s}\t'.format(str(epoch))
        print_message = '[TEST] [Epoch: {:3d}]      '.format(int(epoch))

        for name, loss in losses_dict.items():
            # write_message += '{:.6f}\t'.format(np.mean(loss))
            print_message += name + ': {:.3f}   '.format(np.mean(loss))
            self.test_writer.add_scalar('loss_'+name, np.mean(loss), epoch)

        metrics_dict = self.get_epoch_metrics(output_dict)

        for name, metric in metrics_dict.items():
            write_message += '{:.6f}\t'.format(metric)
            print_message += name + ': {:.3f}   '.format(metric)
            self.test_writer.add_scalar('metric_' + name, metric, epoch)

        with open(self.test_summary_filename, 'a') as log_file:
            log_file.write(write_message + '\n')

        test_time_msg = 'Testing time used: {:.3f}s'.format(test_time)
        print_message += '\n' + test_time_msg
        print(print_message)
        with open(self.test_log_filename, 'a') as log_file:
            log_file.write(test_time_msg + '\n')

    def get_epoch_metrics(self, output_dict):
        """
        Get the downstream task metrics for whole epoch

        Parameters:
            output_dict (OrderedDict)  -- the output dictionary used to compute the downstream task metrics
        """
        if self.param.downstream_task == 'classification':
            y_true = output_dict['y_true'].cpu().numpy()
            y_true_binary = label_binarize(y_true, classes=range(self.param.class_num))
            y_pred = output_dict['y_pred'].cpu().numpy()
            y_prob = output_dict['y_prob'].cpu().numpy()
            if self.param.class_num == 2:
                y_prob = y_prob[:, 1]

            accuracy = sk.metrics.accuracy_score(y_true, y_pred)
            precision = sk.metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = sk.metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = sk.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
            try:
                auc = sk.metrics.roc_auc_score(y_true_binary, y_prob, multi_class='ovo', average='macro')
            except ValueError:
                auc = -1
                print('ValueError: ROC AUC score is not defined in this case.')

            return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

        elif self.param.downstream_task == 'regression':
            y_true = output_dict['y_true'].cpu().numpy()
            y_pred = output_dict['y_pred'].cpu().detach().numpy()

            mse = sk.metrics.mean_squared_error(y_true, y_pred)
            rmse = sk.metrics.mean_squared_error(y_true, y_pred, squared=False)
            mae = sk.metrics.mean_absolute_error(y_true, y_pred)
            medae = sk.metrics.median_absolute_error(y_true, y_pred)
            r2 = sk.metrics.r2_score(y_true, y_pred)

            return {'mse': mse, 'rmse': rmse, 'mae': mae, 'medae': medae, 'r2': r2}

        elif self.param.downstream_task == 'survival':
            metrics_start_time = time.time()

            y_true_E = output_dict['y_true_E'].cpu().numpy()
            y_true_T = output_dict['y_true_T'].cpu().numpy()
            y_pred_risk = output_dict['risk'].cpu().numpy()
            y_pred_survival = output_dict['survival'].cpu().numpy()

            time_points = util.get_time_points(self.param.survival_T_max, self.param.time_num)

            try:
                c_index = metrics.c_index(y_true_T, y_true_E, y_pred_risk)
            except ValueError:
                c_index = -1
                print('ValueError: NaNs detected in input when calculating c-index.')

            try:
                ibs = metrics.ibs(y_true_T, y_true_E, y_pred_survival, time_points)
            except ValueError:
                ibs = -1
                print('ValueError: NaNs detected in input when calculating integrated brier score.')

            metrics_time = time.time() - metrics_start_time
            print('Metrics computing time: {:.3f}s'.format(metrics_time))

            return {'c-index': c_index, 'ibs': ibs}

        elif self.param.downstream_task == 'multitask':
            metrics_start_time = time.time()

            # Survival
            y_true_E = output_dict['y_true_E'].cpu().numpy()
            y_true_T = output_dict['y_true_T'].cpu().numpy()
            y_pred_risk = output_dict['risk'].cpu().numpy()
            y_pred_survival = output_dict['survival'].cpu().numpy()
            time_points = util.get_time_points(self.param.survival_T_max, self.param.time_num)
            try:
                c_index = metrics.c_index(y_true_T, y_true_E, y_pred_risk)
            except ValueError:
                c_index = -1
                print('ValueError: NaNs detected in input when calculating c-index.')
            try:
                ibs = metrics.ibs(y_true_T, y_true_E, y_pred_survival, time_points)
            except ValueError:
                ibs = -1
                print('ValueError: NaNs detected in input when calculating integrated brier score.')

            # Classification
            y_true_cla = output_dict['y_true_cla'].cpu().numpy()
            y_true_cla_binary = label_binarize(y_true_cla, classes=range(self.param.class_num))
            y_pred_cla = output_dict['y_pred_cla'].cpu().numpy()
            y_prob_cla = output_dict['y_prob_cla'].cpu().numpy()
            if self.param.class_num == 2:
                y_prob_cla = y_prob_cla[:, 1]
            accuracy = sk.metrics.accuracy_score(y_true_cla, y_pred_cla)
            precision = sk.metrics.precision_score(y_true_cla, y_pred_cla, average='macro', zero_division=0)
            recall = sk.metrics.recall_score(y_true_cla, y_pred_cla, average='macro', zero_division=0)
            f1 = sk.metrics.f1_score(y_true_cla, y_pred_cla, average='macro', zero_division=0)
            '''
            try:
                auc = sk.metrics.roc_auc_score(y_true_cla_binary, y_prob_cla, multi_class='ovo', average='macro')
            except ValueError:
                auc = -1
                print('ValueError: ROC AUC score is not defined in this case.')
            '''

            # Regression
            y_true_reg = output_dict['y_true_reg'].cpu().numpy()
            y_pred_reg = output_dict['y_pred_reg'].cpu().detach().numpy()
            # mse = sk.metrics.mean_squared_error(y_true_reg, y_pred_reg)
            rmse = sk.metrics.mean_squared_error(y_true_reg, y_pred_reg, squared=False)
            mae = sk.metrics.mean_absolute_error(y_true_reg, y_pred_reg)
            medae = sk.metrics.median_absolute_error(y_true_reg, y_pred_reg)
            r2 = sk.metrics.r2_score(y_true_reg, y_pred_reg)

            metrics_time = time.time() - metrics_start_time
            print('Metrics computing time: {:.3f}s'.format(metrics_time))

            return {'c-index': c_index, 'ibs': ibs, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'rmse': rmse, 'mae': mae, 'medae': medae, 'r2': r2}

        elif self.param.downstream_task == 'alltask':
            metrics_start_time = time.time()

            # Survival
            y_true_E = output_dict['y_true_E'].cpu().numpy()
            y_true_T = output_dict['y_true_T'].cpu().numpy()
            y_pred_risk = output_dict['risk'].cpu().numpy()
            y_pred_survival = output_dict['survival'].cpu().numpy()
            time_points = util.get_time_points(self.param.survival_T_max, self.param.time_num)
            try:
                c_index = metrics.c_index(y_true_T, y_true_E, y_pred_risk)
            except ValueError:
                c_index = -1
                print('ValueError: NaNs detected in input when calculating c-index.')
            try:
                ibs = metrics.ibs(y_true_T, y_true_E, y_pred_survival, time_points)
            except ValueError:
                ibs = -1
                print('ValueError: NaNs detected in input when calculating integrated brier score.')

            # Classification
            accuracy = []
            f1 = []
            auc = []
            for i in range(self.param.task_num - 2):
                y_true_cla = output_dict['y_true_cla'][i].cpu().numpy()
                y_true_cla_binary = label_binarize(y_true_cla, classes=range(self.param.class_num[i]))
                y_pred_cla = output_dict['y_pred_cla'][i].cpu().numpy()
                y_prob_cla = output_dict['y_prob_cla'][i].cpu().numpy()
                if self.param.class_num[i] == 2:
                    y_prob_cla = y_prob_cla[:, 1]
                accuracy.append(sk.metrics.accuracy_score(y_true_cla, y_pred_cla))
                f1.append(sk.metrics.f1_score(y_true_cla, y_pred_cla, average='macro', zero_division=0))
                try:
                    auc.append(sk.metrics.roc_auc_score(y_true_cla_binary, y_prob_cla, multi_class='ovo', average='macro'))
                except ValueError:
                    auc.append(-1)
                    print('ValueError: ROC AUC score is not defined in this case.')

            # Regression
            y_true_reg = output_dict['y_true_reg'].cpu().numpy()
            y_pred_reg = output_dict['y_pred_reg'].cpu().detach().numpy()
            # mse = sk.metrics.mean_squared_error(y_true_reg, y_pred_reg)
            rmse = sk.metrics.mean_squared_error(y_true_reg, y_pred_reg, squared=False)
            # mae = sk.metrics.mean_absolute_error(y_true_reg, y_pred_reg)
            # medae = sk.metrics.median_absolute_error(y_true_reg, y_pred_reg)
            r2 = sk.metrics.r2_score(y_true_reg, y_pred_reg)

            metrics_time = time.time() - metrics_start_time
            print('Metrics computing time: {:.3f}s'.format(metrics_time))

            return {'c-index': c_index, 'ibs': ibs, 'accuracy_1': accuracy[0], 'f1_1': f1[0], 'auc_1': auc[0], 'accuracy_2': accuracy[1], 'f1_2': f1[1], 'auc_2': auc[1], 'accuracy_3': accuracy[2], 'f1_3': f1[2], 'auc_3': auc[2], 'accuracy_4': accuracy[3], 'f1_4': f1[3], 'auc_4': auc[3], 'accuracy_5': accuracy[4], 'f1_5': f1[4], 'auc_5': auc[4], 'rmse': rmse, 'r2': r2}

    def save_output_dict(self, output_dict):
        """
        Save the downstream task output to disk

        Parameters:
            output_dict (OrderedDict)  -- the downstream task output dictionary to be saved
        """
        down_path = os.path.join(self.output_path, 'down_output')
        util.mkdir(down_path)
        if self.param.downstream_task == 'classification':
            # Prepare files
            index = output_dict['index'].numpy()
            y_true = output_dict['y_true'].cpu().numpy()
            y_pred = output_dict['y_pred'].cpu().numpy()
            y_prob = output_dict['y_prob'].cpu().numpy()

            sample_list = self.param.sample_list[index]

            # Output files
            y_df = pd.DataFrame({'sample': sample_list, 'y_true': y_true, 'y_pred': y_pred}, index=index)
            y_df_path = os.path.join(down_path, 'y_df.tsv')
            y_df.to_csv(y_df_path, sep='\t')

            prob_df = pd.DataFrame(y_prob, columns=range(self.param.class_num), index=sample_list)
            y_prob_path = os.path.join(down_path, 'y_prob.tsv')
            prob_df.to_csv(y_prob_path, sep='\t')

        elif self.param.downstream_task == 'regression':
            # Prepare files
            index = output_dict['index'].numpy()
            y_true = output_dict['y_true'].cpu().numpy()
            y_pred = np.squeeze(output_dict['y_pred'].cpu().detach().numpy())

            sample_list = self.param.sample_list[index]

            # Output files
            y_df = pd.DataFrame({'sample': sample_list, 'y_true': y_true, 'y_pred': y_pred}, index=index)
            y_df_path = os.path.join(down_path, 'y_df.tsv')
            y_df.to_csv(y_df_path, sep='\t')

        elif self.param.downstream_task == 'survival':
            # Prepare files
            index = output_dict['index'].numpy()
            y_true_E = output_dict['y_true_E'].cpu().numpy()
            y_true_T = output_dict['y_true_T'].cpu().numpy()
            y_pred_risk = output_dict['risk'].cpu().numpy()
            survival_function = output_dict['survival'].cpu().numpy()
            y_out = output_dict['y_out'].cpu().numpy()

            sample_list = self.param.sample_list[index]
            time_points = util.get_time_points(self.param.survival_T_max, self.param.time_num)

            # Output files
            y_df = pd.DataFrame({'sample': sample_list, 'true_T': y_true_T, 'true_E': y_true_E, 'pred_risk': y_pred_risk}, index=index)
            y_df_path = os.path.join(down_path, 'y_df.tsv')
            y_df.to_csv(y_df_path, sep='\t')

            survival_function_df = pd.DataFrame(survival_function, columns=time_points, index=sample_list)
            survival_function_path = os.path.join(down_path, 'survival_function.tsv')
            survival_function_df.to_csv(survival_function_path, sep='\t')

            y_out_df = pd.DataFrame(y_out, index=sample_list)
            y_out_path = os.path.join(down_path, 'y_out.tsv')
            y_out_df.to_csv(y_out_path, sep='\t')

        elif self.param.downstream_task == 'multitask':
            # Survival
            index = output_dict['index'].numpy()
            y_true_E = output_dict['y_true_E'].cpu().numpy()
            y_true_T = output_dict['y_true_T'].cpu().numpy()
            y_pred_risk = output_dict['risk'].cpu().numpy()
            survival_function = output_dict['survival'].cpu().numpy()
            y_out_sur = output_dict['y_out_sur'].cpu().numpy()
            sample_list = self.param.sample_list[index]
            time_points = util.get_time_points(self.param.survival_T_max, self.param.time_num)
            y_df_sur = pd.DataFrame(
                {'sample': sample_list, 'true_T': y_true_T, 'true_E': y_true_E, 'pred_risk': y_pred_risk}, index=index)
            y_df_sur_path = os.path.join(down_path, 'y_df_survival.tsv')
            y_df_sur.to_csv(y_df_sur_path, sep='\t')
            survival_function_df = pd.DataFrame(survival_function, columns=time_points, index=sample_list)
            survival_function_path = os.path.join(down_path, 'survival_function.tsv')
            survival_function_df.to_csv(survival_function_path, sep='\t')
            y_out_sur_df = pd.DataFrame(y_out_sur, index=sample_list)
            y_out_sur_path = os.path.join(down_path, 'y_out_survival.tsv')
            y_out_sur_df.to_csv(y_out_sur_path, sep='\t')

            # Classification
            y_true_cla = output_dict['y_true_cla'].cpu().numpy()
            y_pred_cla = output_dict['y_pred_cla'].cpu().numpy()
            y_prob_cla = output_dict['y_prob_cla'].cpu().numpy()
            y_df_cla = pd.DataFrame({'sample': sample_list, 'y_true': y_true_cla, 'y_pred': y_pred_cla}, index=index)
            y_df_cla_path = os.path.join(down_path, 'y_df_classification.tsv')
            y_df_cla.to_csv(y_df_cla_path, sep='\t')
            prob_cla_df = pd.DataFrame(y_prob_cla, columns=range(self.param.class_num), index=sample_list)
            y_prob_cla_path = os.path.join(down_path, 'y_prob_classification.tsv')
            prob_cla_df.to_csv(y_prob_cla_path, sep='\t')

            # Regression
            y_true_reg = output_dict['y_true_reg'].cpu().numpy()
            y_pred_reg = np.squeeze(output_dict['y_pred_reg'].cpu().detach().numpy())
            y_df_reg = pd.DataFrame({'sample': sample_list, 'y_true': y_true_reg, 'y_pred': y_pred_reg}, index=index)
            y_df_reg_path = os.path.join(down_path, 'y_df_regression.tsv')
            y_df_reg.to_csv(y_df_reg_path, sep='\t')

        elif self.param.downstream_task == 'alltask':
            # Survival
            index = output_dict['index'].numpy()
            y_true_E = output_dict['y_true_E'].cpu().numpy()
            y_true_T = output_dict['y_true_T'].cpu().numpy()
            y_pred_risk = output_dict['risk'].cpu().numpy()
            survival_function = output_dict['survival'].cpu().numpy()
            y_out_sur = output_dict['y_out_sur'].cpu().numpy()
            sample_list = self.param.sample_list[index]
            time_points = util.get_time_points(self.param.survival_T_max, self.param.time_num)
            y_df_sur = pd.DataFrame(
                {'sample': sample_list, 'true_T': y_true_T, 'true_E': y_true_E, 'pred_risk': y_pred_risk}, index=index)
            y_df_sur_path = os.path.join(down_path, 'y_df_survival.tsv')
            y_df_sur.to_csv(y_df_sur_path, sep='\t')
            survival_function_df = pd.DataFrame(survival_function, columns=time_points, index=sample_list)
            survival_function_path = os.path.join(down_path, 'survival_function.tsv')
            survival_function_df.to_csv(survival_function_path, sep='\t')
            y_out_sur_df = pd.DataFrame(y_out_sur, index=sample_list)
            y_out_sur_path = os.path.join(down_path, 'y_out_survival.tsv')
            y_out_sur_df.to_csv(y_out_sur_path, sep='\t')

            # Classification
            for i in range(self.param.task_num - 2):
                y_true_cla = output_dict['y_true_cla'][i].cpu().numpy()
                y_pred_cla = output_dict['y_pred_cla'][i].cpu().numpy()
                y_prob_cla = output_dict['y_prob_cla'][i].cpu().numpy()
                y_df_cla = pd.DataFrame({'sample': sample_list, 'y_true': y_true_cla, 'y_pred': y_pred_cla}, index=index)
                y_df_cla_path = os.path.join(down_path, 'y_df_classification_'+str(i+1)+'.tsv')
                y_df_cla.to_csv(y_df_cla_path, sep='\t')
                prob_cla_df = pd.DataFrame(y_prob_cla, columns=range(self.param.class_num[i]), index=sample_list)
                y_prob_cla_path = os.path.join(down_path, 'y_prob_classification_'+str(i+1)+'.tsv')
                prob_cla_df.to_csv(y_prob_cla_path, sep='\t')

            # Regression
            y_true_reg = output_dict['y_true_reg'].cpu().numpy()
            y_pred_reg = np.squeeze(output_dict['y_pred_reg'].cpu().detach().numpy())
            y_df_reg = pd.DataFrame({'sample': sample_list, 'y_true': y_true_reg, 'y_pred': y_pred_reg}, index=index)
            y_df_reg_path = os.path.join(down_path, 'y_df_regression.tsv')
            y_df_reg.to_csv(y_df_reg_path, sep='\t')


    def save_latent_space(self, latent_dict, sample_list):
        """
            save the latent space matrix to disc

            Parameters:
                latent_dict (OrderedDict)          -- the latent space dictionary
                sample_list (ndarray)               -- the sample list for the latent matrix
        """
        reordered_sample_list = sample_list[latent_dict['index'].astype(int)]
        latent_df = pd.DataFrame(latent_dict['latent'], index=reordered_sample_list)
        output_path = os.path.join(self.param.checkpoints_dir, self.param.experiment_name, 'latent_space.tsv')
        print('Saving the latent space matrix...')
        latent_df.to_csv(output_path, sep='\t')


    @staticmethod
    def print_phase(phase):
        """
        print the phase information

        Parameters:
            phase (int)             -- the phase of the training process
        """
        if phase == 'p1':
            print('PHASE 1: Unsupervised Phase')
        elif phase == 'p2':
            print('PHASE 2: Supervised Phase')
        elif phase == 'p3':
            print('PHASE 3: Supervised Phase')
