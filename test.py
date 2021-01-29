"""
Separated testing for OmiEmbed
"""
import time
from util import util
from params.test_params import TestParams
from datasets import create_single_dataloader
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    # Get testing parameter
    param = TestParams().parse()
    if param.deterministic:
        util.setup_seed(param.seed)

    # Dataset related
    dataloader, sample_list = create_single_dataloader(param, shuffle=False)  # No shuffle for testing
    print('The size of testing set is {}'.format(len(dataloader)))
    # Get sample list for the dataset
    param.sample_list = dataloader.get_sample_list()
    # Get the dimension of input omics data
    param.omics_dims = dataloader.get_omics_dims()
    if param.downstream_task == 'classification' or param.downstream_task == 'multitask':
        # Get the number of classes for the classification task
        if param.class_num == 0:
            param.class_num = dataloader.get_class_num()
        print('The number of classes: {}'.format(param.class_num))
    if param.downstream_task == 'regression' or param.downstream_task == 'multitask':
        # Get the range of the target values
        values_min = dataloader.get_values_min()
        values_max = dataloader.get_values_max()
        if param.regression_scale == 1:
            param.regression_scale = values_max
        print('The range of the target values is [{}, {}]'.format(values_min, values_max))
    if param.downstream_task == 'survival' or param.downstream_task == 'multitask':
        # Get the range of T
        survival_T_min = dataloader.get_survival_T_min()
        survival_T_max = dataloader.get_survival_T_max()
        if param.survival_T_max == -1:
            param.survival_T_max = survival_T_max
        print('The range of survival T is [{}, {}]'.format(survival_T_min, survival_T_max))

    # Model related
    model = create_model(param)     # Create a model given param.model and other parameters
    model.setup(param)              # Regular setup for the model: load and print networks, create schedulers
    visualizer = Visualizer(param)  # Create a visualizer to print results

    # TESTING
    model.set_eval()
    test_start_time = time.time()  # Start time of testing
    output_dict, losses_dict, metrics_dict = model.init_log_dict()  # Initialize the log dictionaries
    if param.save_latent_space:
        latent_dict = model.init_latent_dict()

    # Start testing loop
    for i, data in enumerate(dataloader):
        dataset_size = len(dataloader)
        actual_batch_size = len(data['index'])
        model.set_input(data)  # Unpack input data from the output dictionary of the dataloader
        model.test()  # Run forward to get the output tensors
        model.update_log_dict(output_dict, losses_dict, metrics_dict, actual_batch_size)  # Update the log dictionaries
        if param.save_latent_space:
            latent_dict = model.update_latent_dict(latent_dict)  # Update the latent space array
        if i % param.print_freq == 0:  # Print testing log
            visualizer.print_test_log(param.epoch_to_load, i, losses_dict, metrics_dict, param.batch_size, dataset_size)

    test_time = time.time() - test_start_time
    visualizer.print_test_summary(param.epoch_to_load, losses_dict, output_dict, test_time)
    visualizer.save_output_dict(output_dict)
    if param.save_latent_space:
        visualizer.save_latent_space(latent_dict, sample_list)
