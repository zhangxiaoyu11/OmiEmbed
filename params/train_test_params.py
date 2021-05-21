from .basic_params import BasicParams


class TrainTestParams(BasicParams):
    """
    This class is a son class of BasicParams.
    This class includes parameters for training & testing and parameters inherited from the father class.
    """
    def initialize(self, parser):
        parser = BasicParams.initialize(self, parser)

        # Training parameters
        parser.add_argument('--epoch_num_p1', type=int, default=50,
                            help='epoch number for phase 1')
        parser.add_argument('--epoch_num_p2', type=int, default=50,
                            help='epoch number for phase 2')
        parser.add_argument('--epoch_num_p3', type=int, default=100,
                            help='epoch number for phase 3')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='initial learning rate')
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='momentum term of adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='The learning rate policy for the scheduler. [linear | step | plateau | cosine]')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, default start from 1')
        parser.add_argument('--epoch_num_decay', type=int, default=50,
                            help='Number of epoch to linearly decay learning rate to zero (lr_policy == linear)')
        parser.add_argument('--decay_step_size', type=int, default=50,
                            help='The original learning rate multiply by a gamma every decay_step_size epoch (lr_policy == step)')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight decay (L2 penalty)')

        # Network saving and loading parameters
        parser.add_argument('--continue_train', action='store_true',
                            help='load the latest model and continue training')
        parser.add_argument('--save_model', action='store_true',
                            help='save the model during training')
        parser.add_argument('--save_epoch_freq', type=int, default=-1,
                            help='frequency of saving checkpoints at the end of epochs, -1 means only save the last epoch')

        # Logging and visualization
        parser.add_argument('--print_freq', type=int, default=1,
                            help='frequency of showing results on console')

        # Dataset parameters
        parser.add_argument('--train_ratio', type=float, default=0.8,
                            help='ratio of training set in the full dataset')
        parser.add_argument('--test_ratio', type=float, default=0.2,
                            help='ratio of testing set in the full dataset')

        self.isTrain = True
        self.isTest = True
        return parser
