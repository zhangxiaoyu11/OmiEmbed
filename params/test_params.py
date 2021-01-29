from .basic_params import BasicParams


class TestParams(BasicParams):
    """
    This class is a son class of BasicParams.
    This class includes parameters for testing and parameters inherited from the father class.
    """
    def initialize(self, parser):
        parser = BasicParams.initialize(self, parser)

        # Testing parameters
        parser.add_argument('--save_latent_space', action='store_true', help='save the latent space of input data to disc')

        # Logging and visualization
        parser.add_argument('--print_freq', type=int, default=1,
                            help='frequency of showing results on console')

        self.isTrain = False
        self.isTest = True
        return parser
