"""
This package contains modules related to objective functions, optimizations, and network architectures.
"""

import importlib
from models.basic_model import BasicModel


def find_model_using_name(model_name):
    """
    Import the module with certain name
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # instantiate the model class
    model = None
    # Change the name format to corresponding class name
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BasicModel):
            model = cls

    if model is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))

    return model


def get_param_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_parameters


def create_model(param):
    """
    Create a model given the parameters
    """
    model = find_model_using_name(param.model)
    # Initialize the model
    instance = model(param)
    print('Model [%s] was created' % type(instance).__name__)
    return instance
