import logging
import torch
import numpy as np
import matplotlib.pyplot as plt


def create_logger(log_file=None, rank=0, log_level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def model_func_decorator(model: torch.nn.Module, data_dict: dict):
    """
    Wrapper for model's forward function

    :param model: model to invoke forward function
    :param data_dict: {'image', 'label'}
    :return: output of model's forward function
    """
    # move data of data_dict to the same device of the model
    device = next(iter(model.parameters())).device
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            data_dict[k] = torch.from_numpy(v).to(device)
        elif isinstance(v, torch.Tensor):
            data_dict[k] = v.to(model.device)

    # invoke model's forward function
    ret_dict = model(data_dict)

    return ret_dict


class BackwardHook:
    """Backward hook to check gradient magnitude of parameters (i.e. weights & biases)"""
    def __init__(self, name, param, is_cuda=False):
        """Constructor of BackwardHook

        Args:
            name (str): name of parameter
            param (torch.nn.Parameter): the parameter hook is registered to
            is_cuda (bool): whether parameter is on cuda or not
        """
        self.name = name
        self.hook_handle = param.register_hook(self.hook)
        self.grad_mag = -1.0
        self.is_cuda = is_cuda

    def hook(self, grad):
        """Function to be registered as backward hook

        Args:
            grad (torch.Tensor): gradient of a parameter W (i.e. dLoss/dW)
        """
        if not self.is_cuda:
            self.grad_mag = torch.norm(torch.flatten(grad, start_dim=0).detach())
        else:
            self.grad_mag = torch.norm(torch.flatten(grad, start_dim=0).detach().cpu())

    def remove(self):
        self.hook_handle.remove()


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("Pred {} {:2.0f}% (True {})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')