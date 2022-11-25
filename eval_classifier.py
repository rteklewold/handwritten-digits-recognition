import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from logging import Logger
from typing import Tuple
from tqdm import tqdm
from utils import model_func_decorator


@torch.no_grad()
def compute_PR_curve(model: Module, dataloader: DataLoader, logger: Logger, nthreshold: int = 100, ncls: int = 10) \
        -> Tuple[np.ndarray, ...]:
    """

    :param model: the classifier to be evaluated
    :param dataloader: to provide data
    :param nthreshold: number of threshold
    :param ncls: number of class of image
    :return: p_array, r_array
    """
    if model.training:
        logger.warning("model.eval() is invoked in compute_PR_curve")
        model.eval()

    # get model's prediction for every data in dataloader
    predictions = {'cls': [], 'prob': []}
    label = []
    for data_dict in dataloader:
        ret_dict = model_func_decorator(model, data_dict)
        predictions['cls'].append(ret_dict['cls'])
        predictions['prob'].append(ret_dict['prob'])
        label.append(data_dict['label'])

    predictions['cls'] = torch.cat(predictions['cls'])  # (N)
    predictions['prob'] = torch.cat(predictions['prob'])  # (N)
    label = torch.cat(label)  # (N)

    # compute precision & recall for each threshold
    p_list, r_list, acc_list = [], [], []
    for threshold in np.linspace(start=0.0, stop=0.99, num=nthreshold):
        # categorize prediction according to threshold
        positive_mask = predictions['prob'] > threshold  # (N)
        negative_mask = torch.logical_not(positive_mask)  # (N)
        # find number of TP, FP, FN, TN
        true_positive = (predictions['cls'][positive_mask] == label[positive_mask]).sum()
        false_positive = positive_mask.sum() - true_positive
        # ---
        false_negative = 0
        for cls in range(ncls):
            false_negative += torch.sum(label[negative_mask] == cls)
        true_negative = negative_mask.sum() - false_negative
        # ---
        precision = true_positive * 1.0 / (true_positive + false_positive)
        recall = true_positive * 1.0 / (true_positive + false_negative)
        acc = (true_positive + true_negative) * 1.0 / label.shape[0]
        p_list.append(precision)
        r_list.append(recall)
        acc_list.append(acc)

    # format output
    p_list = np.array(p_list)
    r_list = np.array(r_list)
    acc_list = np.array(acc_list)
    inds = np.argsort(r_list)
    p_list = p_list[inds]
    r_list = r_list[inds]
    acc_list = acc_list[inds]
    return p_list, r_list, acc_list
