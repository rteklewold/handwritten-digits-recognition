import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np


class MNISTClassifier(nn.Module):
    def __init__(self, cfg: int, nclasses: int = 10, imsize: int = 28, bn: bool = True):
        """
        Constructor of class MNISTClassifier

        :param cfg: number of neurons in the hidden layer
        :param nclasses: number of class of images in MNIST dataset
        :param imsize: size ( = height = width) of an image in MNIST dataset
        :param bn: to use BatchNorm layer or not
        """
        super(MNISTClassifier, self).__init__()
        #define a Multi-Layer Perceptron (MLP) with 1 input layer, 1 hidden layer, 1 output layer
        # NOTE: the number of neurons in the hidden layer is defined by `cfg`
        # NOTE: the resulted MLP is assigend to attribute self.net
        self.net = nn.Sequential(nn.Linear(imsize*imsize, cfg, bias=False), nn.BatchNorm1d(cfg), nn.ReLU(), nn.Linear(cfg, nclasses, bias=False))  # NOTE: this is just a dummy value, overwrite it with your computation

        '''
        Hint: Here is an example of an MLP which takes input is a batch of vectors of size (N, 8) and map it into a 
        batch of vectors of size (N, 16); then apply BatchNorm1d and activate the result with ReLU.  
        
        self.net = nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        
        Documentation:
        Linear layer (i.e. fully connected layer): https://pytorch.org/docs/1.7.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear
        BatchNorm1d: https://pytorch.org/docs/1.7.0/generated/torch.nn.BatchNorm1d.html?highlight=batchnorm1d#torch.nn.BatchNorm1d
        ReLU: https://pytorch.org/docs/1.7.0/generated/torch.nn.ReLU.html?highlight=relu#torch.nn.ReLU
        Sequential: https://pytorch.org/docs/1.7.0/generated/torch.nn.Sequential.html?highlight=sequential#torch.nn.Sequential
        '''

        self.loss_fn = torch.nn.functional.cross_entropy

    def forward(self, data_dict: Dict):
        """
        This method define the forward pass of the model

        :param data_dict: {
            'image': torch.Tensor, shape (N, 784), a batch of FLATTENED images,
            'label': torch.Tensor, shape (N), a batch of labels
        }
        :return: {
            'loss': float
            'cls': predicted class for each image in the batch (only in EVAL mode)
            'prob': probability of predicted class for each image in the batch (only in EVAL mode)
        }
        """
        logits = self.net(data_dict['image'])  # (N, 10)

        ret_dict = dict()
        if self.training:
            loss = self.loss_fn(logits, data_dict['label'])
            ret_dict = {'loss': loss}
        else:
            cls, prob = self.decode_prediction(logits)
            ret_dict = {'cls': cls, 'prob': prob, 'logits': logits}
        return ret_dict

    @staticmethod
    def decode_prediction(logits: torch.Tensor):
        """
        Decode logits into predicted class and associated probability

        :param logits: (N, 10) - predicted logits for each image in the batch
        :return:
            * cls: (N) - predicted class
            * prob: (N) - probability of predicted class
        """
        #activate predicted logits with torch.nn.functional.softmax to get predicted probability
        # NOTE: remember to specify the dimension along which softmax is applied
        # HINT: logits is a 2d tensor (matrix) of size (N, 10), row i-th of its is the predicted logits for image i-th
        # Doc: https://pytorch.org/docs/1.7.0/nn.functional.html?highlight=softmax#torch.nn.functional.softmax
        #prob = np.array(logits.shape[0],logits.shape[1])  # NOTE: this is just a dummy value, overwrite it with your computation
        # HINT: prob should have shape of (N, 10)

        prob = torch.nn.functional.softmax(logits, dim=1)

        # predicted class is defined as the class having the highest probability
        #for each row of `prob` find the index where the probability is the highest
        # HINT: slow method: you can write a for loop
        # HINT: elegant method: use torch.max
        # Doc: https://pytorch.org/docs/1.7.0/generated/torch.max.html?highlight=max#torch.max (look for torch.max(input, dim, keepdim=False, *, out=None))
        #cls = torch.zeros(logits.shape[0])  # NOTE: this is just a dummy value, overwrite it with your computation

        prob, cls = torch.max(prob, dim=1)
        # NOTE: `cls` store the class of N images and the size of (N)
        return cls, prob
