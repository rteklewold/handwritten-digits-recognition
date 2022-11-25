import unittest
import torch
from model import MNISTClassifier
from utils import BackwardHook


class TestMNISTClassifier(unittest.TestCase):
    def test_forward(self):
        cfg = 10
        model = MNISTClassifier(cfg)
        print(model)
        data_dict = {'image': torch.rand(2, 28**2), 'label': torch.tensor([0, 1], dtype=torch.long)}
        # NOTE: if set batch_size to 1 results to error in BatchNorm because it needs more than 1 sample
        # to calculate mean & var
        ret_dict = model(data_dict)
        print(f"loss: {ret_dict['loss'].item()}")

    def test_gradient(self):
        cfg = 20
        model = MNISTClassifier(cfg)
        print(model)
        bw_hooks = [BackwardHook(name, param) for name, param in model.named_parameters()]
        data_dict = {'image': torch.rand(2, 28 ** 2), 'label': torch.tensor([0, 1], dtype=torch.long)}
        ret_dict = model(data_dict)
        loss = ret_dict['loss']
        model.zero_grad()
        loss.backward()
        for hook in bw_hooks:
            self.assertTrue(hook.grad_mag > 1e-5)
            # if hook.grad_mag < 1e-5:
            #     print(f"zero grad @ {hook.name}")


if __name__ == '__main__':
    unittest.main()
