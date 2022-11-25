import unittest
import logging
from dataset import MNISTDataset, build_dataset_and_loader
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.INFO)
mnist_dir = "../data"


class TestDataset(unittest.TestCase):
    def test_dataset_item(self):
        dataset = MNISTDataset("training", mnist_dir, logger)
        data_dict = dataset[23]
        self.assertTrue(type(data_dict) == dict)
        self.assertTrue(type(data_dict['image']) == np.ndarray)
        self.assertTrue(data_dict['image'].shape[0] == 28**2 and len(data_dict['image'].shape) == 1)
        self.assertTrue(type(data_dict['label']) == np.ndarray and data_dict['label'].dtype == np.long)
        image = data_dict['image'].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.title(f"label: {data_dict['label']}")
        plt.show()

    def test_dataloader(self):
        batch_size = 4
        imsize = 28
        dataset, dataloader = build_dataset_and_loader(batch_size, "testing", logger, mnist_dir)
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
        self.assertTrue(type(batch['image']) == np.ndarray)
        self.assertTrue(batch['image'].shape == (batch_size, imsize**2))
        self.assertTrue(type(batch['label']) == np.ndarray)
        self.assertTrue(batch['label'].shape == (batch_size,))
        nrows = 2
        ncols = int(batch_size * 1.0 / nrows)
        fig, axes = plt.subplots(nrows, ncols)
        for i in range(nrows):
            for j in range(ncols):
                idx = i * nrows + j
                image = batch['image'][idx].reshape(imsize, imsize)
                axes[i, j].imshow(image, cmap="gray")
                axes[i, j].set_title(f"label: {batch['label'][idx]}")
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
