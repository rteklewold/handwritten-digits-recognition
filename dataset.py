import numpy as np
from mnist import MNIST
import logging
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


def build_dataset_and_loader(batch_size: int, partition: str, logger: logging.Logger, data_dir="./data/"):
    assert partition in ('training', 'testing'), f"{partition} is an invalid partition"
    dataset = MNISTDataset(partition, data_dir, logger)
    dataloader = DataLoader(dataset, batch_size, shuffle=partition == 'training', num_workers=0,
                            collate_fn=dataset.collate_fnc)
    return dataset, dataloader


class MNISTDataset(Dataset):
    imsize = 28

    def __init__(self, partition: str, mnist_dir: str, logger: logging.Logger):
        assert partition in ('training', 'testing'), f"{partition} is an invalid partition"
        mnist = MNIST(mnist_dir)
        raw_dataset_parser = getattr(mnist, f"load_{partition}")
        self.images, self.labels = raw_dataset_parser()
        self.nimages = len(self.images)
        logger.info(f"Loaded {self.nimages} images for {partition}")

    def __len__(self):
        return self.nimages

    def __getitem__(self, index: int) -> Dict:
        """
        Return a single data point (a paire of image & label) at the position `index` in the dataset

        :param index: of the data point to be accessed
        :return: {
            'image': np.ndarray, shape (28 * 28), NOTE: images loaded from raw_dataset is flatten to a 1-d array
            'label': int
        }
        """
        img = np.array(self.images[index], dtype=np.float32)  # NOTE: this is just a dummy value, overwrite it with your computation
        label = np.array(self.labels[index], dtype=np.long)  # NOTE: this is just a dummy value, overwrite it with your computation
        # NOTE: remember to keep dtype=np.long for label. It is necessary.
        return {'image': img, 'label': label}

    @staticmethod
    def collate_fnc(data_batch: List[Dict]) -> Dict:
        """
        Recipe for batching individual data_dict into a mini batch

        :param data_batch: a List of N dict, each dict is {
            'image': np.ndarray, shape (28 * 28),
            'label': int
        }
        :return: a single dict {
            'image': np.ndarray, shape (N, 28 * 28),
            'label': np.ndarray, shape (N)
        }
        """
        images = []
        labels = []
        for i in range(len(data_batch)):
            images.append(data_batch[i]['image'])
            labels.append(data_batch[i]['label'])

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.longlong)
        return {'image': images, 'label': labels}
