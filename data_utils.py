import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random


def load_data(data_dir, batch_size, conditional=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)

    if not conditional:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return RandomSingleClassDataLoader(dataset, batch_size=batch_size)


class RandomSingleClassDataLoader:
    """
    Yields batches from a SINGLE MNIST CLASS, randomly chosen per batch.
    """

    def __init__(self, dataset, batch_size=64):
        self.dataset = dataset
        self.batch_size = batch_size

        # Pre-index dataset per class for fast sampling
        self.class_to_indices = {k: [] for k in range(10)}
        for idx, (_, label) in enumerate(dataset):
            self.class_to_indices[int(label)].append(idx)

        # Convert to tensors
        for k in self.class_to_indices:
            self.class_to_indices[k] = torch.tensor(self.class_to_indices[k])

        self.num_batches = len(dataset) // batch_size

    def get_batch(self):
        # 1) Randomly choose a MNIST class
        chosen_label = random.randint(0, 9)

        # 2) Choose batch_size indices from that class
        indices = torch.randint(
            low=0,
            high=len(self.class_to_indices[chosen_label]),
            size=(self.batch_size,)
        )

        selected_indices = self.class_to_indices[chosen_label][indices]

        # 3) Fetch images
        images = torch.stack([self.dataset[i][0] for i in selected_indices])

        # label repeated for the whole batch
        labels = torch.full((self.batch_size,), chosen_label, dtype=torch.long)

        return images, labels

    def __len__(self):
        return self.num_batches
