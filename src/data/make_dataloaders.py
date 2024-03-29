import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import os

# from dotenv import load_dotenv
# from pathlib import Path


def make_dataloaders(
    do_train=True,
    train_batch_size=64,
    val_batch_size=64,
    test_batch_size=64,
    val_fraction=0.1,
):
    """
    Create dataloaders for training, validation, and/or testing using the MNIST dataset.

    :param do_train: Whether to create dataloaders for training and validation or for testing
    :type do_train: bool, optional
    :param train_batch_size: Batch size for the training dataloader
    :type train_batch_size: int, optional
    :param val_batch_size: Batch size for the validation dataloader
    :type val_batch_size: int, optional
    :param test_batch_size: Batch size for the testing dataloader
    :type test_batch_size: int, optional
    :param val_fraction: Fraction of training data to be used for validation
    :type val_fraction: float, optional
    :rtype: DataLoader or tuple of DataLoader
    :return: DataLoader for testing if do_train is False; otherwise, a tuple containing DataLoaders for training and validation.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # project_root = Path(__file__).resolve().parents[2]
    # load_dotenv(dotenv_path=project_root / ".env")
    # data_dir = os.environ.get("MNIST_DATA_DIR")

    if do_train:
        mnist_train = MNIST(root=".", train=True, transform=transform, download=True)

        num_train = len(mnist_train)
        num_val = int(val_fraction * num_train)
        mnist_train, mnist_val = random_split(
            mnist_train, [num_train - num_val, num_val]
        )

        train_loader = DataLoader(
            mnist_train, batch_size=train_batch_size, shuffle=True
        )
        val_loader = DataLoader(mnist_val, batch_size=val_batch_size, shuffle=False)

        return train_loader, val_loader

    else:
        mnist_test = MNIST(root=".", train=False, transform=transform, download=True)

        test_loader = DataLoader(mnist_test, batch_size=test_batch_size, shuffle=False)

        return test_loader
