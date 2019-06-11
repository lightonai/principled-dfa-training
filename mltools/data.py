""" Tools for datasets, data augmentation, and general data-handling.

Include easy access to a number of datasets (MNIST and associated, CIFAR10/CIFAR100, ImageNette/ImageWoof, ImageNet).
"""
import numpy as np
import os
import torch
import torch.cuda as cuda
import torch.utils.data as data_utils
import torchvision.datasets as datasets
import torch.distributed as distributed
import torchvision.transforms as transforms

import mltools.logging as log

LOGTAG = "DATA"

# Implemented datasets name and their underlying data types.
# Datasets with same data types can use networks with similar architectures and the same preprocessing functions.
DATASETS = {'MNIST': 'MNIST', 'FashionMNIST': 'MNIST', 'CIFAR-10': 'CIFAR', 'CIFAR-100': 'CIFAR',
            'ImageNette': 'ImageNet', 'ImageWoof': 'ImageNet', 'ImageNet': 'ImageNet'}

FOLDER_NAME = {'MNIST': 'MNIST', 'FashionMNIST': 'FashionMNIST', 'CIFAR-10': 'CIFAR10', 'CIFAR-100': 'CIFAR100',
               'ImageNette': 'imagenette', 'ImageWoof': 'imagewoof', 'ImageNet': 'ImageNet'}

# Pre-calculated normalization values [mean, std] for pre-processing.
NORMALIZATION = {'MNIST': ((0.1307,), (0.3081,)), 'CIFAR': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                 'ImageNet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

# Default transforms ensure we get normalized data as tensors and not as PIL images.
default_mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORMALIZATION['MNIST'])])
default_cifar_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORMALIZATION['CIFAR'])])
default_imagenet_transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor(),
                                                 transforms.Normalize(*NORMALIZATION['ImageNet'])])

default_transforms = {'MNIST': default_mnist_transform, 'CIFAR': default_cifar_transform,
                      'ImageNet': default_imagenet_transform}

# Fast default transforms made to be used in combination with FastLoader.
# Contains only transforms to be applied on the PIL image.
fast_default_imagenet_transform = transforms.CenterCrop(256)

fast_default_transforms = {'MNIST': None, 'CIFAR': None, 'ImageNet': fast_default_imagenet_transform}


class FastLoader:
    """Wraps a PyTorch :class:`torch.utils.data.DataLoader` and provides faster data loading to GPU. Also performs
    normalization at run time.

    Iterate over the :class:`torch.utils.data.DataLoader` to receive the (input, output) batches from the data loader.

    :param data_loader: data loader from which to fetch data.
    :type data_loader: torch.utils.data.DataLoader
    :param data_mean: per RGB channel mean of the data to use for normalization.
    :type data_mean: list[float, float, float]
    :type data_std: per RGB channel standard deviation of the data to use for normalization.
    :param data_std: list[float, float, float]
    """
    def __init__(self, data_loader, device, data_mean, data_std):
        self.data_loader = data_loader
        self.device = device
        self.data_mean, self.data_std = self._preprocess(data_mean), self._preprocess(data_std)

    def _preprocess(self, data):
        """Utility internal function to transform normalization data to :class:`torch.Tensor` and move them to GPU.

        :param data: RGB list of values to convert.
        :type data: list[float, float, float]
        :rtype: torch.Tensor
        """
        return torch.tensor([d for d in data]).to(self.device).view(1, 3, 1, 1)

    def __iter__(self):
        """Yield a batch of (input, output) from the data loader, with the inputs normalized.

        :return: batch of (input, output).
        :rtype: (torch.Tensor, torch.Tensor)
        """
        stream = cuda.Stream(self.device)
        first_entry = True
        for next_input, next_target in self.data_loader:
            with cuda.stream(stream):
                # Pre-load a batch of input and targets to the GPU, and normalize the input:
                next_input = next_input.to(self.device, non_blocking=True)
                next_target = next_target.to(self.device, non_blocking=True)
                next_input = next_input.float()
                next_input = next_input.sub_(self.data_mean).div_(self.data_std)
            if not first_entry:
                yield input, target  # Yield the pre-loaded batch of input and targets.
            else:
                # On the first entry, we have to do the pre-loading step twice (as nothing as been pre-loaded before!)
                first_entry = False
            cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
        yield input, target

    def __len__(self):
        return len(self.data_loader)


def get_loaders(dataset, batch_size, test, data_path=None, train_transform=None, validation_transform=None,
                train_percentage=0.85):
    """Return PyTorch :class:`torch.utils.data.DataLoader` for training and validation, outfitted with a random sampler.

    This doesn't support multiple workers and distributed training.
    For performance, use :func:get_fast_loaders` instead.

    If set to run on the test set, :param:`train_percentage` will be ignored and set to 1.

    :param dataset: name of the dataset, (MNIST, FashionMNIST, CIFAR-10, CIFAR-100, ImageNette, ImageWoof, ImageNet)
                    are available.
    :type dataset: str
    :param batch_size: batch size for training and validation.
    :type batch_size: int
    :param test: run validation on the test set.
    :type test: bool
    :param data_path: path to folder containing dataset.
    :type data_path: str
    :param train_transform: PyTorch transform to apply to images for training.
    :type train_transform: torchvision.transforms.Compose
    :param validation_transform: PyTorch transform to apply to images for validation.
    :type validation_transform: torchvision.transforms.Compose
    :param train_percentage: percentage of the data in the training set.
    :type train_percentage: float
    :return: training and validation data loaders.
    :rtype: (torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    # Check if any parameters has been set to its default value, and if so, setup the defaults.
    data_path, train_transform, validation_transform = _setup_defaults(dataset, data_path, train_transform,
                                                                       validation_transform, False)

    # Get all of the training data available.
    train_data = _get_train_data(dataset, data_path, train_transform)

    if test:
        # If in test mode, fetch the test data and prepare the validation loader with it:
        test_data = _get_test_data(dataset, data_path, validation_transform)
        train_sampler = data_utils.RandomSampler(train_data)  # Sample over the entire training set.
        # We always use pinned memory as data is loaded on the CPU and then pushed to the GPU. This fastens transfers.
        validation_loader = data_utils.DataLoader(test_data, batch_size=batch_size, pin_memory=True)
    else:
        # Otherwise, perform a train/validation split on the training data available:
        dataset_size = len(train_data)
        split_index = int(dataset_size * train_percentage)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)  # Make sure we don't always get the same train/validation split.
        train_indices, validation_indices = indices[:split_index], indices[split_index:]
        train_sampler = data_utils.SubsetRandomSampler(train_indices)  # Sample over only a subset of the training set.
        validation_sampler = data_utils.SubsetRandomSampler(validation_indices)
        validation_loader = data_utils.DataLoader(train_data, sampler=validation_sampler, batch_size=batch_size,
                                                  pin_memory=True)
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, pin_memory=True, sampler=train_sampler,
                                         drop_last=True)

    return train_loader, validation_loader


def get_fast_loaders(dataset, batch_size, test, device, data_path=None, train_transform=None, validation_transform=None,
                     train_percentage=0.85, workers=4):
    """Return :class:`FastLoader` for training and validation, outfitted with a random sampler.

    If set to run on the test set, :param:`train_percentage` will be ignored and set to 1.

    The transforms should only include operations on PIL images and should not convert the images to a tensor, nor
    handle normalization of the tensors. This is handled at runtime by the fast loaders.

    If you are not looking for high-performance, prefer :func:`get_loaders`.

    :param dataset: name of the dataset, (MNIST, FashionMNIST, CIFAR-10, CIFAR-100, ImageNette, ImageWoof, ImageNet)
                    are available.
    :type dataset: str
    :param batch_size: batch size for training and validation.
    :type batch_size: int
    :param test: run validation on the test set.
    :type test: bool
    :param data_path: path to folder containing dataset.
    :type data_path: str
    :param train_transform: PyTorch transform to apply to images for training.
    :type train_transform: torchvision.transforms.Compose
    :param validation_transform: PyTorch transform to apply to images for validation.
    :type validation_transform: torchvision.transforms.Compose
    :param train_percentage: percentage of the data in the training set.
    :type train_percentage: float
    :param workers: number of subprocesses to use for data loading. Use 0 for loading in the main process.
    :type workers: int
    :return: training and validation fast data loaders.
    :rtype: (FastLoader, FastLoader)
    """
    # Check if any parameters has been set to its default value, and if so, setup the defaults.
    data_path, train_transform, validation_transform = _setup_defaults(dataset, data_path, train_transform,
                                                                       validation_transform, fast=True)

    # Get all of the training data available.
    train_data = _get_train_data(dataset, data_path, train_transform)
    log.log("Training data succesfully fetched!", LOGTAG, log.Level.DEBUG)

    if not test:
        # Perform a train/validation split on the training data available:
        # For performance reasons, the train/validation split will always be the same.
        # TODO: Implement random train/validation split with fast loading and distributed training.
        log.log("Running in standard training/validation mode.", LOGTAG, log.Level.INFO)
        dataset_size = len(train_data)
        split_index = int(dataset_size * train_percentage)
        log.log("{0}:{1}".format(dataset_size, split_index), LOGTAG, log.Level.HIGHLIGHT)
        validation_data = train_data[split_index:]
        train_data = train_data[:split_index]
        log.log("Validation data succesfully fetched!", LOGTAG, log.Level.DEBUG)
    else:
        # Fetch the test data:
        log.log("Running in <b>test</b> mode. All training data available will be used, and "
                "validation will be done on the test set. Are you really ready to publish?", LOGTAG, log.Level.WARNING)
        validation_data = _get_test_data(dataset, data_path, validation_transform)
        log.log("Test data succesfully fetched!", LOGTAG, log.Level.DEBUG)

    if distributed.is_initialized():
        # If running in distributed mode, use a DistributedSampler:
        log.log("Running in <b>distributed</b> mode. This hasn't been thoroughly tested, beware!",
                LOGTAG, log.Level.WARNING)
        train_sampler = data_utils.distributed.DistributedSampler(train_data)
    else:
        # Otherwise, default to a RandomSampler:
        train_sampler = data_utils.RandomSampler(train_data)

    # Build the train and validation loaders, using pinned memory and a custom collate function to build the batches.
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, num_workers=workers, pin_memory=True,
                                         sampler=train_sampler, collate_fn=_fast_collate, drop_last=True)
    log.log("Train loader succesfully created!", LOGTAG, log.Level.DEBUG)
    validation_loader = data_utils.DataLoader(validation_data, batch_size=batch_size, num_workers=workers,
                                              pin_memory=True, collate_fn=_fast_collate)
    log.log("Validation loader succesfully created!", LOGTAG, log.Level.DEBUG)

    # Wrap the PyTorch loaders in the custom FastLoader class and feed it the normalization parameters associated
    # with the dataset.
    return FastLoader(train_loader, device, *NORMALIZATION[DATASETS[dataset]]), \
           FastLoader(validation_loader, device, *NORMALIZATION[DATASETS[dataset]])


# UTILITY FUNCTIONS
# The utility functions below are not meant to be used outside this module.

def _setup_defaults(dataset, data_path, train_transform, validation_transform, fast):
    """Helper function to setup default path and transforms when creating data loaders.

    If any of :param:`data_path`, :param:`train_transform`, or :param:`test_transform` are None, they will be replaced
    by default values.

    If :param:`fast` is set, transforms compatible with :class:`FastLoader` will be used.

    :param dataset: name of the dataset, (MNIST, FashionMNIST, CIFAR-10, CIFAR-100, ImageNette, ImageWoof, ImageNet)
                    are available.
    :type dataset: str
    :param data_path: path to folder containing dataset.
    :param train_transform: PyTorch transform to apply to images for training.
    :type train_transform: torchvision.transforms.Compose
    :param validation_transform: PyTorch transform to apply to images for validation.
    :type validation_transform: torchvision.transforms.Compose
    :param fast: whether fast loaders are used.
    :type fast: bool
    :return: path to data, train transform, and test transform.
    :rtype: (str, torchvision.transforms.Compose, torchvision.transforms.Compose)
    """
    # Setup the path to the dataset.
    if data_path is None:
        if dataset in ['ImageNette', 'ImageWoof', 'ImageNet']:
            # For these datasets, we cannot rely on torchvision for automatic downloading=
            # TODO: Implement automatic downloading of ImageNette, Imagewoof, and ImageNet.
            log.log("Auto-download of dataset {0} is not currently supported. "
                    "Specify a path containing the 'train' and 'val' folders of the dataset.".format(dataset),
                    LOGTAG, log.Level.ERROR)
            raise NotImplementedError("Auto-download of dataset {0} is not currently supported, select a path.")
        data_path = dataset  # Default to putting the dataset in a folder named 'dataset' in the working folder.

    # Setup the train and validation/test transforms.
    if fast:
        # Use the fast default transforms base instead:
        transforms_base = fast_default_transforms
    else:
        transforms_base = default_transforms
    # Currently, the same train and validation/test transforms are used.
    # If default data augmentation is implemented, it should be on the training set and not on the validation/test one.
    if train_transform is None:
        train_transform = transforms_base[DATASETS[dataset]]
    if validation_transform is None:
        validation_transform = transforms_base[DATASETS[dataset]]

    return data_path, train_transform, validation_transform


def _get_train_data(dataset, data_path, transform):
    """Helper function to retrieve training data associated with a dataset.

    :param dataset: name of the dataset, (MNIST, FashionMNIST, CIFAR-10, CIFAR-100, ImageNette, ImageWoof, ImageNet)
                    are available.
    :type dataset: str
    :param data_path: path to a folder containing the dataset.
    :type data_path: str
    :param transform: PyTorch transform to apply to the data.
    :type transform: torchvision.transforms.Compose
    :return: full training data from dataset with transform applied.
    :rtype: torch.utils.data.Dataset
    """
    return _get_data(dataset, data_path, transform, False)


def _get_test_data(dataset, data_path, transform):
    """Helper function to retrieve test data associated with a dataset.

    :param dataset: name of the dataset, (MNIST, FashionMNIST, CIFAR-10, CIFAR-100, ImageNette, ImageWoof, ImageNet)
                    are available.
    :type dataset: str
    :param data_path: path to a folder containing the dataset.
    :type data_path: str
    :param transform: PyTorch transform to apply to the data.
    :type transform: torchvision.transforms.Compose
    :return: full test data from dataset with transform applied.
    :rtype: torch.utils.data.Dataset
    """
    return _get_data(dataset, data_path, transform, True)


def _get_data(dataset, data_path, transform, test):
    """Helper function to retrieve training/test data associated with a dataset.

    :param dataset: name of the dataset, (MNIST, FashionMNIST, CIFAR-10, CIFAR-100, ImageNette, ImageWoof, ImageNet)
                    are available.
    :type dataset: str
    :param data_path: path to a folder containing the dataset.
    :type data_path: str
    :param transform: PyTorch transform to apply to the data.
    :type transform: torchvision.transforms.Compose
    :param test: if true, return test data instead of training data.
    :type test: bool
    :return: full training data from dataset with transform applied.
    :rtype: torch.utils.data.Dataset
    """
    if dataset in ['ImageNet', 'ImageNette', 'ImageWoof']:
        data_path = os.path.join(data_path, FOLDER_NAME[dataset])
    if dataset == 'MNIST':
        data = datasets.MNIST(data_path, train=not test, download=True, transform=transform)
    elif dataset == 'FashionMNIST':
        data = datasets.FashionMNIST(data_path, train=not test, download=True, transform=transform)
    elif dataset == 'CIFAR-10':
        data = datasets.CIFAR10(data_path, train=not test, download=True, transform=transform)
    elif dataset == 'CIFAR-100':
        data = datasets.CIFAR100(data_path, train=not test, download=True, transform=transform)
    elif dataset in ['ImageNette', 'ImageWoof', 'ImageNet']:
        # These datasets are not available in torchvision, so we find and build them ourselves:
        train_directory = os.path.join(data_path, 'val' if test else 'train')
        data = datasets.ImageFolder(train_directory, transform)
    else:
        log.log("Dataset {0} is not available ! Choose from (MNIST, FashionMNIST, CIFAR-10, CIFAR-100, "
                "ImageNette, ImageWoof, ImageNet).".format(dataset), LOGTAG, log.Level.ERROR)
        raise NotImplementedError("Dataset {0} is not available!".format(dataset))
    return data


def _fast_collate(batch):
    """Faster batch collation function.

    Adapted from NVIDIA recommendations.

    :param batch: batch of PIL images to process.
    :type batch: list[PIL.Image]
    :return: batch of (input, output).
    :rtype: (torch.Tensor, torch.Tensor))
    """
    images = [image[0] for image in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    width = images[0].size[0]
    height = images[0].size[1]
    tensor = torch.zeros((len(images), 3, height, width), dtype=torch.uint8)
    for i, image in enumerate(images):
        numpy_array = np.asarray(image, dtype=np.uint8)
        if numpy_array.ndim < 3:
            numpy_array = np.expand_dims(numpy_array, axis=-1)
        numpy_array = np.rollaxis(numpy_array, 2)
        tensor[i] += torch.from_numpy(numpy_array)
    return tensor, targets
