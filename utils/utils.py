import os
import enum
import json

from utils.sampler import BalancedBatchSampler
from utils.dataset import PatchDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import yaml


def manage_GPU(config):
    """Function to manage GPU and CPU

    Parameters
    ----------
    disable_cuda : flag
        whether use CUDA or not
    """
    # check if gpu training is available
    if not config["disable_cuda"] and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

def strip_extension(path):
    """Function to strip file extension

    Parameters
    ----------
    path : string
        Absoluate path to a slide

    Returns
    -------
    path : string
        Path to a file without file extension
    """
    p = Path(path)
    return str(p.with_suffix(''))

def create_patch_id(path, patch_pattern=None, rootpath=None):
    """Function to create patch ID either by
    1) patch_pattern to find the words to use for ID
    2) rootpath to clip the patch path from the left to form patch ID

    Parameters
    ----------
    path : string
        Absolute path to a patch

    patch_pattern : dict
        Dictionary describing the directory structure of the patch path. The words can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    rootpath : str
        The root directory path containing patch to clip from patch path. Assumes patch contains rootpath.

    Returns
    -------
    patch_id : string
        Remove useless information before patch id for h5 file storage
    """
    if patch_pattern is not None:
        len_of_patch_id = -(len(patch_pattern) + 1)
        patch_id = strip_extension(path).split('/')[len_of_patch_id:]
        return '/'.join(patch_id)
    elif rootpath is not None:
        return strip_extension(path[len(rootpath):].lstrip('/'))
    else:
        return ValueError("Either patch_pattern or rootpath should be set.")

def get_label_by_patch_id(patch_id, patch_pattern, CategoryEnum, is_binary=False):
    """Get category label from patch id. The label can be either 'annotation' or 'subtype' based on is_binary flag.

    Parameters
    ----------
    patch_id : string
        Patch ID get label from

    patch_pattern : dict
        Dictionary describing the directory structure of the patch paths used to find the label word in the patch ID. The words can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    CategoryEnum : enum.Enum
        Acts as the lookup table for category label

    is_binary : bool
        For binary classification, i.e., we will use BinaryEnum instead of SubtypeEnum

    Returns
    -------
    enum.Enum
        label from CategoryEnum
    """
    label = patch_id.split('/')[patch_pattern['annotation' if is_binary else 'subtype']]
    return CategoryEnum[label if is_binary else label.upper()]

def load_chunks(chunk_file_location, chunk_ids):
    """Load patch paths from specified chunks in chunk file

    Parameters
    ----------
    chunks : list of int
        The IDs of chunks to retrieve patch paths from

    Returns
    -------
    list of str
        Patch paths from the chunks
    """
    patch_paths = []
    with open(chunk_file_location) as f:
        data = json.load(f)
        chunks = data['chunks']
        for chunk in data['chunks']:
            if chunk['id'] in chunk_ids:
                patch_paths.extend([[x,chunk['id']] for x in chunk['imgs']])
    if len(patch_paths) == 0:
        raise ValueError(
                f"chunks {tuple(chunk_ids)} not found in {chunk_file_location}")
    return patch_paths

def extract_label_from_patch(CategoryEnum, patch_pattern, patch_path):
    """Get the label value according to CategoryEnum from the patch path

    Parameters
    ----------
    patch_path : str

    Returns
    -------
    int
        The label id for the patch
    """
    '''
    Returns the CategoryEnum
    '''
    patch_path = patch_path[0]
    patch_id = create_patch_id(patch_path, patch_pattern)
    label = get_label_by_patch_id(patch_id, patch_pattern,
            CategoryEnum, is_binary=False)
    return label.value

def extract_labels(CategoryEnum, patch_pattern, patch_paths):
    return [extract_label_from_patch(CategoryEnum, patch_pattern, path) for path in patch_paths]

def create_data_loader(config, chunk_id, training_set=True):
    patch_paths  = load_chunks(config["chunk_file_location"], chunk_id)
    CategoryEnum = enum.Enum('SubtypeEnum', config["subtypes"])
    patch_pattern = {k: i for i, k in enumerate(config["patch_pattern"].split('/'))}
    labels = extract_labels(CategoryEnum, patch_pattern, patch_paths)
    patch_dataset = PatchDataset(patch_paths, labels, normalize=config["normalize"])
    if training_set:
        batch_sampler = BalancedBatchSampler(labels=labels, batch_size=config["batch_size"])
        return DataLoader(patch_dataset,  batch_sampler=batch_sampler,
                          num_workers=config["num_patch_workers"], pin_memory=True,)
    return DataLoader(patch_dataset, batch_size=config["batch_size"], sampler=None,
                      shuffle=True, num_workers=config["num_patch_workers"], pin_memory=True,
                      drop_last=True)

def data_loader(config):
    train_dataset = create_data_loader(config, config["training_chunks"], training_set=True)
    valid_dataset = create_data_loader(config, config["validation_chunks"], training_set=False)
    return train_dataset, valid_dataset

def visualize_dataset(dataset):
    for (x, xis, xjs, _) in dataset:
        # print(x.shape)
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(x[0,:,:,:].permute(1, 2, 0))
        axarr[0].set_title('Original')
        axarr[1].imshow(xis[0,:,:,:].permute(1, 2, 0))
        axarr[1].set_title('First set of Transformers')
        axarr[2].imshow(xjs[0,:,:,:].permute(1, 2, 0))
        axarr[2].set_title('Second set of Transformers')
        plt.show()
