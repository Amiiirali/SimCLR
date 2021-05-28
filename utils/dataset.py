import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, x_set, y_set, training_set=False, normalize=False):

        """
        Args:
            x_set (string): List of paths to images
            y_set (int): Labels of each image in x_set
            # resize (int): if we want to resize the image first
            # training_set: Whether training dataset or not
            normalize: Whether normalize the dataset or not
        """
        self.x_set = x_set
        self.y_set = y_set
        # self.training_set = training_set
        self.normalize = normalize
        self.transform = self.get_transform()
        self.length = len(x_set)
        if len(x_set) != len(y_set):
            raise ValueError('x set length does not match y set length')

    def get_transform(self):
        transforms_array = []
        # Kernel for Gaussian should be ODD
        kernel_size = int(0.06 * self.original_size())
        kernel_size = kernel_size if kernel_size%2!=0 else kernel_size+1
        transforms_array.extend([transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                 transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5),
                                 transforms.RandomResizedCrop(size=self.original_size()),
                                 transforms.RandomRotation(20),
                                 transforms.ToTensor()])
        if self.normalize:
            transforms_array.extend(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
        transforms_ = transforms.Compose(transforms_array)
        return transforms_

    def original_size(self):
        x = Image.open(self.x_set[0][0]).convert('RGB')
        return transforms.ToTensor()(x).shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = Image.open(self.x_set[idx][0]).convert('RGB')
        y = self.y_set[idx]
        x1 = self.transform(x)
        x2 = self.transform(x)
        return transforms.ToTensor()(x), x1, x2, torch.tensor(y)
