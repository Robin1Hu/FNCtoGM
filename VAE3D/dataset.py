from typing import Union, Sequence
import os
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torchio as tio


class NIfTIDataset(Dataset):
    def __init__(self, directory: str, transform=None):
        self.directory = directory
        self.transform = transform
        #print(os.listdir(directory))
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if
                      f.endswith('.nii') or f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # Load the NIfTI file
        image = nib.load(file_path).get_fdata()
        # Convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            #print(image.shape)
            image = self.transform(image)
            #print(image.shape)

        return image


class VAEDataset:
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 4,
            val_batch_size: int = 4,
            patch_size: Union[int, Sequence[int]] = 64,
            num_workers: int = 4,
            pin_memory: bool = False,
            **kwargs,
    ):
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.setup()

    def setup(self) -> None:
        # Define transforms, adjust as necessary
        train_transforms = transforms.Compose([
            transforms.Lambda(lambda x: torch.unsqueeze(x, 0)),  # Add channel dimension
            tio.Resize(self.patch_size)
        ])

        val_transforms = transforms.Compose([
            transforms.Lambda(lambda x: torch.unsqueeze(x, 0)),  # Add channel dimension
            tio.Resize(self.patch_size)
        ])

        self.train_dataset = NIfTIDataset(
            os.path.join(self.data_dir, 'train'),
            transform=train_transforms
        )

        self.val_dataset = NIfTIDataset(
            os.path.join(self.data_dir, 'val'),
            transform=val_transforms
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        # Assuming test data is stored in the same way as validation
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
