import os
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import torch
import torchvision.transforms as transforms
import wget
from torchvision.datasets import ImageNet


def get_dataloaders(batch_size, root="/mnt/data"):
    mean=(0.485, 0.456, 0.406)
    # std=(1.5 * 0.229, 1.5 * 0.224, 1.5 * 0.225)  # TODO: check why Maxime added a *1.5. Not present in https://github.com/pytorch/examples/blob/main/imagenet/main.py
    std=(0.229, 0.224, 0.225)

    final_transform = transforms.Normalize(mean, std)

    train_transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        final_transform])
    
    test_transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        final_transform])

    training_data = ImageNet224Dataset(
        root=root,
        split='train',
        transform=train_transform,
    )

    # Download test data
    test_data = ImageNet224Dataset(
        root=root,
        split='val',
        transform=test_transform,
    )

    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


    return train_dataloader, val_dataloader


class ImageNet224Dataset(ImageNet):
    """
    Original ImageNet Dataset using academic torrents.
    TODO: Decide if we want to hide this from releases.
    """
    train_url: ClassVar[str] = "https://academictorrents.com/download/a306397ccf9c2ead27155983c254227c0fd938e2.torrent"
    val_url: ClassVar[str] = "https://academictorrents.com/download/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.torrent"
    devkit_url: ClassVar[str] = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
    devkit_filename: ClassVar[str] = "ILSVRC2012_devkit_t12.tar.gz"
    train_filename: ClassVar[str] = "ILSVRC2012_img_train.tar" 
    val_filename: ClassVar[str] = "ILSVRC2012_img_val.tar" 
    base_folder: ClassVar[str] = "ILSVRC2012"
    
    def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
        
        root_path = Path(root).absolute()
        extracted_path = (root_path / self.base_folder).absolute()
        train_archive_path = (extracted_path / self.train_filename).absolute()
        val_archive_path = (extracted_path / self.val_filename).absolute()
        devkit_path = (extracted_path / self.devkit_filename).absolute()
        
        os.makedirs(extracted_path, exist_ok=True)
        

        if not train_archive_path.exists():
            print(f"Downloading the train archive via transmission-cli to {root_path}")
            # Using transmission-cli to download the torrent
            subprocess.run([
                "transmission-cli", 
                "-w", str(extracted_path),  # Set the download location
                "-er",                # -e: exit when done, -r: delete .torrent when done
                self.train_url             # URL to the torrent file
            ], check=True)
        else: print(f"Training archive exists at {train_archive_path}")

        if not val_archive_path.exists():
            print(f"Downloading the validation archive via transmission-cli to {root_path}")
            subprocess.run([
                "transmission-cli", 
                "-w", str(extracted_path),  # Set the download location
                "-er",                # -e: exit when done, -r: delete .torrent when done
                self.val_url             # URL to the torrent file
            ], check=True)
        else: print(f"Validation archive exists at {val_archive_path}")
        
        if not devkit_path.exists():
            os.makedirs(extracted_path.parent, exist_ok=True)
            print(f"Downloading ImageNet devkit to {devkit_path}")
            wget.download(self.devkit_url, out=str(devkit_path))
        else: print(f"ImageNet devkit archive exists at {devkit_path}")
            
        ImageNet.__init__(self, extracted_path, split, **kwargs)