import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Tuple, List, Dict

class CasiaNirVisDataset(Dataset):
    """
    Custom Dataset for CASIA NIR-VIS 2.0.
    Expects folder structure:
    root_dir/
        train/
            0001/
                VIS/
                    img1.jpg
                    ...
                NIR/
                    img1.bmp
                    ...
            0002/
               ...
        test/
            ...
    """
    def __init__(self, root_dir: str, split: str = 'train', img_size: int = 112):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.img_size = img_size
        
        # Determine standard ImageNet normalization
        # or specific normalization if needed. We use generic here.
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
                                         
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                # Consider adding RandomRotation or RandomCrop if needed
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize
            ])

        self.samples = [] # List of tuples: (image_path, label_id, domain)
                          # Domain: 0 for VIS, 1 for NIR
        self.label_to_idx = {}
        
        # In a real environment, you'd parse the specific list files provided by CASIA 
        # (e.g., View1, View2) which specify exact file pairs for protocols.
        # Here we simulate parsing a folder structure.
        self._load_data()

    def _load_data(self):
        """
        Loads image paths, identity labels, and domain (VIS/NIR) from directory structure.
        """
        if not os.path.exists(self.root_dir):
            # For demonstration if dir doesn't exist, we will create dummy data later,
            # but ideally we log a warning here.
            print(f"Warning: Directory {self.root_dir} does not exist. Awaiting data.")
            return

        identities = sorted(os.listdir(self.root_dir))
        for idx, identity in enumerate(identities):
            id_path = os.path.join(self.root_dir, identity)
            if not os.path.isdir(id_path):
                continue
                
            self.label_to_idx[identity] = idx
            
            # Load VIS
            vis_dir = os.path.join(id_path, 'VIS')
            if os.path.exists(vis_dir):
                for img_name in os.listdir(vis_dir):
                    if img_name.endswith(('.jpg', '.png', '.bmp')):
                        self.samples.append((os.path.join(vis_dir, img_name), idx, 0))
                        
            # Load NIR
            nir_dir = os.path.join(id_path, 'NIR')
            if os.path.exists(nir_dir):
                for img_name in os.listdir(nir_dir):
                    if img_name.endswith(('.jpg', '.png', '.bmp')):
                        self.samples.append((os.path.join(nir_dir, img_name), idx, 1))
                        
        print(f"Loaded {len(self.samples)} images from {len(self.label_to_idx)} identities for {self.split} split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        img_path, label, domain = self.samples[index]
        
        # Convert all to RGB to ensure 3 channels
        # If NIR is grayscale (1 channel), converting to RGB duplicates it to 3 channels
        image = Image.open(img_path).convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label, domain

def get_dataloader(root_dir: str, split: str, batch_size: int, img_size: int, num_workers: int):
    dataset = CasiaNirVisDataset(root_dir=root_dir, split=split, img_size=img_size)
    shuffle = (split == 'train')
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=shuffle)
