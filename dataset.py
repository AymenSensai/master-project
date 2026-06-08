import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Tuple, List, Dict
import numpy as np
import cv2

from utils import detect_and_crop_face

class TuftsFaceDataset(Dataset):
    """
    Custom Dataset for Tufts Face Database.
    Expects folder structure:
    root_dir/
        TD_RGB/ (or VIS/)
            001/
                TD_RGB_A_1_0.jpg
            002/
                ...
        TD_IR/ (or NIR/)
            001/
                TD_IR_A_1_0.jpg
            ...
    """
    def __init__(self, root_dir: str, split: str = 'train', img_size: int = 112, crop_faces: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.crop_faces = crop_faces
        
        # Determine standard ImageNet normalization
        # or specific normalization if needed. We use generic here.
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
                                         
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
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
        Loads image paths, identity labels, and domain (VIS/NIR) from Tufts structure.
        """
        if not os.path.exists(self.root_dir):
            print(f"Warning: Directory {self.root_dir} does not exist.")
            return

        # Modality folder names (common variations for Tufts)
        vis_mod_names = ['TD_RGB', 'TD_VIS', 'VIS', 'RGB']
        nir_mod_names = ['TD_IR', 'TD_NIR', 'NIR', 'IR']

        vis_root = None
        nir_root = None
        
        for name in vis_mod_names:
            path = os.path.join(self.root_dir, name)
            if os.path.exists(path):
                vis_root = path
                break
        
        for name in nir_mod_names:
            path = os.path.join(self.root_dir, name)
            if os.path.exists(path):
                nir_root = path
                break

        if not vis_root or not nir_root:
            print(f"Error: Could not find VIS ({vis_mod_names}) or NIR ({nir_mod_names}) folders in {self.root_dir}")
            return

        # Identify subjects (common to both modalities)
        vis_identities = set(d for d in os.listdir(vis_root) if os.path.isdir(os.path.join(vis_root, d)))
        nir_identities = set(d for d in os.listdir(nir_root) if os.path.isdir(os.path.join(nir_root, d)))
        
        common_identities = sorted(list(vis_identities.intersection(nir_identities)))
        
        # Simple train/test split if requested (Tufts doesn't come pre-split)
        if self.split == 'train':
            identities_to_use = common_identities[:int(0.8 * len(common_identities))]
        else:
            identities_to_use = common_identities[int(0.8 * len(common_identities)):]

        for idx, identity in enumerate(identities_to_use):
            self.label_to_idx[identity] = idx
            
            # Load VIS (RGB)
            id_vis_path = os.path.join(vis_root, identity)
            for img_name in os.listdir(id_vis_path):
                if img_name.lower().endswith(('.jpg', '.png', '.bmp')):
                    self.samples.append((os.path.join(id_vis_path, img_name), idx, 0)) # VIS=0
            
            # Load NIR (IR)
            id_nir_path = os.path.join(nir_root, identity)
            for img_name in os.listdir(id_nir_path):
                if img_name.lower().endswith(('.jpg', '.png', '.bmp')):
                    self.samples.append((os.path.join(id_nir_path, img_name), idx, 1)) # NIR=1
                        
        print(f"Loaded {len(self.samples)} images from {len(self.label_to_idx)} identities for {self.split} split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        img_path, label, domain = self.samples[index]
        
        # Load image with OpenCV to allow face detection
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback if image is corrupt
            image = Image.new('RGB', (self.img_size, self.img_size))
        else:
            if self.crop_faces:
                # Use the same unified cropping as in inference
                face_img, _ = detect_and_crop_face(img_bgr, padding=0.15)
                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(face_rgb)
            else:
                image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label, domain

def get_dataloader(root_dir: str, split: str, batch_size: int, img_size: int, num_workers: int, crop_faces: bool = True):
    dataset = TuftsFaceDataset(root_dir=root_dir, split=split, img_size=img_size, crop_faces=crop_faces)
    shuffle = (split == 'train')
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=shuffle)
