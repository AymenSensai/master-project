import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from typing import Tuple, List, Dict
import numpy as np
import cv2

from utils import detect_and_crop_face


class BalancedIdentitySampler(Sampler):
    """
    Samples num_identities identities per batch, each with samples_per_identity samples.
    Guarantees every batch has multiple samples per identity — required for triplet loss.
    """
    def __init__(self, dataset, num_identities: int = 32, samples_per_identity: int = 4):
        self.dataset = dataset
        self.num_identities = num_identities
        self.samples_per_identity = samples_per_identity

        self.label_to_indices = {}
        for idx, (_, label, _) in enumerate(dataset.samples):
            self.label_to_indices.setdefault(label, []).append(idx)

        self.labels = list(self.label_to_indices.keys())

    def __iter__(self):
        indices = []
        labels = self.labels.copy()
        random.shuffle(labels)

        for label in labels:
            pool = self.label_to_indices[label]
            chosen = random.choices(pool, k=self.samples_per_identity)
            indices.extend(chosen)

        # Yield in batches of num_identities * samples_per_identity
        batch_size = self.num_identities * self.samples_per_identity
        random.shuffle(indices)
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            yield from indices[i:i + batch_size]

    def __len__(self):
        return len(self.labels) * self.samples_per_identity

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
                face_img, _ = detect_and_crop_face(img_bgr, padding=0.15, fast=(self.split == 'train'))
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
    if split == 'train':
        sampler = BalancedIdentitySampler(dataset, num_identities=32, samples_per_identity=4)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, drop_last=True,
            multiprocessing_context='fork' if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
        multiprocessing_context='fork' if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
