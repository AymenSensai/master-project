import os
import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple

class GalleryManager:
    def __init__(self, model, preprocess_fn, device, gallery_dir='gallery'):
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.gallery_dir = gallery_dir
        self.embeddings = []  # List of (embedding, identity_name, image_path)
        
        if not os.path.exists(self.gallery_dir):
            os.makedirs(self.gallery_dir)
            print(f"Created gallery directory: {self.gallery_dir}")
        else:
            self.refresh_gallery()

    def refresh_gallery(self):
        """
        Scans the gallery directory and computes embeddings for all VIS images.
        Expected structure: gallery/identity_name/image.jpg
        """
        self.embeddings = []
        if not os.path.exists(self.gallery_dir):
            return

        identities = [d for d in os.listdir(self.gallery_dir) if os.path.isdir(os.path.join(self.gallery_dir, d))]
        
        for identity in identities:
            id_path = os.path.join(self.gallery_dir, identity)
            for img_name in os.listdir(id_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(id_path, img_name)
                    try:
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                        
                        tensor = self.preprocess_fn(img_bytes)
                        with torch.no_grad():
                            embedding = self.model(tensor).cpu().numpy()[0]
                        
                        self.embeddings.append({
                            'embedding': embedding,
                            'identity': identity,
                            'path': img_path
                        })
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        print(f"Gallery refreshed: {len(self.embeddings)} images loaded.")

    def search(self, probe_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Searches for the closest matches in the gallery.
        """
        if not self.embeddings:
            return []

        results = []
        for item in self.embeddings:
            # Cosine similarity (since embeddings are L2 normalized, it's just dot product)
            similarity = np.dot(probe_embedding, item['embedding'])
            results.append({
                'identity': item['identity'],
                'similarity': float(similarity),
                'path': item['path']
            })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def get_identities(self) -> List[str]:
        return sorted(list(set(item['identity'] for item in self.embeddings)))
