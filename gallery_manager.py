import os
import io
import cv2
import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple

from utils import detect_and_crop_face


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
            images = sorted([f for f in os.listdir(id_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

            if not images:
                continue

            embed_list = []
            for img_name in images:
                img_path = os.path.join(id_path, img_name)
                try:
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        continue
                        
                    face_img, _ = detect_and_crop_face(img_bgr, padding=0.15)
                    
                    # Convert BGR to RGB for PIL/Torch
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    
                    # Reuse the same preprocess logic as app.py
                    # (Assuming self.preprocess_fn expects bytes, I'll update it to handle PIL or modify app.py)
                    # Actually, app.py's preprocess_image expects bytes.
                    # Let's just use the tensor logic directly here to be safe.
                    
                    # Wait, let's keep it simple. If we have face_img, we can encode it back to bytes.
                    _, buf = cv2.imencode('.jpg', face_img)
                    tensor = self.preprocess_fn(buf.tobytes())
                    
                    with torch.no_grad():
                        emb = self.model(tensor).cpu().numpy()[0]
                    embed_list.append(emb)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

            if not embed_list:
                continue

            # Average all embeddings and re-normalize to unit sphere
            mean_emb = np.mean(embed_list, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb /= norm

            self.embeddings.append({
                'embedding': mean_emb,
                'identity': identity,
                'path': os.path.join(id_path, images[0])  # first image for UI display
            })

        print(f"Gallery refreshed: {len(self.embeddings)} identities loaded (mean embeddings).")

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
                'path': item['path'],
                'embedding': item['embedding']
            })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def get_identities(self) -> List[str]:
        return sorted(list(set(item['identity'] for item in self.embeddings)))
