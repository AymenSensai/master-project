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
        self.skipped_identities = []
        if not os.path.exists(self.gallery_dir):
            return

        identities = sorted([d for d in os.listdir(self.gallery_dir) if os.path.isdir(os.path.join(self.gallery_dir, d))])
        total = len(identities)

        for identity in identities:
            id_path = os.path.join(self.gallery_dir, identity)
            images = sorted([f for f in os.listdir(id_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

            if not images:
                print(f"  [SKIP] '{identity}' — dossier vide ou aucun fichier image valide.")
                self.skipped_identities.append({'identity': identity, 'reason': 'empty_dir'})
                continue

            embed_list = []
            failed_images = []
            for img_name in images:
                img_path = os.path.join(id_path, img_name)
                try:
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        print(f"  [WARN] '{identity}/{img_name}' — impossible de lire l'image (corrompue ?).")
                        failed_images.append(img_name)
                        continue

                    face_img, _ = detect_and_crop_face(img_bgr, padding=0.15)

                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                    _, buf = cv2.imencode('.jpg', face_img)
                    tensor = self.preprocess_fn(buf.tobytes())

                    with torch.no_grad():
                        emb = self.model(tensor).cpu().numpy()[0]
                    embed_list.append(emb)
                except Exception as e:
                    print(f"  [WARN] '{identity}/{img_name}' — erreur durant l'embedding : {e}")
                    failed_images.append(img_name)

            if not embed_list:
                print(f"  [SKIP] '{identity}' — aucun embedding généré ({len(failed_images)}/{len(images)} images en échec).")
                self.skipped_identities.append({'identity': identity, 'reason': 'all_images_failed', 'failed': failed_images})
                continue

            if failed_images:
                print(f"  [WARN] '{identity}' — {len(failed_images)}/{len(images)} image(s) ignorée(s), embedding calculé sur {len(embed_list)}.")

            # Average all embeddings and re-normalize to unit sphere
            mean_emb = np.mean(embed_list, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb /= norm

            self.embeddings.append({
                'embedding': mean_emb,
                'identity': identity,
                'path': os.path.join(id_path, images[0])
            })

        skipped = len(self.skipped_identities)
        loaded  = len(self.embeddings)
        print(f"Gallery refreshed: {loaded}/{total} identités chargées" +
              (f" ({skipped} ignorées — voir warnings ci-dessus)." if skipped else "."))

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
