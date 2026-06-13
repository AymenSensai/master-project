import os
import io
import cv2
import torch
import pickle
import hashlib
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple

from utils import detect_and_crop_face


class GalleryManager:
    def __init__(self, model, preprocess_fn, device, gallery_dir='gallery', cache_path='checkpoints/gallery_cache.pkl'):
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.gallery_dir = gallery_dir
        self.cache_path = cache_path
        self.embeddings = []
        self.skipped_identities = []

        if not os.path.exists(self.gallery_dir):
            os.makedirs(self.gallery_dir)
            print(f"Created gallery directory: {self.gallery_dir}")
        else:
            self.refresh_gallery()

    def _fingerprint(self) -> str:
        """MD5 of every image path + its last-modified time in the gallery."""
        entries = []
        for identity in sorted(os.listdir(self.gallery_dir)):
            id_path = os.path.join(self.gallery_dir, identity)
            if not os.path.isdir(id_path):
                continue
            for f in sorted(os.listdir(id_path)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    full = os.path.join(id_path, f)
                    entries.append(f"{full}:{os.path.getmtime(full)}")
        return hashlib.md5('\n'.join(entries).encode()).hexdigest()

    def _load_cache(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, 'rb') as f:
                cache = pickle.load(f)
            if cache.get('fingerprint') != self._fingerprint():
                print("Gallery: cache obsolète (fichiers modifiés), recalcul en cours...")
                return False
            self.embeddings         = cache['embeddings']
            self.skipped_identities = cache.get('skipped_identities', [])
            print(f"Gallery: {len(self.embeddings)} identités chargées depuis le cache (démarrage instantané).")
            return True
        except Exception as e:
            print(f"Gallery: échec du chargement du cache ({e}), recalcul en cours...")
            return False

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'fingerprint':        self._fingerprint(),
                    'embeddings':         self.embeddings,
                    'skipped_identities': self.skipped_identities,
                }, f)
            print(f"Gallery: cache sauvegardé → {self.cache_path}")
        except Exception as e:
            print(f"Gallery: impossible de sauvegarder le cache ({e})")

    def refresh_gallery(self, force: bool = False):
        """
        Scans the gallery directory and computes embeddings for all VIS images.
        Loads from disk cache when available and gallery has not changed.
        Pass force=True to bypass the cache and recompute everything.
        Expected structure: gallery/identity_name/image.jpg
        """
        self.embeddings = []
        self.skipped_identities = []
        if not os.path.exists(self.gallery_dir):
            return

        if not force and self._load_cache():
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
        self._save_cache()

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
