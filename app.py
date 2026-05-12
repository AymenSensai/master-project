import os
import io
import random
import base64
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2

from model import build_model
from gallery_manager import GalleryManager
from xai_utils import get_base64_heatmap
from utils import load_config, detect_and_crop_face

import logging

# Configure logging to console
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Configuration
config = load_config('config.yaml')
CHECKPOINT_PATH = 'checkpoints/model_best.pth'
IMG_SIZE = config['dataset']['img_size']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = config['dataset']['root_dir']
VIS_GALLERY_PATH = os.path.join(DATA_ROOT, 'VIS')
NIR_GALLERY_PATH = os.path.join(DATA_ROOT, 'NIR')
_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# Initialize Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    logging.error("Failed to load haarcascade_frontalface_default.xml")

# Load Model
def load_checkpoint(path, device):
    print(f"System: Initializing computation on {device}")
    if not os.path.exists(path):
        print(f"Warning: Checkpoint not found at {path}. Using untrained ResNet-18 for Demo Mode.")
        model = build_model(embedding_dim=512, pretrained=True)
        model.to(device)
        model.eval()
        return model
    
    try:
        checkpoint = torch.load(path, map_location=device)
        num_classes = checkpoint.get('num_classes', 0)
        model = build_model(embedding_dim=512, pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Status: Success! Loaded trained checkpoint from {path}")
        return model
    except Exception as e:
        print(f"Error: Failed to load checkpoint: {e}. Falling back to untrained model.")
        return build_model(embedding_dim=512, pretrained=True).to(device)

model = load_checkpoint(CHECKPOINT_PATH, DEVICE)

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    return image


# Initialize Gallery
gallery = GalleryManager(model, preprocess_image, DEVICE, gallery_dir=VIS_GALLERY_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(DATA_ROOT, filename)

@app.route('/api/gallery', methods=['GET'])
def get_gallery():
    # Construct relative paths for the frontend
    gallery_items = []
    for item in gallery.embeddings:
        # item['path'] is absolute or relative to project root
        # We need a path relative to DATA_ROOT to use with /data/ route
        rel_path = os.path.relpath(item['path'], DATA_ROOT)
        gallery_items.append({
            'identity': item['identity'],
            'image_url': f'/data/{rel_path}'
        })
    return jsonify({'gallery': gallery_items})

@app.route('/api/compare', methods=['POST'])
def compare():
    if 'vis_image' not in request.files or 'nir_image' not in request.files:
        return jsonify({'error': 'Deux images (vis_image et nir_image) sont requises'}), 400
    
    vis_file = request.files['vis_image']
    nir_file = request.files['nir_image']
    
    vis_data = vis_file.read()
    nir_data = nir_file.read()
    
    try:
        nparr_vis = np.frombuffer(vis_data, np.uint8)
        img_vis = cv2.imdecode(nparr_vis, cv2.IMREAD_COLOR)
        
        nparr_nir = np.frombuffer(nir_data, np.uint8)
        img_nir = cv2.imdecode(nparr_nir, cv2.IMREAD_COLOR)

        if img_vis is None or img_nir is None:
            return jsonify({'error': 'Images invalides'}), 400

        # Detect and crop faces for both images to ensure model consistency
        face_vis, _ = detect_and_crop_face(img_vis, padding=0.15)
        face_nir, _ = detect_and_crop_face(img_nir, padding=0.15)

        vis_tensor = transform(Image.fromarray(cv2.cvtColor(face_vis, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
        nir_tensor = transform(Image.fromarray(cv2.cvtColor(face_nir, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            vis_embed = model(vis_tensor)
            nir_embed = model(nir_tensor)
            
            similarity = torch.mm(vis_embed, nir_embed.t()).item()
            
        threshold = 0.50 
        match = similarity > threshold
        
        face_crop_b64 = None
        heatmap_b64   = None
        if match:
            nparr   = np.frombuffer(nir_data, np.uint8)
            nir_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if nir_img is not None:
                face_img, _ = detect_and_crop_face(nir_img)
                gallery_embed = vis_embed.cpu().numpy()[0]
                heatmap_b64, face_crop_b64 = get_base64_heatmap(
                    face_img, model, transform, DEVICE, gallery_embed
                )

        return jsonify({
            'similarity': round(similarity, 4),
            'match': bool(match),
            'threshold': threshold,
            'face_crop': face_crop_b64,
            'heatmap':   heatmap_b64,
            'gallery_image_url': f'/data/{os.path.relpath(vis_file.filename, DATA_ROOT)}' if match else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'Une image est requise'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    try:
        # 1. Face Detection (Unified)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Image invalide'}), 400

        face_img, face_box = detect_and_crop_face(img, padding=0.15)
        face_detected = face_box is not None
        
        face_crop_b64 = None
        if face_detected:
            _, buf = cv2.imencode('.jpg', face_img)
            face_crop_b64 = base64.b64encode(buf).decode('utf-8')
            app.logger.info(f"Recognize: Face detected at [{face_box['x']}, {face_box['y']}]")
        else:
            app.logger.info("Recognize: No face detected, using full frame")

        # Embed the face crop when available — the full frame includes
        # background, hands, phone borders which hurt matching accuracy.
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probe_embed = model(tensor).cpu().numpy()[0]

        results = gallery.search(probe_embed, top_k=1)

        if results:
            best_match = results[0]
            threshold = 0.60
            app.logger.info(f"DEBUG: Current recognition threshold is {threshold}")
            similarity = float(best_match['similarity'])
            matched = similarity >= threshold
            face_detected = face_box is not None

            app.logger.info(
                f"Recognize: {'Match' if matched else 'No match'} – "
                f"{best_match['identity']} ({similarity:.3f}) face={face_detected}"
            )

            heatmap_b64 = None
            xai_crop_b64 = None
            if matched and face_detected:
                heatmap_b64, xai_crop_b64 = get_base64_heatmap(
                    face_img, model, transform, DEVICE, best_match['embedding']
                )

            # Construction of gallery image URL
            rel_gallery_path = os.path.relpath(best_match['path'], DATA_ROOT)
            gallery_url = f'/data/{rel_gallery_path}'

            return jsonify({
                'matched': matched,
                'face_detected': face_detected,
                'identity': best_match['identity'],
                'similarity': round(float(similarity), 4),
                'face_box': face_box,
                'face_crop': xai_crop_b64 or face_crop_b64,
                'heatmap':   heatmap_b64,
                'gallery_image_url': gallery_url
            })

        
    except Exception as e:
        app.logger.error(f"Recognize Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/identities', methods=['GET'])
def get_identities():
    identities = gallery.get_identities()
    return jsonify({'identities': sorted(identities)})

@app.route('/api/refresh_gallery', methods=['POST'])
def refresh_gallery():
    gallery.refresh_gallery()
    return jsonify({'status': 'success', 'count': len(gallery.embeddings)})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        vis_root = os.path.join(DATA_ROOT, 'VIS')
        nir_root = os.path.join(DATA_ROOT, 'NIR')
        
        vis_count = sum([len(files) for r, d, files in os.walk(vis_root) if files])
        nir_count = sum([len(files) for r, d, files in os.walk(nir_root) if files])
        identities_count = len(gallery.get_identities())
        
        return jsonify({
            'identities': identities_count,
            'vis_images': vis_count,
            'nir_images': nir_count,
            'device': str(DEVICE).upper(),
            'status': 'Online'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _list_nir_images(identity):
    id_dir = os.path.join(NIR_GALLERY_PATH, identity)
    if not os.path.isdir(id_dir):
        return []
    return [os.path.join(id_dir, f) for f in os.listdir(id_dir)
            if f.lower().endswith(_IMG_EXTS)]

@app.route('/api/challenge/new', methods=['GET'])
def challenge_new():
    try:
        gallery_items = list(gallery.embeddings)
        if len(gallery_items) < 2:
            return jsonify({'error': 'Galerie insuffisante (minimum 2 identités requises).'}), 400

        identities_with_nir = [it for it in gallery_items if _list_nir_images(it['identity'])]
        if not identities_with_nir:
            return jsonify({'error': "Aucune image NIR trouvée pour générer un défi."}), 400

        # Choose a target identity for the NIR probe
        target = random.choice(identities_with_nir)
        nir_candidates = _list_nir_images(target['identity'])
        probe_path = random.choice(nir_candidates)

        # Decide if this challenge will be a "Same" or "Different" case
        is_same = random.choice([True, False])
        
        if is_same:
            candidate = target
        else:
            distractor_pool = [it for it in gallery_items if it['identity'] != target['identity']]
            candidate = random.choice(distractor_pool)

        # AI prediction: embed the NIR probe and compare with candidate VIS embedding
        nir_bgr = cv2.imread(probe_path)
        face_img, _ = detect_and_crop_face(nir_bgr, padding=0.15)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probe_embed = model(tensor).cpu().numpy()[0]

        similarity = float(np.dot(probe_embed, candidate['embedding']))
        threshold = 0.50 # Using the same threshold as in /api/compare
        ai_same = similarity >= threshold

        return jsonify({
            'probe_url': f"/data/{os.path.relpath(probe_path, DATA_ROOT)}",
            'candidate_url': f"/data/{os.path.relpath(candidate['path'], DATA_ROOT)}",
            'is_same': is_same,
            'ai_same': ai_same,
            'similarity': round(similarity, 4),
            'target_identity': target['identity'],
            'candidate_identity': candidate['identity']
        })
    except Exception as e:
        app.logger.error(f"Challenge error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
