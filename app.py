import os
import io
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

# Configure logging to file
logging.basicConfig(filename='app_debug.log', level=logging.INFO, 
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

@app.route('/logo.png')
def get_logo():
    return send_from_directory('/Users/mac/.gemini/antigravity/brain/3015868c-c70f-43e2-900e-a4f9f8de695c', 'spectralface_icon_logo_1778523183111.png')

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
        vis_tensor = preprocess_image(vis_data)
        nir_tensor = preprocess_image(nir_data)
        
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
            threshold = 0.70
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

if __name__ == '__main__':
    app.run(debug=True, port=5001)
