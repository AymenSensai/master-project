import os
import io
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

import cv2

from model import build_model
from gallery_manager import GalleryManager
from xai_utils import get_base64_heatmap

import logging

# Configure logging to file
logging.basicConfig(filename='app_debug.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Configuration
CHECKPOINT_PATH = 'checkpoints/model_best.pth'
IMG_SIZE = 112
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    logging.error("Failed to load haarcascade_frontalface_default.xml")

# Load Model
def load_checkpoint(path, device):
    if not os.path.exists(path):
        print(f"Warning: Checkpoint not found at {path}. Using untrained model for demo.")
        return build_model(embedding_dim=512, pretrained=True).to(device)
    
    checkpoint = torch.load(path, map_location=device)
    num_classes = checkpoint.get('num_classes', 0)
    model = build_model(embedding_dim=512, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {path}")
    return model

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
gallery = GalleryManager(model, preprocess_image, DEVICE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compare', methods=['POST'])
def compare():
    if 'vis_image' not in request.files or 'nir_image' not in request.files:
        return jsonify({'error': 'Two images (vis_image and nir_image) are required'}), 400
    
    vis_file = request.files['vis_image']
    nir_file = request.files['nir_image']
    
    try:
        vis_tensor = preprocess_image(vis_file.read())
        nir_tensor = preprocess_image(nir_file.read())
        
        with torch.no_grad():
            vis_embed = model(vis_tensor)
            nir_embed = model(nir_tensor)
            
            similarity = torch.mm(vis_embed, nir_embed.t()).item()
            
        threshold = 0.50 
        match = similarity > threshold
        
        # Generate XAI heatmap for the NIR image
        heatmap_b64 = None
        if match:
            # We use layer4 as target
            target_layer = model.features[7]
            nparr = np.frombuffer(nir_file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                heatmap_b64 = get_base64_heatmap(img, model, target_layer, transform, DEVICE)

        return jsonify({
            'similarity': round(similarity, 4),
            'match': bool(match),
            'threshold': threshold,
            'heatmap': heatmap_b64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    try:
        # 1. Face Detection (Haar Cascade)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            app.logger.error("Recognize: Failed to decode image")
            return jsonify({'error': 'Invalid image'}), 400
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sensitive detection for webcam
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
        
        face_box = None
        face_img = None if img is None else img.copy() # Default to full image
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]
            face_box = {
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'img_w': int(img.shape[1]), 'img_h': int(img.shape[0])
            }
            app.logger.info(f"Recognize: Face detected at [{x}, {y}]")
        else:
            app.logger.info("Recognize: No face detected")

        # 2. Recognition Logic
        # Preprocess the original image for recognition if needed, 
        # or use the face crop if detected for better accuracy.
        img_to_process = face_img if face_img is not None else img

        tensor = preprocess_image(img_bytes) # This uses the full image from bytes
        with torch.no_grad():
            probe_embed = model(tensor).cpu().numpy()[0]
            
        results = gallery.search(probe_embed, top_k=1)
        heatmap_b64 = None

        if results:
            best_match = results[0]
            threshold = 0.50
            similarity = float(best_match['similarity'])
            matched = similarity > threshold
            
            if matched:
                app.logger.info(f"Recognize: Match! {best_match['identity']} ({similarity:.2f})")
                # Generate XAI Heatmap for the detected face OR full image
                target_layer = model.features[7]
                heatmap_b64 = get_base64_heatmap(img_to_process, model, target_layer, transform, DEVICE)

            return jsonify({
                'matched': matched,
                'identity': best_match['identity'],
                'similarity': round(float(similarity), 4),
                'face_box': face_box,
                'heatmap': heatmap_b64
            })

        
    except Exception as e:
        app.logger.error(f"Recognize Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/identities', methods=['GET'])
def get_identities():
    identities = gallery.get_identities()
    return jsonify({'identities': sorted(identities)})

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    identities = gallery.get_identities()
    
    # Demographic mapping for the demo
    total = len(identities)
    
    # Simple heuristic based on the names we assigned
    asian_names = ['Wei', 'Li', 'Hao', 'Mei', 'Min-Jun', 'Xiao', 'Zhi', 'Yuki', 'Hiroshi', 'Kenji', 'Sun-Hee', 'Sato', 'Tanaka', 'Chen', 'Park']
    female_names = ['Alice', 'Chloe', 'Elena', 'Katherine', 'Mei', 'Sarah', 'Sun-Hee', 'Tasha', 'Xiao', 'Sarah', 'Priya']
    
    asian_count = sum(1 for name in identities if any(an in name for an in asian_names))
    female_count = sum(1 for name in identities if any(fn in name for fn in female_names))
    male_count = total - female_count
    
    return jsonify({
        'total_identities': total,
        'gender_distribution': {
            'Male': male_count,
            'Female': female_count
        },
        'ethnicity_distribution': {
            'Asian': asian_count,
            'Other': total - asian_count
        },
        'system_activity': [
            {'date': '2026-03-21', 'count': 45},
            {'date': '2026-03-22', 'count': 120},
            {'date': '2026-03-23', 'count': 89}
        ]
    })

@app.route('/api/refresh_gallery', methods=['POST'])
def refresh_gallery():
    gallery.refresh_gallery()
    return jsonify({'status': 'success', 'count': len(gallery.embeddings)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
