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

app = Flask(__name__)
CORS(app)

# Configuration
CHECKPOINT_PATH = 'checkpoints/model_best.pth'
IMG_SIZE = 112
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
            
        threshold = 0.6 
        match = similarity > threshold
        
        return jsonify({
            'similarity': round(similarity, 4),
            'match': bool(match),
            'threshold': threshold
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
        # Compute Embedding for Recognition
        tensor = preprocess_image(img_bytes)
        with torch.no_grad():
            probe_embed = model(tensor).cpu().numpy()[0]
            
        results = gallery.search(probe_embed, top_k=3)
        
        # Detect Face for Highlighting
        # Convert bytes to numpy array for OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_box = None
        if len(faces) > 0:
            # Take the largest face if multiple detected
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            # Convert to list for JSON serialization and normalize by image size if needed,
            # but for now we'll send raw pixels and image dimensions.
            face_box = {
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'img_w': int(img.shape[1]),
                'img_h': int(img.shape[0])
            }

        if not results:
            return jsonify({
                'matched': False,
                'message': 'Gallery is empty',
                'face_box': face_box
            })
            
        best_match = results[0]
        threshold = 0.55
        
        return jsonify({
            'matched': best_match['similarity'] > threshold,
            'identity': best_match['identity'],
            'similarity': round(best_match['similarity'], 4),
            'top_matches': results,
            'face_box': face_box
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/identities', methods=['GET'])
def get_identities():
    return jsonify({
        'identities': gallery.get_identities()
    })

@app.route('/api/refresh_gallery', methods=['POST'])
def refresh_gallery():
    gallery.refresh_gallery()
    return jsonify({'status': 'success', 'count': len(gallery.embeddings)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
