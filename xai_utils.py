import torch
import torch.nn as nn
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Backward pass: we use the max activation as the target
        # For face recognition (embeddings), we can use the norm or mean
        score = output.mean()
        self.model.zero_grad()
        score.backward()

        # Weight the feature maps
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().numpy(), 0)
        heatmap /= np.max(heatmap)
        
        return heatmap

def apply_heatmap(image_np, heatmap):
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap on image
    superimposed_img = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return superimposed_img

def get_base64_heatmap(face_img, model, target_layer, transform, device):
    """
    Generate a Grad-CAM heatmap and return it as a base64 string.
    """
    try:
        # Prepare input
        input_tensor = transform(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        cam = GradCAM(model, target_layer)
        heatmap_raw = cam.generate_heatmap(input_tensor)
        
        overlay = apply_heatmap(face_img, heatmap_raw)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', overlay)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    except Exception as e:
        print(f"XAI Error: {e}")
        return None
