"""
Run in Colab to find the right similarity threshold for the new model.
Usage: python3 test_threshold.py
"""

import torch
import numpy as np
from dataset import get_dataloader
from model import build_model
from utils import load_config

config = load_config('config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load best checkpoint
checkpoint = torch.load('checkpoints/model_best.pth', map_location=device, weights_only=False)
num_classes = checkpoint.get('num_classes', 0)
model = build_model(embedding_dim=512, pretrained=False, num_classes=num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"Loaded checkpoint from epoch {checkpoint.get('epoch')}")

# Extract embeddings from test set
test_loader = get_dataloader(
    root_dir=config['dataset']['root_dir'],
    split='test',
    batch_size=config['eval']['batch_size'],
    img_size=config['dataset']['img_size'],
    num_workers=0,
    crop_faces=False,
)

all_embeds, all_labels, all_domains = [], [], []
with torch.no_grad():
    for images, labels, domains in test_loader:
        embeds = model(images.to(device), domain=domains.to(device))
        all_embeds.append(embeds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_domains.extend(domains.numpy())

all_embeds = np.vstack(all_embeds)
all_labels = np.array(all_labels)
all_domains = np.array(all_domains)

vis_mask = all_domains == 0
nir_mask = all_domains == 1

vis_embeds = all_embeds[vis_mask]
vis_labels = all_labels[vis_mask]
nir_embeds = all_embeds[nir_mask]
nir_labels = all_labels[nir_mask]

# Compute similarity matrix
sim_matrix = np.dot(nir_embeds, vis_embeds.T)

# Collect genuine (same person) and impostor (different person) scores
genuine_scores = []
impostor_scores = []

for i in range(len(nir_labels)):
    for j in range(len(vis_labels)):
        score = sim_matrix[i, j]
        if nir_labels[i] == vis_labels[j]:
            genuine_scores.append(score)
        else:
            impostor_scores.append(score)

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print(f"\n--- Similarity Score Statistics ---")
print(f"Genuine  pairs — min: {genuine_scores.min():.4f} | mean: {genuine_scores.mean():.4f} | max: {genuine_scores.max():.4f}")
print(f"Impostor pairs — min: {impostor_scores.min():.4f} | mean: {impostor_scores.mean():.4f} | max: {impostor_scores.max():.4f}")

# Find best threshold by sweeping
print(f"\n--- Threshold Sweep ---")
print(f"{'Threshold':>10} | {'TAR (recall)':>12} | {'FAR':>8} | {'Accuracy':>10}")
print("-" * 50)
for threshold in np.arange(0.1, 0.9, 0.05):
    tp = np.sum(genuine_scores >= threshold)
    fn = np.sum(genuine_scores < threshold)
    fp = np.sum(impostor_scores >= threshold)
    tn = np.sum(impostor_scores < threshold)

    tar = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)

    print(f"{threshold:>10.2f} | {tar*100:>11.2f}% | {far*100:>7.2f}% | {acc*100:>9.2f}%")

# Best threshold = highest accuracy
best_thresh = None
best_acc = 0
for threshold in np.arange(0.1, 0.9, 0.01):
    tp = np.sum(genuine_scores >= threshold)
    fn = np.sum(genuine_scores < threshold)
    fp = np.sum(impostor_scores >= threshold)
    tn = np.sum(impostor_scores < threshold)
    acc = (tp + tn) / (tp + tn + fp + fn)
    if acc > best_acc:
        best_acc = acc
        best_thresh = threshold

print(f"\nBest threshold: {best_thresh:.2f} → Accuracy: {best_acc*100:.2f}%")
print(f"Use this value in app.py for both thresholds.")
