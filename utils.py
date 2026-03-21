import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(log_file: str = "training.log") -> logging.Logger:
    logger = logging.getLogger("CrossSpectral")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def save_checkpoint(state: Dict[str, Any], is_best: bool, save_dir: str, filename: str = "checkpoint.pth"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(save_dir, "model_best.pth")
        torch.save(state, best_filepath)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_tar_at_far(scores: np.ndarray, labels: np.ndarray, far_target: float) -> float:
    """
    Compute True Acceptance Rate (TAR) at a specific False Acceptance Rate (FAR).
    scores: Pairwise matching scores (e.g. cosine similarity)
    labels: Binary labels where 1 means positive pair (same identity), 0 means negative pair.
    """
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    
    # Sort negative scores in descending order
    neg_scores = np.sort(neg_scores)[::-1]
    
    # Find the threshold to achieve the target FAR
    threshold_idx = int(len(neg_scores) * far_target)
    # Handle edge case where threshold_idx is out of bounds
    if threshold_idx >= len(neg_scores):
        threshold_idx = len(neg_scores) - 1
        
    threshold = neg_scores[threshold_idx]
    
    # Calculate TAR
    tar = np.sum(pos_scores >= threshold) / len(pos_scores)
    return tar

def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, domains: np.ndarray, save_path: str = "tsne_plot.png"):
    """
    Visualize embeddings using t-SNE.
    embeddings: Shape (N, D)
    labels: Identity labels Shape (N,)
    domains: Domain labels (0 for VIS, 1 for NIR) Shape (N,)
    """
    print("Computing t-SNE... This may take a moment.")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    # Try to plot a few distinct identities
    unique_labels = np.unique(labels)[:10] # Plot up to 10 identities
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = {0: 'o', 1: '^'} # Circle for VIS, Triangle for NIR
    
    for idx, identity in enumerate(unique_labels):
        for domain in [0, 1]:
            mask = (labels == identity) & (domains == domain)
            if np.any(mask):
                domain_name = "VIS" if domain == 0 else "NIR"
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                            color=colors[idx], marker=markers[domain], 
                            label=f'ID {identity} - {domain_name}', alpha=0.7)
                
    plt.title("t-SNE Visualization of Shared Embedding Space")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")
