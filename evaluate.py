import argparse
import os
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from dataset import get_dataloader
from model import build_model
from utils import setup_logger, compute_tar_at_far, plot_tsne, load_config

@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """
    Extracts embeddings, labels, and domain information.
    """
    model.eval()
    all_embeds = []
    all_labels = []
    all_domains = []
    
    with torch.no_grad():
        for images, labels, domains in dataloader:
            images = images.to(device)
            # Eval mode only returns embeddings
            embeds = model(images, domain=domains.to(device))
            
            all_embeds.append(embeds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_domains.extend(domains.numpy())
            
    all_embeds = np.vstack(all_embeds)
    all_labels = np.array(all_labels)
    all_domains = np.array(all_domains)
    
    return all_embeds, all_labels, all_domains

def evaluate(config_path: str, checkpoint_path: str, split: str = 'test'):
    config = load_config(config_path)
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    
    logger = setup_logger("cross_spectral_eval.log")
    logger.info(f"Starting evaluation on {device} using {split} split")
    
    # 1. Dataset (Selected split)
    test_loader = get_dataloader(
        root_dir=config['dataset']['root_dir'],
        split=split,
        batch_size=config['eval']['batch_size'],
        img_size=config['dataset']['img_size'],
        num_workers=config['dataset']['num_workers'],
    )
    
    if len(test_loader.dataset) == 0:
        logger.error(f"Dataset for split '{split}' is empty! Check your folder structure.")
        return
    
    # 2. Model & Checkpoint
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    num_classes = checkpoint.get('num_classes', 0)
    
    model = build_model(embedding_dim=512, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # 3. Extract Embeddings
    logger.info("Extracting embeddings for test set...")
    embeds, labels, domains = extract_embeddings(model, test_loader, device)
    
    # Separate VIS (gallery) and NIR (probe)
    # Standard cross-spectral evaluation: Match NIR probes to VIS gallery
    vis_mask = domains == 0
    nir_mask = domains == 1
    
    vis_embeds = embeds[vis_mask]
    vis_labels = labels[vis_mask]
    
    nir_embeds = embeds[nir_mask]
    nir_labels = labels[nir_mask]
    
    logger.info(f"Found {len(vis_embeds)} VIS images and {len(nir_embeds)} NIR images.")
    
    if len(vis_embeds) == 0 or len(nir_embeds) == 0:
        logger.error("Not enough data to perform cross-spectral evaluation. Generate visualizations on all data and exiting.")
        plot_tsne(embeds, labels, domains, save_path="tsne_eval.png")
        return
        
    # 4. Identification (Rank-1 Accuracy)
    # For every NIR probe, find the nearest VIS gallery embedding
    logger.info("Computing Rank-1 Accuracy...")
    
    # Compute Cosine Similarity matrix (N_probes x N_gallery)
    # Since embeds are L2 normalized, cosine similarity is just the dot product
    sim_matrix = np.dot(nir_embeds, vis_embeds.T)
    
    # Find argmax for each probe
    rank1_preds = np.argmax(sim_matrix, axis=1)
    predicted_labels = vis_labels[rank1_preds]
    
    correct = np.sum(predicted_labels == nir_labels)
    rank1_acc = correct / len(nir_labels)
    logger.info(f"Rank-1 Accuracy: {rank1_acc * 100:.2f}%")
    
    # 5. Verification (ROC and TAR@FAR)
    logger.info("Computing Verification Metrics...")
    
    # Construct verification pairs from the similarity matrix
    # Label is 1 if probe_label == gallery_label, else 0
    pair_scores = sim_matrix.flatten()
    
    # Expand nir_labels and vis_labels to create the ground truth pair matrix
    nir_labels_exp = np.repeat(nir_labels[:, np.newaxis], len(vis_labels), axis=1)
    vis_labels_exp = np.repeat(vis_labels[np.newaxis, :], len(nir_labels), axis=0)
    pair_labels = (nir_labels_exp == vis_labels_exp).astype(int).flatten()
    
    # Calculate TAR @ specific FARs
    for far_target in config['eval']['far_targets']:
        tar = compute_tar_at_far(pair_scores, pair_labels, far_target)
        logger.info(f"TAR @ FAR={far_target}: {tar * 100:.2f}%")
        
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(pair_labels, pair_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic: Cross-Spectral ({split})')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{split}.png')
    logger.info(f"Saved ROC curve to roc_curve_{split}.png")
    
    # 6. Visualization
    plot_tsne(embeds, labels, domains, save_path=f"tsne_eval_{split}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cross Spectral Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate on")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, split=args.split)
