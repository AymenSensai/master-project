import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time

from utils import load_config, setup_logger, save_checkpoint, set_seed
from dataset import get_dataloader
from model import build_model
from losses import CrossSpectralLoss

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, logger):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_triplet = 0.0
    
    start_time = time.time()
    
    for i, (images, labels, domains) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        domains = domains.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        embeds, logits = model(images, domain=domains)
        
        # Losses
        loss, loss_ce, loss_triplet = criterion(embeds, logits, labels, domains)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_ce += loss_ce.item()
        running_triplet += loss_triplet.item()
        
        if (i + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch}], Step [{i+1}/{len(dataloader)}], "
                        f"Loss: {loss.item():.4f} "
                        f"(CE: {loss_ce.item():.4f}, Triplet: {loss_triplet.item():.4f})")
                        
    epoch_loss = running_loss / len(dataloader)
    epoch_time = time.time() - start_time
    logger.info(f"Epoch [{epoch}] completed in {epoch_time:.2f}s, Average Loss: {epoch_loss:.4f}")
    return epoch_loss

@torch.no_grad()
def compute_rank1(model, dataloader, device):
    model.eval()
    all_embeds, all_labels, all_domains = [], [], []

    for images, labels, domains in dataloader:
        embeds = model(images.to(device), domain=domains.to(device))
        all_embeds.append(embeds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_domains.extend(domains.numpy())

    all_embeds = np.vstack(all_embeds)
    all_labels = np.array(all_labels)
    all_domains = np.array(all_domains)

    vis_mask = all_domains == 0
    nir_mask = all_domains == 1
    vis_embeds, vis_labels = all_embeds[vis_mask], all_labels[vis_mask]
    nir_embeds, nir_labels = all_embeds[nir_mask], all_labels[nir_mask]

    if len(vis_embeds) == 0 or len(nir_embeds) == 0:
        return 0.0

    sim_matrix = np.dot(nir_embeds, vis_embeds.T)
    predicted_labels = vis_labels[np.argmax(sim_matrix, axis=1)]
    model.train()
    return float(np.sum(predicted_labels == nir_labels) / len(nir_labels))


def main(config_path: str, resume: bool = False):
    config = load_config(config_path)
    
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    
    logger = setup_logger("cross_spectral_train.log")
    logger.info(f"Starting training on {device}")
    
    # 1. Dataset
    train_loader = get_dataloader(
        root_dir=config['dataset']['root_dir'],
        split='train',
        batch_size=config['train']['batch_size'],
        img_size=config['dataset']['img_size'],
        num_workers=config['dataset']['num_workers'],
    )

    val_loader = get_dataloader(
        root_dir=config['dataset']['root_dir'],
        split='test',
        batch_size=config['eval']['batch_size'],
        img_size=config['dataset']['img_size'],
        num_workers=config['dataset']['num_workers'],
    )
    
    num_classes = len(train_loader.dataset.label_to_idx)
    logger.info(f"Training on {num_classes} identities.")
    
    # 2. Model
    model = build_model(
        embedding_dim=512, 
        pretrained=(not resume), 
        num_classes=num_classes
    )
    model = model.to(device)
    
    # 3. Loss & Optimizer
    criterion = CrossSpectralLoss(
        margin=config['train']['margin'],
        lambda_softmax=config['train']['lambda_softmax'],
        lambda_triplet=config['train']['lambda_triplet']
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['train']['learning_rate'], 
        weight_decay=config['train']['weight_decay']
    )
    
    scheduler = StepLR(
        optimizer, 
        step_size=config['train']['step_size'], 
        gamma=config['train']['gamma']
    )
    
    save_dir = config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    start_epoch = 1
    best_rank1 = 0.0

    # Resume logic
    checkpoint_path = os.path.join(save_dir, "model_best.pth")
    if resume and os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            for _ in range(checkpoint['epoch']):
                scheduler.step()

        start_epoch = checkpoint['epoch'] + 1
        best_rank1 = checkpoint.get('rank1', 0.0)
        logger.info(f"Resuming from epoch {start_epoch}, best Rank-1: {best_rank1 * 100:.2f}%")

    # 4. Training Loop
    epochs = config['train']['epochs']
    patience = config['train'].get('early_stopping_patience', 20)
    patience_counter = 0

    for epoch in range(start_epoch, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        scheduler.step()

        rank1 = compute_rank1(model, val_loader, device)
        logger.info(f"Epoch [{epoch}] Val Rank-1: {rank1 * 100:.2f}% (best: {best_rank1 * 100:.2f}%)")

        is_best = rank1 > best_rank1
        if is_best:
            best_rank1 = rank1
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'rank1': rank1,
            'num_classes': num_classes
        }, is_best, save_dir, filename=f"checkpoint_epoch_{epoch}.pth")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}: no Rank-1 improvement for {patience} epochs.")
            break

    logger.info(f"Training complete. Best Rank-1: {best_rank1 * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cross Spectral Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    args = parser.parse_args()
    
    main(args.config, resume=args.resume)
