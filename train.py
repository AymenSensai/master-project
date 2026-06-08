import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
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
    
    logger.info(f"Epoch [{epoch}] starting — {len(dataloader)} batches total.")
    for i, (images, labels, domains) in enumerate(dataloader):
        logger.info(f"Epoch [{epoch}] batch {i+1} loaded, running forward pass...")
        images = images.to(device)
        labels = labels.to(device)
        domains = domains.to(device)

        optimizer.zero_grad()

        embeds, logits = model(images, domain=domains)
        loss, loss_ce, loss_triplet = criterion(embeds, logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce += loss_ce.item()
        running_triplet += loss_triplet.item()

        logger.info(f"Epoch [{epoch}], Step [{i+1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f} "
                    f"(CE: {loss_ce.item():.4f}, Triplet: {loss_triplet.item():.4f})")
                        
    epoch_loss = running_loss / len(dataloader)
    epoch_time = time.time() - start_time
    logger.info(f"Epoch [{epoch}] completed in {epoch_time:.2f}s, Average Loss: {epoch_loss:.4f}")
    return epoch_loss

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
        crop_faces=False,
    )
    
    num_classes = len(train_loader.dataset.label_to_idx)
    logger.info(f"Training on {num_classes} identities.")
    
    # 2. Model
    logger.info("Building model...")
    model = build_model(
        embedding_dim=512,
        pretrained=(not resume),
        num_classes=num_classes
    )
    logger.info("Moving model to device...")
    model = model.to(device)
    logger.info("Model ready.")

    # 3. Loss & Optimizer
    logger.info("Setting up loss, optimizer, scheduler...")
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
    logger.info("Setup complete. Starting training loop...")
    
    save_dir = config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    start_epoch = 1
    best_loss = float('inf')

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
            # Manually step scheduler to catch up if state wasn't saved
            for _ in range(checkpoint['epoch']):
                scheduler.step()
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # 4. Training Loop
    epochs = config['train']['epochs']
    for epoch in range(start_epoch, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        scheduler.step()
        
        # Save Checkpoint
        is_best = loss < best_loss
        if is_best:
            best_loss = loss
            
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'num_classes': num_classes
        }, is_best, save_dir, filename=f"checkpoint_epoch_{epoch}.pth")

    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cross Spectral Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    args = parser.parse_args()
    
    main(args.config, resume=args.resume)
