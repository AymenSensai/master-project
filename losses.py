import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Computes Triplet Loss with online hard negative mining.
    For a given batch of embeddings and labels, it finds the hardest positive 
    and hardest negative for each anchor.
    """
    def __init__(self, margin: float = 0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: Shape (B, D) where B is batch size and D is embedding dimension
        labels: Shape (B,) containing identity labels
        """
        # Compute pairwise distance matrix
        pwd = torch.cdist(embeddings, embeddings, p=2)
        
        # Create masks for positives and negatives (exclude self on diagonal)
        labels = labels.view(-1, 1)
        eye = torch.eye(pwd.size(0), dtype=torch.bool, device=embeddings.device)
        mask_pos = (labels == labels.t()) & ~eye
        mask_neg = labels != labels.t()

        # For each anchor, find the hardest positive (max distance) and hardest negative (min distance)
        dist_ap = torch.zeros(pwd.size(0), device=embeddings.device)
        dist_an = torch.zeros(pwd.size(0), device=embeddings.device)

        for i in range(pwd.size(0)):
            pos_distances = pwd[i][mask_pos[i]]
            dist_ap[i] = pos_distances.max() if pos_distances.numel() > 0 else 0.0

            neg_distances = pwd[i][mask_neg[i]]
            dist_an[i] = neg_distances.min() if neg_distances.numel() > 0 else 0.0
            
        y = torch.ones_like(dist_an)
        
        # loss = max(0, dist_ap - dist_an + margin)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class CrossSpectralLoss(nn.Module):
    """
    Combined Loss for Cross Spectral Face Recognition.
    L = lambda_softmax * SoftmaxLoss + lambda_triplet * TripletLoss
    """
    def __init__(self, margin: float = 0.3, lambda_softmax: float = 1.0, lambda_triplet: float = 1.0):
        super(CrossSpectralLoss, self).__init__()
        self.triplet_loss = TripletLoss(margin=margin)
        self.cross_entropy = nn.CrossEntropyLoss()
        
        self.lambda_softmax = lambda_softmax
        self.lambda_triplet = lambda_triplet

    def forward(self, embeddings: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
        """
        embeddings: Output of the L2 normalization layer
        logits: Output of the classifier FC layer
        labels: Ground truth identity labels
        """
        loss_ce = self.cross_entropy(logits, labels) if logits is not None else torch.tensor(0.0).to(embeddings.device)
        loss_triplet = self.triplet_loss(embeddings, labels)
        
        total_loss = self.lambda_softmax * loss_ce + self.lambda_triplet * loss_triplet
        
        return total_loss, loss_ce, loss_triplet

if __name__ == "__main__":
    # Test losses
    B = 8
    D = 512
    C = 50
    embeds = torch.randn(B, D)
    logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))
    
    criterion = CrossSpectralLoss()
    total, ce, trip = criterion(embeds, logits, labels)
    
    print(f"Total Loss: {total.item():.4f} (CE: {ce.item():.4f}, Triplet: {trip.item():.4f})")
