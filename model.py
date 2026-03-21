import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TwoStreamResNet(nn.Module):
    """
    Two-stream ResNet-18 for Cross-Spectral Face Recognition.
    Shares convolutional weights across VIS and NIR domains,
    but can be configured to use Domain-Specific Batch Normalization.
    """
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True, num_classes: int = 0):
        super(TwoStreamResNet, self).__init__()
        
        # Load pre-trained ResNet-18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = resnet18(weights=weights)
        
        # We share the feature extractor for both streams
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool
        )
        
        # Extract feature dimension (512 for ResNet-18)
        self.feature_dim = base_model.fc.in_features
        
        # Final embedding layer replacing the original classification FC
        self.embedding = nn.Linear(self.feature_dim, embedding_dim)
        
        # Optional Classification layer if training with Softmax
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
            
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the shared backbone.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor, domain: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor (B, 3, H, W)
            domain: Domain labels (0=VIS, 1=NIR). Can be used if domain-specific BN is added later.
                    For this implementation, we use fully shared weights constraint.
        Returns:
            Embeddings of shape (B, embedding_dim)
            (Optional) Classification logits if num_classes > 0
        """
        # Feature extraction
        feats = self.forward_features(x)
        
        # Project to embedding space
        embeds = self.embedding(feats)
        
        # L2 Normalize embeddings to fall on a unit hypersphere
        # This is standard practice in metric learning for face recognition
        normalized_embeds = nn.functional.normalize(embeds, p=2, dim=1)
        
        if self.classifier is not None and self.training:
            logits = self.classifier(normalized_embeds)
            return normalized_embeds, logits
            
        return normalized_embeds

def build_model(embedding_dim: int = 512, pretrained: bool = True, num_classes: int = 0) -> nn.Module:
    """
    Factory function to build the TwoStream ResNet Model.
    """
    model = TwoStreamResNet(embedding_dim=embedding_dim, pretrained=pretrained, num_classes=num_classes)
    return model

if __name__ == "__main__":
    # Test model forward pass
    model = build_model(num_classes=50) # Assuming 50 identities
    x = torch.randn(4, 3, 112, 112) # Batch of 4 images
    domains = torch.tensor([0, 1, 0, 1]) # Mixed VIS/NIR batch
    
    # Train mode (Outputs embeds and logits)
    model.train()
    embeds, logits = model(x, domain=domains)
    print("Train Mode:")
    print(f"Embeddings shape: {embeds.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Eval mode (Outputs only embeds)
    model.eval()
    embeds_eval = model(x, domain=domains)
    print("\nEval Mode:")
    print(f"Embeddings shape: {embeds_eval.shape}")
