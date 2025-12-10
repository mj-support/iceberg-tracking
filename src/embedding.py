import cv2
from datetime import timedelta
import functools
import logging
import matplotlib.pyplot as plt
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from dataclasses import dataclass
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, Callable, Optional, Any

from utils.helpers import PROJECT_ROOT, extract_candidates, extract_matches, load_icebergs_by_frame, \
    parse_annotations, get_sequences, get_image_ext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)
logger = logging.getLogger(__name__)

"""
Iceberg Similarity Learning and Embedding Generation Pipeline

This module implements a deep learning pipeline for learning and generating
appearance embeddings of icebergs using Vision Transformers (ViT) and Siamese Neural Networks.
The learned embeddings capture iceberg appearance similarity and are used downstream for
multi-object tracking and data association.

Architecture Overview:
    The system uses a Siamese Neural Network with a shared Vision Transformer backbone to
    learn discriminative features that capture iceberg appearance. The network is trained
    on pairs of iceberg crops with binary similarity labels (same iceberg vs different icebergs).

Key Components:
    1. Vision Transformer (ViT): Powerful backbone for extracting visual features
    2. Siamese Network: Learns similarity through shared weights and pairwise training
    3. Combined Loss: Blends contrastive loss with cosine similarity loss
    4. Multi-sequence Dataset: Efficient data loading with in-memory caching
    5. Training Pipeline: Complete orchestration with early stopping and checkpointing

Pipeline Stages:
    Training Phase:
        1. Load multi-sequence datasets with positive/negative pairs
        2. Train Siamese network with combined loss function
        3. Validate using AUC metric on validation set
        4. Early stopping based on validation performance
        5. Save best model checkpoint

    Embedding Generation Phase:
        1. Load trained model
        2. Extract features for all iceberg detections
        3. Cache embeddings for efficient downstream use
        4. Save embeddings per sequence

Technical Details:
    - Vision Transformer Configuration: 6-layer encoder, 6 attention heads, 384-dim embeddings
    - Training: Adam optimizer with learning rate scheduling
    - Loss: α * Contrastive + (1-α) * Cosine, where α=0.7
    - Data Augmentation: Color jitter, flips, rotation for robustness
    - Evaluation: ROC-AUC and Average Precision metrics
"""


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IcebergEmbeddingsConfig:
    """
    Centralized configuration for iceberg similarity learning pipeline.

    This dataclass encapsulates all hyperparameters and settings for the complete
    training and embedding generation pipeline. Using a configuration object ensures
    reproducibility and experimentation with different settings.

    Categories:
        - Data: Dataset paths and pair generation settings
        - Model Architecture: Vision Transformer configuration
        - Training: Optimization and regularization parameters
        - Loss: Contrastive and cosine loss weighting
        - Hardware: GPU/CPU device selection

    Attributes:
        dataset (str): Name/path of the dataset directory (required)

        # Data Configuration
        negative_ratio_train (float): Ratio of negative to positive pairs for training
            Higher values = more negative examples (helps with hard negatives)
        negative_ratio_val (float): Ratio of negative to positive pairs for validation
            Typically higher than training to better assess generalization
        num_workers (int): Number of worker processes for data loading parallelization

        # Model Architecture Parameters (Vision Transformer)
        img_size (int): Input image size in pixels (assumed square)
        patch_size (int): Size of patches for ViT (16 = divide image into 16x16 patches)
        in_channels (int): Number of input channels (3 for RGB images)
        embed_dim (int): Embedding dimension for transformer layers
        depth (int): Number of stacked transformer encoder blocks
        n_heads (int): Number of attention heads in multi-head attention
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension
        dropout (float): Dropout probability for regularization [0, 1]
        feature_dim (int): Final feature dimension for embeddings

        # Training Configuration
        batch_size (int): Training batch size (limited by GPU memory)
        val_batch_size (int): Validation batch size (can be larger since no gradients)
        learning_rate (float): Initial learning rate for Adam optimizer
        weight_decay (float): L2 regularization weight for Adam
        num_epochs (int): Maximum number of training epochs
        patience (int): Early stopping patience (epochs without improvement)
        min_delta (float): Minimum AUC improvement to reset patience counter

        # Loss Function Configuration
        loss_margin (float): Margin for contrastive loss (minimum separation for negatives)
        loss_alpha (float): Weight for combining losses [0, 1]
            α=1.0: pure contrastive, α=0.0: pure cosine, α=0.7: balanced

        # Hardware Configuration
        device (str): PyTorch device ('cuda' for GPU, 'cpu' for CPU)

    Example:
        >>> config = IcebergEmbeddingsConfig(
        ...     dataset="hill/train",
        ...     num_epochs=50,
        ...     learning_rate=1e-4,
        ...     feature_dim=256
        ... )
        >>> trainer = IcebergEmbeddingsTrainer(config)
        >>> trainer.run_complete_pipeline()
    """
    # Data configuration
    dataset: str  # Required field
    negative_ratio_train: float = 1.0
    negative_ratio_val: float = 2.0
    num_workers: int = 4

    # Model configuration - Vision Transformer architecture
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 384
    depth: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    feature_dim: int = 256

    # Training configuration
    batch_size: int = 16
    val_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    patience: int = 8
    min_delta: float = 0.002

    # Loss configuration
    loss_margin: float = 1.0
    loss_alpha: float = 0.7

    # Device configuration
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATASET FOR EMBEDDING GENERATION
# ============================================================================

class IcebergEmbeddingsDataset(Dataset):
    """
    Dataset for loading individual iceberg crops from full frame images.

    This dataset is used during embedding generation (inference) to efficiently
    load and preprocess iceberg crops from detection results. It uses LRU caching
    to avoid repeatedly loading the same full frame images.

    Workflow:
        1. Load full frame image (with LRU caching)
        2. Crop iceberg based on bounding box coordinates
        3. Resize while maintaining aspect ratio
        4. Pad to square target size
        5. Apply transforms (normalization)
        6. Return processed tensor with unique identifier

    Attributes:
        detections (list[dict]): List of detection dictionaries with bbox info
        full_frame_dir (str): Directory containing full frame images
        transform (callable): Image transformation pipeline
        target_size (int): Target size for output crops (square)
        image_ext (str): Image file extension (jpg, png, etc.)

    Methods:
        __len__(): Returns number of detections
        __getitem__(idx): Returns (processed_crop, unique_identifier)
        _load_image(path): Cached image loading
    """

    def __init__(self, detections: list[dict], full_frame_dir: str, transform=None,
                 target_size: int = 224, image_ext: str = 'jpg'):
        """
        Initialize the embeddings dataset.

        Args:
            detections (list[dict]): List of detection dicts with 'frame', 'id', 'bbox' keys
            full_frame_dir (str): Directory containing full frame images
            transform (callable, optional): Transform pipeline for preprocessing
            target_size (int): Target size for square crops (default: 224)
            image_ext (str): Image file extension (default: 'jpg')
        """
        self.detections = detections
        self.full_frame_dir = full_frame_dir
        self.transform = transform
        self.target_size = target_size
        self.image_ext = image_ext

        # Use default ImageNet normalization if no transform provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    @functools.lru_cache(maxsize=32)
    def _load_image(self, img_path: str) -> Image.Image:
        """
        Load an image from disk with LRU caching.

        Uses functools.lru_cache to cache recently loaded images in memory.
        This dramatically reduces disk I/O when multiple icebergs are cropped
        from the same frame. Cache size of 32 means the 32 most recently accessed
        frames are kept in memory.

        Args:
            img_path (str): Full path to the image file

        Returns:
            Image.Image: Loaded PIL image in RGB format
        """
        return Image.open(img_path).convert('RGB')

    def __len__(self) -> int:
        """Return the total number of iceberg detections in the dataset."""
        return len(self.detections)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        Get a preprocessed iceberg crop and its unique identifier.

        This method performs the complete pipeline for extracting and preprocessing
        an iceberg crop from a full frame image. It handles aspect ratio preservation,
        padding, and transformation application.

        Args:
            idx (int): Index of the detection to retrieve

        Returns:
            tuple[torch.Tensor, str]:
                - Preprocessed image tensor of shape [C, H, W]
                - Unique identifier string: "{frame_name}_{iceberg_id}"
        """
        detection = self.detections[idx]
        frame_name = detection['frame']
        iceberg_id = detection['id']

        # Construct paths and identifiers
        img_path = os.path.join(self.full_frame_dir, f"{frame_name}.{self.image_ext}")
        unique_iceberg_name = f"{frame_name}_{iceberg_id}"

        try:
            # Load full image (cached for efficiency)
            full_img = self._load_image(img_path)

            # Extract crop coordinates and crop the iceberg region
            left, top = detection['bb_left'], detection['bb_top']
            right, bottom = left + detection['bb_width'], top + detection['bb_height']
            iceberg_crop = full_img.crop((left, top, right, bottom))

            # Resize while maintaining aspect ratio
            w, h = iceberg_crop.size
            if w > h:
                # Width-dominant: scale to target width
                new_w, new_h = self.target_size, int(h * self.target_size / w)
            else:
                # Height-dominant: scale to target height
                new_h, new_w = self.target_size, int(w * self.target_size / h)

            iceberg_crop = iceberg_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Pad to square with black borders
            padded_crop = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
            paste_x = (self.target_size - new_w) // 2
            paste_y = (self.target_size - new_h) // 2
            padded_crop.paste(iceberg_crop, (paste_x, paste_y))

            # Apply transforms (normalization, augmentation, etc.)
            if self.transform:
                padded_crop = self.transform(padded_crop)

            return padded_crop, unique_iceberg_name

        except Exception as e:
            # Graceful error handling - return black image
            logger.error(f"Error loading detection at index {idx}: {e}")
            black_img = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
            if self.transform:
                black_img = self.transform(black_img)
            return black_img, f"error_{idx}"


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Vision Transformer patch embedding layer.

    Converts an input image into a sequence of patch embeddings suitable for
    transformer processing. This is the first step in the ViT architecture.

    Process:
        1. Divide image into non-overlapping patches (e.g., 16x16 pixels each)
        2. Flatten each patch into a 1D vector
        3. Project to embedding dimension via linear transformation
        4. Add learnable position embeddings
        5. Prepend learnable classification token

    The classification token aggregates information from all patches through
    self-attention and is used as the final image representation.

    Attributes:
        img_size (int): Input image size (assumes square)
        patch_size (int): Size of each patch
        n_patches (int): Total number of patches
        proj (nn.Conv2d): Convolutional projection layer
        cls_token (nn.Parameter): Learnable classification token
        pos_embed (nn.Parameter): Learnable position embeddings

    Args:
        img_size (int): Input image size (square)
        patch_size (int): Patch size (square)
        in_channels (int): Number of input channels (3 for RGB)
        embed_dim (int): Embedding dimension for transformer
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        """Initialize patch embedding layer with learnable parameters."""
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        # Using Conv2d with kernel_size=patch_size and stride=patch_size
        # is equivalent to dividing into patches + linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Classification token (learnable, randomly initialized)
        # Shape: [1, 1, embed_dim] - broadcasted across batch
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Position embeddings (learnable, randomly initialized)
        # Shape: [1, n_patches+1, embed_dim] - one for each patch + cls token
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        """
        Forward pass of patch embedding.

        Args:
            x (torch.Tensor): Input images of shape [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Patch embeddings of shape [batch_size, n_patches+1, embed_dim]
        """
        batch_size = x.shape[0]

        # Project patches to embedding dimension
        x = self.proj(x)  # Shape: [batch, embed_dim, n_patches_h, n_patches_w]
        x = x.flatten(2)  # Shape: [batch, embed_dim, n_patches]
        x = x.transpose(1, 2)  # Shape: [batch, n_patches, embed_dim]

        # Expand cls_token for batch and prepend to sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [batch, n_patches+1, embed_dim]

        # Add position embeddings to give transformer spatial awareness
        x = x + self.pos_embed

        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block with multi-head self-attention and MLP.

    This implements a single layer of the Transformer encoder as described in
    "Attention is All You Need" and used in Vision Transformers. Each block
    consists of two sub-layers with residual connections and layer normalization.

    Architecture:
        1. Multi-head self-attention with residual connection
        2. Layer normalization
        3. Feed-forward MLP with GELU activation and residual connection
        4. Layer normalization

    The residual connections help with gradient flow and training stability,
    while layer normalization stabilizes the hidden state distribution.

    Attributes:
        attention (nn.MultiheadAttention): Multi-head self-attention mechanism
        norm1 (nn.LayerNorm): Layer normalization after attention
        mlp (nn.Sequential): Feed-forward network
        norm2 (nn.LayerNorm): Layer normalization after MLP

    Args:
        embed_dim (int): Embedding dimension
        n_heads (int): Number of attention heads (embed_dim must be divisible by n_heads)
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension
        dropout (float): Dropout probability for regularization
    """

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """Initialize transformer block with attention and MLP layers."""
        super().__init__()

        # Multi-head self-attention mechanism
        # batch_first=True means input/output shape is [batch, seq_len, embed_dim]
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward MLP (typically 4x wider than embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),  # Smooth activation function (better than ReLU for transformers)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass of transformer block.

        Implements the standard transformer encoder layer with pre-normalization
        and residual connections.

        Args:
            x (torch.Tensor): Input sequence of shape [batch_size, seq_len, embed_dim]

        Returns:
            torch.Tensor: Output sequence of same shape [batch_size, seq_len, embed_dim]

        Process:
            1. Self-attention: allows each token to attend to all other tokens
            2. Residual: x = x + attention(x)
            3. Normalize: x = norm(x)
            4. MLP: non-linear transformation
            5. Residual: x = x + mlp(x)
            6. Normalize: x = norm(x)
        """
        # Self-attention sub-layer with residual connection
        attn_output, _ = self.attention(x, x, x)  # Query, Key, Value all come from x
        x = x + attn_output  # Residual connection
        x = self.norm1(x)  # Layer normalization

        # MLP sub-layer with residual connection
        mlp_output = self.mlp(x)
        x = x + mlp_output  # Residual connection
        x = self.norm2(x)  # Layer normalization

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for extracting image feature representations.

    Architecture Overview:
        1. Patch Embedding: Convert image to sequence of patch embeddings
        2. Transformer Encoder: Stack of L transformer blocks
        3. Classification Head: Linear projection to feature space

    Key Design Choices:
        - Uses classification token (cls_token) for image-level features
        - Position embeddings provide spatial information
        - Layer normalization for stable training
        - GELU activations in MLPs

    Attributes:
        config (IcebergEmbeddingsConfig): Configuration object
        patch_embed (PatchEmbedding): Patch embedding layer
        blocks (nn.ModuleList): Stack of transformer encoder blocks
        norm (nn.LayerNorm): Final layer normalization
        head (nn.Linear): Projection to final feature dimension

    Args:
        config (IcebergEmbeddingsConfig): Configuration with model hyperparameters
    """

    def __init__(self, config: IcebergEmbeddingsConfig):
        """Initialize Vision Transformer with configuration."""
        super().__init__()
        self.config = config

        # Patch embedding layer - converts image to token sequence
        self.patch_embed = PatchEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim
        )

        # Stack of L transformer encoder blocks
        # Each block applies self-attention and feed-forward transformations
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                n_heads=config.n_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout
            )
            for _ in range(config.depth)
        ])

        # Final normalization before projection head
        self.norm = nn.LayerNorm(config.embed_dim)

        # Projection head to final feature dimension
        # Maps from transformer embedding space to final feature space
        self.head = nn.Linear(config.embed_dim, config.feature_dim)

    def forward(self, x):
        """
        Forward pass of Vision Transformer.

        Processes an image through the complete ViT pipeline to extract
        a fixed-dimensional feature vector suitable for similarity comparison.

        Args:
            x (torch.Tensor): Input images of shape [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Feature vectors of shape [batch_size, feature_dim]

        Process Flow:
            [B, C, H, W] -> patch_embed -> [B, N+1, D] -> transformers -> [B, N+1, D]
            -> select cls_token -> [B, D] -> projection -> [B, F]

            where:
                B = batch_size
                C = channels
                H, W = image dimensions
                N = number of patches
                D = transformer embed_dim
                F = final feature_dim
        """
        # Step 1: Convert image to sequence of patch embeddings
        x = self.patch_embed(x)  # Shape: [batch, n_patches+1, embed_dim]

        # Step 2: Process through stack of transformer blocks
        for block in self.blocks:
            x = block(x)  # Self-attention + MLP, shape unchanged

        # Step 3: Extract classification token (first token in sequence)
        # The cls_token aggregates information from all patches via self-attention
        x = self.norm(x[:, 0])  # Shape: [batch, embed_dim]

        # Step 4: Project to final feature dimension
        x = self.head(x)  # Shape: [batch, feature_dim]

        return x


class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network for learning image similarity.

    A Siamese network uses weight sharing to learn similarity between pairs
    of inputs. It processes both inputs through identical networks (shared weights)
    and compares their representations in a learned embedding space.

    Architecture:
        Input1, Input2 -> SharedBackbone -> Feature1, Feature2 -> Similarity

    Key Advantages:
        - Weight sharing reduces parameters and improves generalization
        - Learns metric space where similar items are close
        - Can generalize to unseen classes (few-shot learning)
        - Efficient inference (compute embeddings once, compare many times)

    Training Strategy:
        - Positive pairs: Same iceberg in consecutive frames (should be similar)
        - Negative pairs: Different icebergs in consecutive frames (should be dissimilar)
        - Loss: Encourage small distance for positives, large distance for negatives

    Attributes:
        backbone (VisionTransformer): Shared ViT backbone for feature extraction

    Args:
        config (IcebergEmbeddingsConfig): Configuration object

    Methods:
        forward(img1, img2): Extract features for both images
        compute_similarity(feat1, feat2): Compute cosine similarity between features
    """

    def __init__(self, config: IcebergEmbeddingsConfig):
        """Initialize Siamese network with shared ViT backbone."""
        super().__init__()
        # Shared backbone processes both images identically
        self.backbone = VisionTransformer(config)

    def forward(self, img1, img2):
        """
        Forward pass through Siamese network.

        Processes both images through the shared backbone to extract feature vectors.
        Weight sharing ensures consistent feature extraction for both inputs.

        Args:
            img1 (torch.Tensor): First image batch of shape [batch_size, C, H, W]
            img2 (torch.Tensor): Second image batch of shape [batch_size, C, H, W]

        Returns:
            tuple: (features1, features2) where each is [batch_size, feature_dim]
        """
        # Process both images through shared backbone
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        return feat1, feat2

    def compute_similarity(self, feat1, feat2):
        """
        Compute cosine similarity between feature pairs.

        Cosine similarity measures the angle between feature vectors,
        normalized to [-1, 1] where:
            1.0 = identical direction (most similar)
            0.0 = orthogonal (unrelated)
           -1.0 = opposite direction (most dissimilar)

        Args:
            feat1 (torch.Tensor): First set of features [batch_size, feature_dim]
            feat2 (torch.Tensor): Second set of features [batch_size, feature_dim]

        Returns:
            torch.Tensor: Cosine similarity scores [batch_size]

        Formula:
            similarity = (feat1 · feat2) / (||feat1|| * ||feat2||)
        """
        return F.cosine_similarity(feat1, feat2, dim=1)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning similarity in embedding space.

    Contrastive loss encourages the model to:
    1. Place similar pairs close together in embedding space
    2. Push dissimilar pairs at least a margin distance apart

    This creates a metric space where distance correlates with semantic similarity.

    Loss Formula:
        L = (1-Y) * D² + Y * max(0, margin - D)²

        where:
            Y = label (1 for similar, 0 for dissimilar)
            D = euclidean distance between embeddings
            margin = minimum desired separation for dissimilar pairs

    Intuition:
        - Similar pairs (Y=1): Minimize distance D (term 1 is zero)
        - Dissimilar pairs (Y=0): Penalize if distance < margin (term 2 is zero)

    The margin parameter is crucial: too small and the model doesn't separate
    well; too large and optimization becomes difficult.

    Attributes:
        margin (float): Minimum distance for negative pairs

    Args:
        margin (float): Margin value (default: 1.0)
    """

    def __init__(self, margin: float = 1.0):
        """Initialize contrastive loss with specified margin."""
        super().__init__()
        self.margin = margin

    def forward(self, feat1, feat2, labels):
        """
        Compute contrastive loss for a batch of feature pairs.

        Args:
            feat1 (torch.Tensor): First set of features [batch_size, feature_dim]
            feat2 (torch.Tensor): Second set of features [batch_size, feature_dim]
            labels (torch.Tensor): Binary labels [batch_size]
                1 = similar pair, 0 = dissimilar pair

        Returns:
            torch.Tensor: Scalar loss value (mean over batch)

        Computation:
            1. Calculate pairwise Euclidean distances
            2. Apply loss formula separately for positive/negative pairs
            3. Average over batch
        """
        # Compute euclidean distance between feature pairs
        distances = F.pairwise_distance(feat1, feat2)

        # Loss for positive pairs (label=1): should be close (minimize distance²)
        loss_positive = labels * torch.pow(distances, 2)

        # Loss for negative pairs (label=0): should be at least margin apart
        # Only penalize if distance < margin
        loss_negative = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        # Combine and average
        loss = 0.5 * (loss_positive + loss_negative)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function blending contrastive and cosine similarity losses.

    This loss combines two complementary objectives:
    1. Contrastive Loss: Learns distance-based separation in Euclidean space
    2. Cosine Loss: Learns angle-based similarity in normalized space

    The combination is more robust than either alone:
    - Contrastive loss: Good for absolute separation
    - Cosine loss: Good for relative similarity regardless of magnitude

    Total Loss Formula:
        L_total = α * L_contrastive + (1-α) * L_cosine

        where:
            L_contrastive = contrastive loss with margin
            L_cosine = mean squared error between label and cosine similarity
            α = weighting factor [0, 1]

    Typical values:
        α = 0.7: Emphasizes distance-based separation (used in this system)
        α = 0.5: Equal weighting
        α = 0.3: Emphasizes angle-based similarity

    Attributes:
        contrastive_loss (ContrastiveLoss): Contrastive loss component
        alpha (float): Weight for combining losses

    Args:
        margin (float): Margin for contrastive loss (default: 1.0)
        alpha (float): Combining weight (default: 0.7)
    """

    def __init__(self, margin: float = 1.0, alpha: float = 0.7):
        """Initialize combined loss with specified parameters."""
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(margin)
        self.alpha = alpha

    def forward(self, feat1, feat2, labels):
        """
        Compute combined loss for a batch of feature pairs.

        Args:
            feat1 (torch.Tensor): First set of features [batch_size, feature_dim]
            feat2 (torch.Tensor): Second set of features [batch_size, feature_dim]
            labels (torch.Tensor): Binary labels [batch_size]

        Returns:
            torch.Tensor: Scalar combined loss value

        Computation:
            1. Calculate contrastive loss component
            2. Calculate cosine similarity and MSE with labels
            3. Weighted combination of both losses
        """
        # Contrastive loss component (distance-based)
        contrastive = self.contrastive_loss(feat1, feat2, labels)

        # Cosine similarity loss component (angle-based)
        # Compute cosine similarity [-1, 1]
        cos_sim = F.cosine_similarity(feat1, feat2, dim=1)
        # MSE loss: encourages cos_sim to match labels (0 or 1)
        cosine_loss = torch.mean((labels - cos_sim) ** 2)

        # Weighted combination
        total_loss = self.alpha * contrastive + (1 - self.alpha) * cosine_loss
        return total_loss


# ============================================================================
# TRAINING DATASET
# ============================================================================

class IcebergSequenceComparisonDataset(Dataset):
    """
    Dataset for training on iceberg comparison pairs from a single sequence.

    This dataset creates positive and negative training pairs from ground truth
    annotations of a timelapse sequence. It's designed for learning iceberg
    appearance similarity through pairwise comparison.

    Pair Generation Strategy:
        Positive Pairs:
            - Same iceberg in consecutive frames
            - Label = 1 (should be similar)
            - Created from ground truth tracking data

        Negative Pairs:
            - Different icebergs in consecutive frames
            - Label = 0 (should be dissimilar)
            - Created by random sampling with configurable ratio

    Data Processing Pipeline:
        1. Load ground truth annotations
        2. Extract positive matches (same iceberg across frames)
        3. Generate negative pairs through random sampling
        4. Precompute and cache all iceberg crops in memory
        5. Apply transforms during training

    Attributes:
        sequence_name (str): Name of the sequence
        config (IcebergEmbeddingsConfig): Configuration object
        image_dir (Path): Path to images directory
        transform (callable): Image transformation pipeline
        target_size (int): Target size for crops
        negative_ratio (float): Ratio of negative to positive pairs
        annotation_file (Path): Path to ground truth file
        image_ext (str): Image file extension
        candidates (dict): Candidate icebergs per frame
        matches (list): Ground truth matches
        pairs (list): All training pairs (positive + negative)
        icebergs_by_frame (dict): Icebergs organized by frame
        crop_cache (dict): Precomputed crop cache

    Args:
        sequence_name (str): Name of the sequence
        images_dir (Path): Path to images directory
        gt_file (Path): Path to ground truth file
        config (IcebergEmbeddingsConfig): Configuration object
        transform (callable, optional): Image transforms
        negative_ratio (float, optional): Ratio of negative to positive pairs
        image_ext (str): Image file extension
    """

    def __init__(self, sequence_name: str, images_dir, gt_file, config: IcebergEmbeddingsConfig,
                 transform=None, negative_ratio: Optional[float] = None, image_ext="jpg"):
        """Initialize sequence comparison dataset with pair generation."""
        self.sequence_name = sequence_name
        self.config = config
        self.image_dir = images_dir
        self.transform = transform
        self.target_size = config.img_size
        self.negative_ratio = negative_ratio or config.negative_ratio_train
        self.annotation_file = gt_file
        self.image_ext = image_ext

        # Load annotations and create pairs
        self.candidates = extract_candidates(str(self.annotation_file))
        self.matches = extract_matches(self.candidates)
        self.pairs = self._create_pairs()
        self.icebergs_by_frame = load_icebergs_by_frame(str(self.annotation_file))

        # Precompute crops for efficiency (trades memory for speed)
        self.crop_cache = self._precompute_crops()

    def _create_pairs(self):
        """
        Create positive and negative training pairs from annotations.

        Positive pairs are straightforward: they come directly from ground truth
        tracking data indicating the same iceberg across frames.

        Negative pairs require sampling: for each positive match, we randomly
        select non-matching icebergs from the next frame. The negative_ratio
        parameter controls how many negatives per positive.

        Returns:
            list: List of pair dictionaries with labels
                Each dict contains: frame, next_frame, id, [negative_candidate], label

        The number of negative pairs is typically equal to or greater than
        positive pairs to help the model learn hard negatives effectively.
        """
        positive_pairs = []
        negative_pairs = []

        # Create positive pairs from ground truth matches
        for match in self.matches:
            positive_pair = match.copy()
            positive_pair["label"] = 1  # Label 1 = same iceberg
            positive_pairs.append(positive_pair)

        # Create negative pairs by random sampling
        for match in self.matches:
            # Get all candidates in next frame except the true match
            candidates = [c for c in self.candidates[match["next_frame"]] if c != match["id"]]

            # Sample negative candidates based on ratio
            n_negatives = min(len(candidates), int(self.negative_ratio))
            if n_negatives > 0:
                negative_candidates = random.sample(candidates, n_negatives)

                for neg_candidate in negative_candidates:
                    negative_pair = match.copy()
                    negative_pair["negative_candidate"] = neg_candidate
                    negative_pair["label"] = 0  # Label 0 = different icebergs
                    negative_pairs.append(negative_pair)

        logger.info(
            f"  {self.sequence_name}: Created {len(positive_pairs)} positive pairs "
            f"and {len(negative_pairs)} negative pairs")
        return positive_pairs + negative_pairs

    def _precompute_crops(self):
        """
        Precompute and cache all iceberg crops in memory.

        This method performs all image loading and cropping operations upfront,
        storing the processed crops in a dictionary for instant access during
        training. This dramatically speeds up training at the cost of memory.

        Process:
            1. Iterate through all frames and their icebergs
            2. Load full frame image (once per frame)
            3. Crop all icebergs from that frame
            4. Resize and pad each crop to target size
            5. Store in cache with (frame_name, iceberg_id) key

        Returns:
            dict: Cache mapping (frame_name, iceberg_id) -> PIL.Image
                Crops are stored as PIL Images (transforms applied later)
        """
        crop_cache = {}

        # Count unique crops for progress reporting
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            unique_crops = sum(1 for _ in f)

        logger.info(f"  {self.sequence_name}: Precomputing {unique_crops} unique crops...")

        # Process each frame and its icebergs
        for frame_name, icebergs in self.icebergs_by_frame.items():
            image_path = os.path.join(self.image_dir, frame_name + "." + self.image_ext)

            # Load full image (OpenCV for consistency with bbox coordinates)
            full_image = cv2.imread(image_path)

            # Process each iceberg in this frame
            for iceberg_id, iceberg_data in icebergs.items():
                x, y, w, h = iceberg_data['bbox']

                # Crop iceberg region
                crop = full_image[int(y):int(y + h), int(x):int(x + w)]

                # Convert to PIL Image for transform compatibility
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                # Resize and pad to target size while maintaining aspect ratio
                processed_crop = self._resize_and_pad(img)

                # Store in cache with composite key
                crop_cache[(frame_name, iceberg_id)] = processed_crop

        return crop_cache

    def _resize_and_pad(self, img):
        """
        Resize image while maintaining aspect ratio and pad to square.

        This ensures all crops have uniform dimensions for batch processing
        while preserving the original aspect ratio of each detection.

        Process:
            1. Calculate new dimensions maintaining aspect ratio
            2. Resize to fit within target_size
            3. Create square canvas with black padding
            4. Center-paste resized image

        Args:
            img (PIL.Image): Input image to process

        Returns:
            PIL.Image: Resized and padded square image of size (target_size, target_size)

        Example:
            Input: 60x40 image, target_size=224
            Resize: 224x149 (maintains aspect ratio)
            Pad: 224x224 (adds 37px top/bottom black padding)
        """
        w, h = img.size

        # Calculate new dimensions maintaining aspect ratio
        if w > h:
            # Width-dominant: scale to target width
            new_w, new_h = self.target_size, int(h * self.target_size / w)
        else:
            # Height-dominant: scale to target height
            new_h, new_w = self.target_size, int(w * self.target_size / h)

        # Resize image with high-quality resampling
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create square padded image with black background
        padded_img = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))

        # Center the resized image
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        padded_img.paste(img, (paste_x, paste_y))

        return padded_img

    def __len__(self):
        """Return total number of training pairs."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Get a pair of iceberg images and their similarity label.

        Retrieves precomputed crops for both images in the pair and applies
        any specified transformations. This method is called by the DataLoader
        during training to construct batches.

        Args:
            idx (int): Index of the pair to retrieve

        Returns:
            tuple: (img1, img2, label) where:
                - img1: Tensor of first image [C, H, W]
                - img2: Tensor of second image [C, H, W]
                - label: Float tensor (1.0 for similar, 0.0 for dissimilar)

        Pair Types:
            - Positive (label=1): Same iceberg in consecutive frames
            - Negative (label=0): Different icebergs in consecutive frames
        """
        pair = self.pairs[idx]

        # Format frame names (handle both integer and string formats)
        try:
            img1_name = f"{int(pair['frame']):06d}"
            img2_name = f"{int(pair['next_frame']):06d}"
        except:
            img1_name = pair['frame']
            img2_name = pair['next_frame']

        label = pair["label"]
        img1_id = pair["id"]

        # For positive pairs: same iceberg ID in both frames
        # For negative pairs: different iceberg ID in second frame
        img2_id = img1_id if label == 1 else pair["negative_candidate"]

        # Retrieve precomputed crops from cache (instant access)
        img1 = self.crop_cache[(img1_name, img1_id)]
        img2 = self.crop_cache[(img2_name, img2_id)]

        # Apply transforms (augmentation for training, normalization for validation)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def get_sequence_info(self):
        """
        Get statistics about this sequence dataset.

        Returns:
            dict: Dictionary containing:
                - sequence_name: Name of the sequence
                - num_pairs: Total number of training pairs
                - num_unique_icebergs: Number of unique iceberg crops
        """
        return {
            'sequence_name': self.sequence_name,
            'num_pairs': len(self.pairs),
            'num_unique_icebergs': len(self.crop_cache)
        }


# ============================================================================
# MAIN ORCHESTRATOR CLASS
# ============================================================================

class IcebergEmbeddingsTrainer:
    """
    Main orchestrator for the complete iceberg similarity learning pipeline.

    This class coordinates all aspects of training a Siamese network for learning
    iceberg appearance similarity. It provides a high-level interface that handles:
    - Data loading and preprocessing across multiple sequences
    - Model initialization and configuration
    - Training loop with validation and early stopping
    - Model checkpointing and loading
    - Evaluation and metrics computation
    - Embedding generation for downstream tasks
    - Results visualization

    Architecture Pattern:
        Uses dependency injection through factory functions to allow customization
        of key components (model, transforms) while providing sensible defaults.

    Key Features:
        ✓ Multi-sequence training support (train on diverse data)
        ✓ Early stopping based on validation AUC
        ✓ Automatic model checkpointing (saves best model)
        ✓ Comprehensive training history tracking
        ✓ Built-in evaluation metrics (AUC, Average Precision)
        ✓ Results visualization (loss curves, AUC, similarity distributions)
        ✓ Efficient embedding generation with caching

    Typical Usage:
        >>> config = IcebergEmbeddingsConfig(dataset="hill/train")
        >>> trainer = IcebergEmbeddingsTrainer(config)
        >>> trainer.run_complete_pipeline()

    Attributes:
        config (IcebergEmbeddingsConfig): Training configuration
        dataset (str): Dataset name/path
        model_path (str): Path for saving/loading model weights
        model_factory (callable): Factory function for creating model
        transform_factory (callable): Factory function for creating transforms
        model (SiameseNetwork): The Siamese network model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (optim.Optimizer): Adam optimizer
        scheduler (optim.Scheduler): Learning rate scheduler
        criterion (nn.Module): Combined loss function
        device (torch.device): Computing device (GPU/CPU)
        history (dict): Training history and metrics

    Args:
        config (IcebergEmbeddingsConfig): Configuration object
        model_factory (callable, optional): Custom model factory
        transform_factory (callable, optional): Custom transform factory
    """

    def __init__(self,
                 config: IcebergEmbeddingsConfig,
                 model_factory: Optional[Callable] = None,
                 transform_factory: Optional[Callable] = None):
        """
        Initialize the trainer with configuration and optional factories.

        Args:
            config (IcebergEmbeddingsConfig): Complete configuration object
            model_factory (callable, optional): Custom factory for model creation
                Signature: () -> SiameseNetwork
            transform_factory (callable, optional): Custom factory for transforms
                Signature: () -> (train_transform, val_transform)
        """
        self.config = config
        self.dataset = config.dataset
        self.model_path = os.path.join(PROJECT_ROOT, "models", "iceberg_embedding_model.pth")

        # Set up factory functions (use defaults if not provided)
        self.model_factory = model_factory or self._default_model_factory
        self.transform_factory = transform_factory or self._default_transform_factory

        # Create models directory if it doesn't exist
        os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

        # Initialize components (will be set up later)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = config.device if config.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training history tracking
        self.history = {
            'train_losses': [],  # Training loss per epoch
            'val_losses': [],  # Validation loss per epoch
            'val_aucs': [],  # Validation AUC per epoch
            'best_auc': 0.0,  # Best AUC achieved
            'early_stopped': False,  # Whether early stopping was triggered
            'final_epoch': 0  # Final epoch reached
        }

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete training and evaluation pipeline.

        This is the main entry point that orchestrates the entire process
        from start to finish. It's designed to be a one-liner for running
        everything with sane defaults.

        Pipeline Stages:
            1. Data Setup: Load and preprocess multi-sequence datasets
            2. Model Setup: Initialize model, optimizer, scheduler, loss
            3. Training: Train with early stopping and checkpointing
            4. Evaluation: Compute final metrics on validation set
            5. Embedding Generation: Generate features for ground truth
            6. Visualization: Plot training curves and results
            7. Summary: Report final performance

        Returns:
            Dict[str, Any]: Complete training history and evaluation results
                Contains: train_losses, val_losses, val_aucs, best_auc,
                         early_stopped, final_epoch, and evaluation metrics
        """
        logger.info("🚀 Starting Iceberg Similarity Training Pipeline")
        logger.info("=" * 60)

        try:
            # Phase 1: Setup
            self._setup_data()
            self._setup_model()

            # Phase 2: Training
            self._train()

            # Phase 3: Evaluation
            self._evaluate()

            logger.info(f"\n🎉 Training completed!")
            logger.info(f"Final AUC: {self.history['best_auc']:.4f}")
            logger.info(f"Early stopped: {self.history['early_stopped']}")
            logger.info(f"Final epoch: {self.history['final_epoch']}")

            # Phase 4: Embedding Generation
            self.generate_iceberg_embeddings(annotation_source="ground_truth")

            # Phase 5: Visualization
            self._plot_results()

            logger.info("\n✅ Pipeline completed successfully!")

            return self.history

        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {str(e)}")
            raise

    def generate_iceberg_embeddings(self, annotation_source: str):
        """
        Generate and cache feature embeddings for all iceberg detections.

        This method loads the trained Siamese network and uses it to extract
        feature vectors for all icebergs across all sequences. The embeddings
        are saved to disk for efficient use in downstream tracking.

        Process Flow:
            1. Load trained model weights
            2. Set up validation transforms (no augmentation)
            3. For each sequence:
               a. Load annotations (ground_truth, detections, or tracking)
               b. Create dataset and dataloader
               c. Extract features in batches
               d. Save embeddings to disk

        Args:
            annotation_source (str): Source of annotations. Must be one of:
                - 'ground_truth': Use ground truth annotations (gt.txt)
                - 'detections': Use detection results (det.txt)
                - 'tracking': Use tracking results (track.txt)

        Raises:
            ValueError: If annotation_source is not one of the valid options

        Output Format:
            Saved embeddings are dictionaries: {unique_id: feature_vector}
            where unique_id = "{frame_name}_{iceberg_id}"
        """
        total_start_time = time.time()
        logger.info("\n=== EMBEDDING GENERATION PHASE ===")
        logger.info("Generating iceberg embeddings...")

        # Step 1: Load the trained embedding model
        model = SiameseNetwork(self.config).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        # Step 2: Set up validation transforms (no augmentation for inference)
        val_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

        # Set model to evaluation mode (disables dropout, batch norm training, etc.)
        model.eval()

        # Validate annotation source
        if annotation_source not in ["ground_truth", "detections", "tracking"]:
            raise ValueError(
                f"Invalid annotation_source '{annotation_source}'. "
                f"Must be 'ground_truth', 'detections', or 'tracking'"
            )

        # Get all sequences to process
        sequences = get_sequences(self.dataset)

        # Process each sequence independently
        for sequence_name, paths in sequences.items():
            logger.info(f"\nProcessing sequence: {sequence_name}")

            # Determine paths based on annotation source
            if annotation_source == "ground_truth":
                embeddings_path = paths["gt_embeddings"]
                if not paths["ground_truth"].exists():
                    logger.info(f"⚠ Warning: No gt.txt found at {paths['ground_truth']}, skipping...")
                    continue
            elif annotation_source == "detections":
                embeddings_path = paths["det_embeddings"]
                if not paths["detections"].exists():
                    logger.info(f"⚠ Warning: No det.txt found at {paths['detections']}, skipping...")
                    continue
            elif annotation_source == "tracking":
                embeddings_path = paths["track_embeddings"]
                if not paths["tracking"].exists():
                    logger.info(f"⚠ Warning: No track.txt found at {paths['tracking']}, skipping...")
                    continue

            image_ext = get_image_ext(paths["images"])

            # Step 3: Parse all detections from annotation file
            all_detections = parse_annotations(paths[annotation_source])

            # Step 4: Sort detections by frame name
            # This groups all crops from the same image together, maximizing cache hits
            # for the LRU cache in the dataset and dramatically improving performance
            all_detections.sort(key=lambda x: x['frame'])

            # Step 5: Create dataset and dataloader for batch processing
            dataset = IcebergEmbeddingsDataset(detections=all_detections,
                                               full_frame_dir=paths["images"],
                                               transform=val_transform,
                                               target_size=224,
                                               image_ext=image_ext)

            # DataLoader with multiple workers for parallel loading
            # The LRU cache works per-worker, so with 4 workers an image might be
            # loaded up to 4 times, but this is still vastly better than thousands
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                                    num_workers=4, pin_memory=True)

            # Step 6: Extract features for all detections
            features_dict = {}
            logger.info(f"Extracting features from {len(all_detections)} icebergs...")

            with torch.no_grad():  # Disable gradient computation for inference
                # Process images in batches for efficiency
                for image_batch, name_batch in dataloader:
                    image_batch = image_batch.to(self.device)

                    # Extract features using the backbone of the Siamese network
                    feature_batch = model.backbone(image_batch)

                    # Store features for each iceberg detection
                    for i, img_name in enumerate(name_batch):
                        features_dict[img_name] = feature_batch[i].cpu()

            logger.info(f"Feature extraction complete for {sequence_name}.")

            # Step 7: Save computed features for future use
            torch.save(features_dict, embeddings_path)
            logger.info(f"Embeddings saved to {embeddings_path}")

        elapsed_time = timedelta(seconds=int(time.time() - total_start_time))
        logger.info(f"\nEmbedding generation for all sequences completed in {elapsed_time}")

    def _default_model_factory(self) -> SiameseNetwork:
        """
        Default factory for creating the Siamese network model.

        Creates a standard Siamese network with Vision Transformer backbone
        using the configuration parameters.

        Returns:
            SiameseNetwork: Initialized Siamese network ready for training
        """
        return SiameseNetwork(self.config)

    def _default_transform_factory(self) -> tuple:
        """
        Default factory for creating image transformation pipelines.

        Creates separate transform pipelines for training and validation:

        Training Transforms:
            - Color jitter: Varies brightness, contrast, saturation, hue
            - Random horizontal flip: 50% probability
            - Random vertical flip: 50% probability
            - Random rotation: ±15 degrees
            - Tensor conversion and ImageNet normalization

        Validation Transforms:
            - Tensor conversion and ImageNet normalization only
            - No augmentation to ensure consistent evaluation

        Returns:
            tuple: (train_transform, val_transform)
                Each is a torchvision.transforms.Compose object
        """
        # Training transforms with data augmentation
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Validation transforms without augmentation
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_transform, val_transform

    def _setup_model(self):
        """
        Initialize model, optimizer, scheduler, and loss function.

        Sets up all components needed for training:
        - Model: Siamese network with ViT backbone
        - Optimizer: Adam with weight decay
        - Scheduler: Step learning rate decay
        - Loss: Combined contrastive + cosine loss

        All components are configured using parameters from self.config.
        """
        logger.info("\n=== MODEL SETUP PHASE ===")
        logger.info("Setting up model and training components...")

        # Initialize model using factory function
        self.model = self.model_factory().to(self.device)

        # Setup Adam optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Setup learning rate scheduler (decay by 0.1 every 10 epochs)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Setup combined loss criterion
        self.criterion = CombinedLoss(
            margin=self.config.loss_margin,
            alpha=self.config.loss_alpha
        ).to(self.device)

        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"Model device: {self.device}")

    def _setup_data(self):
        """
        Setup data loaders for training and validation across all sequences.

        Creates multi-sequence datasets with appropriate transforms and negative
        sampling ratios, then wraps them in PyTorch DataLoaders for efficient
        batch processing.

        Process:
            1. Get train and validation transforms from factory
            2. Create training dataset (all sequences combined)
            3. Create validation dataset (all sequences combined)
            4. Wrap in DataLoaders with batching and multi-worker loading
        """
        logger.info("\n=== DATA SETUP PHASE ===")
        logger.info("Setting up data loaders...")

        # Get transforms from factory
        train_transform, val_transform = self.transform_factory()

        # Create multi-sequence training dataset
        logger.info("\n--- Creating Training Dataset ---")
        train_multi_seq_dataset = self._get_multi_seq_dataset(
            train_transform,
            self.config.negative_ratio_train
        )

        # Create multi-sequence validation dataset
        logger.info("\n--- Creating Validation Dataset ---")
        val_multi_seq_dataset = self._get_multi_seq_dataset(
            val_transform,
            self.config.negative_ratio_val
        )

        # Create data loaders with batching and multi-worker loading
        self.train_loader = DataLoader(
            train_multi_seq_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=self.config.num_workers,
            pin_memory=True  # Faster GPU transfer
        )

        self.val_loader = DataLoader(
            val_multi_seq_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        logger.info(f"\nData loaders ready!")
        logger.info(f"Training pairs: {len(train_multi_seq_dataset)}")
        logger.info(f"Validation pairs: {len(val_multi_seq_dataset)}")
        logger.info(f"Training batches per epoch: {len(self.train_loader)}")
        logger.info(f"Validation batches per epoch: {len(self.val_loader)}")

    def _get_multi_seq_dataset(self, transform, negative_ratio):
        """
        Create a combined dataset from all sequences.

        Args:
            transform (callable): Transform pipeline to apply
            negative_ratio (float): Ratio of negative to positive pairs

        Returns:
            ConcatDataset: Combined dataset from all sequences
        """
        # Load all sequences using helper function
        sequences = get_sequences(self.dataset)

        # Create individual sequence datasets
        sequence_datasets = []
        sequence_names = []

        logger.info(f"\nCreating comparison datasets for {len(sequences)} sequences...")

        for sequence_name, paths in sequences.items():
            # Skip if ground truth doesn't exist
            if not paths["ground_truth"].exists():
                logger.info(f"⚠ Warning: No gt.txt found at {paths['ground_truth']}, skipping...")
                continue

            # Get image extension
            image_ext = get_image_ext(paths['images'])

            # Create sequence dataset
            seq_dataset = IcebergSequenceComparisonDataset(
                sequence_name=sequence_name,
                images_dir=paths['images'],
                gt_file=paths['ground_truth'],
                config=self.config,
                transform=transform,
                negative_ratio=negative_ratio,
                image_ext=image_ext
            )
            sequence_datasets.append(seq_dataset)
            sequence_names.append(sequence_name)

        # Combine all sequences into one dataset
        combined_dataset = ConcatDataset(sequence_datasets)

        logger.info(f"\nTotal pairs: {len(combined_dataset)}")

        # Log individual sequence info
        for seq_dataset in sequence_datasets:
            info = seq_dataset.get_sequence_info()
            logger.info(
                f"  - {info['sequence_name']}: {info['num_pairs']} pairs, "
                f"{info['num_unique_icebergs']} unique icebergs"
            )

        return combined_dataset

    def _train(self):
        """
        Main training loop with validation and early stopping.

        Orchestrates the complete training process across all epochs:
        - Runs training and validation for each epoch
        - Tracks best model based on validation AUC
        - Implements early stopping to prevent overfitting
        - Updates learning rate schedule
        - Saves model checkpoints
        - Logs progress and time estimates
        """
        logger.info("\n=== TRAINING PHASE ===")
        logger.info("Starting training...")
        logger.info(f"Training for up to {self.config.num_epochs} epochs")
        logger.info(f"Early stopping patience: {self.config.patience} epochs")
        logger.info(f"Minimum improvement: {self.config.min_delta:.4f} AUC")

        best_auc = 0.0
        patience_counter = 0
        total_start_time = time.time()

        # Main epoch loop
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss = self._train_epoch()
            self.history['train_losses'].append(train_loss)

            # Validation phase
            val_loss, val_auc = self._validate_epoch()
            self.history['val_losses'].append(val_loss)
            self.history['val_aucs'].append(val_auc)

            # Calculate time estimates
            elapsed_time = time.time() - total_start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            estimated_remaining = avg_time_per_epoch * (self.config.num_epochs - epoch - 1)

            # Log progress
            logger.info(f'Epoch {epoch + 1}/{self.config.num_epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}')
            logger.info(f'  Val Loss: {val_loss:.4f}')
            logger.info(f'  Val AUC: {val_auc:.4f}')
            logger.info(f'  Best AUC: {best_auc:.4f}')
            logger.info(f'  Time: {timedelta(seconds=int(elapsed_time))}<'
                        f'{timedelta(seconds=int(estimated_remaining))}, '
                        f'{timedelta(seconds=int(avg_time_per_epoch))}/Epoch')

            # Check for improvement and save best model
            if val_auc > best_auc + self.config.min_delta:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                logger.info(f"  ✅ New best model saved! AUC: {best_auc:.4f}")
            else:
                patience_counter += 1
                # Early stopping check
                if patience_counter >= self.config.patience:
                    logger.info(f"  🛑 Early stopping triggered after {self.config.patience} "
                                f"epochs without improvement")
                    self.history['early_stopped'] = True
                    break

            # Update learning rate schedule
            self.scheduler.step()

            logger.info('-' * 60)

        # Update final history
        self.history['best_auc'] = best_auc
        self.history['final_epoch'] = epoch + 1

        logger.info(f"\n{'=' * 60}")
        logger.info(f"TRAINING COMPLETED")
        logger.info(f"{'=' * 60}")
        logger.info(f"Best AUC: {best_auc:.4f}")
        logger.info(f"Total training time: {timedelta(seconds=int(time.time() - total_start_time))}")
        logger.info(f"Model saved to: {self.model_path}")

    def _train_epoch(self) -> float:
        """
        Train for one epoch over the training dataset.

        Performs a single complete pass through the training data:
        1. Sets model to training mode (enables dropout, batch norm updates)
        2. Iterates through training batches
        3. Computes forward pass and loss
        4. Computes gradients via backpropagation
        5. Updates model parameters via optimizer
        6. Returns average training loss

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()  # Set to training mode
        total_loss = 0.0

        for img1, img2, labels in self.train_loader:
            # Move data to device (GPU/CPU)
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

            # Zero gradients from previous iteration
            self.optimizer.zero_grad()

            # Forward pass through Siamese network
            feat1, feat2 = self.model(img1, img2)

            # Compute combined loss
            loss = self.criterion(feat1, feat2, labels)

            # Backward pass and optimization
            loss.backward()  # Compute gradients
            self.optimizer.step()  # Update parameters

            # Accumulate loss
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self) -> tuple:
        """
        Validate for one epoch over the validation dataset.

        Performs validation to assess model generalization:
        1. Sets model to evaluation mode (disables dropout, etc.)
        2. Disables gradient computation for efficiency
        3. Computes predictions and losses
        4. Calculates ROC-AUC metric
        5. Returns validation loss and AUC

        Returns:
            tuple: (average_validation_loss, validation_auc)

        AUC Interpretation:
            1.0 = Perfect discrimination (all positives ranked above negatives)
            0.5 = Random chance (no discrimination ability)
            <0.5 = Worse than random (inverted predictions)
        """
        self.model.eval()  # Set to evaluation mode
        total_loss = 0.0
        all_similarities = []
        all_labels = []

        with torch.no_grad():  # Disable gradient computation for efficiency
            for img1, img2, labels in self.val_loader:
                # Move data to device
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                # Forward pass
                feat1, feat2 = self.model(img1, img2)
                loss = self.criterion(feat1, feat2, labels)

                total_loss += loss.item()

                # Compute similarities for AUC calculation
                similarities = self.model.compute_similarity(feat1, feat2)
                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate average loss and AUC
        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(all_labels, all_similarities)

        return avg_loss, auc

    def _evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the trained model on validation set with comprehensive metrics.

        Loads the best saved model and computes detailed evaluation metrics
        including ROC-AUC, Average Precision, and similarity distributions.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - auc: ROC-AUC score
                - average_precision: Average Precision score
                - similarities: List of all similarity scores
                - labels: List of all ground truth labels

        Metrics:
            - ROC-AUC: Area under ROC curve (threshold-independent)
            - Average Precision: Area under Precision-Recall curve
        """
        logger.info("\n=== EVALUATION PHASE ===")
        logger.info("Evaluating model...")

        # Load best model weights
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        all_similarities = []
        all_labels = []

        with torch.no_grad():
            for img1, img2, labels in self.val_loader:
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                # Get embeddings and compute similarities
                feat1, feat2 = self.model(img1, img2)
                similarities = self.model.compute_similarity(feat1, feat2)

                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute evaluation metrics
        auc = roc_auc_score(all_labels, all_similarities)
        ap = average_precision_score(all_labels, all_similarities)

        results = {
            'auc': auc,
            'average_precision': ap,
            'similarities': all_similarities,
            'labels': all_labels
        }

        # Add results to history for plotting
        self.history.update(results)

        logger.info(f"Final Evaluation Results:")
        logger.info(f"  ROC-AUC: {auc:.4f}")
        logger.info(f"  Average Precision: {ap:.4f}")

        return results

    def _plot_results(self):
        """
        Plot comprehensive training results and evaluation metrics.

        Creates a three-panel visualization showing:
        1. Training and validation loss curves over epochs
        2. Validation AUC progression over epochs
        3. Distribution of similarity scores for positive vs negative pairs

        This visualization helps understand:
        - Training dynamics and convergence
        - Overfitting (gap between train and validation loss)
        - Model discrimination ability (separation of positive/negative distributions)

        The plot is displayed interactively using matplotlib.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Training and validation losses
        axes[0].plot(self.history['train_losses'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_losses'], label='Val Loss', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Validation AUC over epochs
        axes[1].plot(self.history['val_aucs'], label='Val AUC', color='green', linewidth=2)
        axes[1].set_title('Validation AUC', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Similarity score distributions
        if 'similarities' in self.history and 'labels' in self.history:
            # Separate similarities by label
            pos_sim = [s for s, l in zip(self.history['similarities'], self.history['labels']) if l == 1]
            neg_sim = [s for s, l in zip(self.history['similarities'], self.history['labels']) if l == 0]

            # Plot histograms with transparency for overlap visibility
            axes[2].hist(pos_sim, alpha=0.7, label='Positive pairs (same iceberg)',
                         bins=30, color='green', edgecolor='black')
            axes[2].hist(neg_sim, alpha=0.7, label='Negative pairs (different icebergs)',
                         bins=30, color='red', edgecolor='black')
            axes[2].set_title('Similarity Distribution', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Cosine Similarity')
            axes[2].set_ylabel('Frequency')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()