import cv2
from datetime import timedelta
import functools
import math
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
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Callable, Optional, Any

from utils.helpers import DATA_DIR, extract_candidates, extract_matches, load_icebergs_by_frame, parse_annotations


"""
Iceberg Similarity Learning Pipeline

This module implements a Siamese Neural Network training pipeline for 
learning and generating embeddings that capture iceberg appearance 
similarity. The pipeline uses Vision Transformers as the backbone 
architecture and trains on pairs of iceberg crops to learn 
discriminative features for iceberg tracking and identification tasks.

Key Components:
- Vision Transformer backbone for feature extraction
- Siamese network architecture for pairwise learning
- Combined loss function (contrastive + cosine similarity)
- Data pipeline with in-memory caching for efficiency
- Comprehensive training and evaluation framework
"""


@dataclass
class IcebergEmbeddingsConfig:
    """
    Configuration class for all training parameters.

    This dataclass centralizes all hyperparameters and settings for the training pipeline,
    making it easy to experiment with different configurations and maintain reproducibility.

    Attributes:
        dataset (str): Name of the dataset directory
        image_format (str): Image file format (e.g., 'JPG', 'PNG')
        negative_ratio_train (float): Ratio of negative to positive pairs for training
        negative_ratio_val (float): Ratio of negative to positive pairs for validation
        num_workers (int): Number of worker processes for data loading

        # Model architecture parameters
        img_size (int): Input image size (assumed square)
        patch_size (int): Size of patches for Vision Transformer
        in_channels (int): Number of input channels (3 for RGB)
        embed_dim (int): Embedding dimension for transformer
        depth (int): Number of transformer layers
        n_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim
        dropout (float): Dropout rate
        feature_dim (int): Final feature dimension for embeddings

        # Training parameters
        batch_size (int): Training batch size
        val_batch_size (int): Validation batch size
        learning_rate (float): Initial learning rate
        weight_decay (float): L2 regularization weight
        num_epochs (int): Maximum number of training epochs
        patience (int): Early stopping patience
        min_delta (float): Minimum improvement for early stopping

        # Loss function parameters
        loss_margin (float): Margin for contrastive loss
        loss_alpha (float): Weight for combining contrastive and cosine losses

        # Hardware configuration
        device (str): Device to use for training ('cuda' or 'cpu')
    """
    # Data configuration
    dataset: str
    image_format: str = "JPG"
    negative_ratio_train: float = 2.0
    negative_ratio_val: float = 1.0
    num_workers: int = 4

    # Model configuration
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
    patience: int = 5
    min_delta: float = 0.002

    # Loss configuration
    loss_margin: float = 1.0
    loss_alpha: float = 0.7

    # Device configuration
    device: str = 'cuda'


class IcebergEmbeddingsDataset(Dataset):
    """
    PyTorch Dataset for loading iceberg crops from full frame images.

    This dataset loads full images and crops icebergs on the fly, with an in-memory
    cache to avoid re-loading the same full frames repeatedly. This approach is
    memory efficient while maintaining reasonable performance.

    The dataset handles:
    - Loading full frame images with LRU caching
    - Cropping icebergs based on bounding box annotations
    - Resizing and padding crops to maintain aspect ratio
    - Applying image transformations

    Args:
        detections (list[dict]): List of iceberg detection dictionaries
        full_frame_dir (str): Directory containing full frame images
        transform (callable, optional): Image transformations to apply
        target_size (int): Target size for cropped images (default: 224)
        img_extension (str): Image file extension (default: '.JPG')
    """

    def __init__(self, detections: list[dict], full_frame_dir: str, transform=None, target_size: int = 224,
                 img_extension: str = '.JPG'):
        self.detections = detections
        self.full_frame_dir = full_frame_dir
        self.transform = transform
        self.target_size = target_size
        self.img_extension = img_extension

        # Use default normalization if no transform provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    @functools.lru_cache(maxsize=32)
    def _load_image(self, img_path: str) -> Image.Image:
        """
        Load an image from disk with LRU caching.

        This method caches loaded images to avoid repeated disk I/O when multiple
        icebergs are cropped from the same frame. The cache size can be adjusted
        based on available memory and typical usage patterns.

        Args:
            img_path (str): Path to the image file

        Returns:
            Image.Image: Loaded PIL image in RGB format
        """
        return Image.open(img_path).convert('RGB')

    def __len__(self) -> int:
        """Return the number of detections in the dataset."""
        return len(self.detections)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        Get an iceberg crop and its identifier.

        This method:
        1. Loads the full frame image (with caching)
        2. Crops the iceberg based on bounding box coordinates
        3. Resizes and pads the crop to maintain aspect ratio
        4. Applies transformations
        5. Returns the processed tensor and unique identifier

        Args:
            idx (int): Index of the detection to retrieve

        Returns:
            tuple[torch.Tensor, str]: Processed image tensor and unique iceberg name
        """
        detection = self.detections[idx]
        frame_name = detection['frame']
        iceberg_id = detection['id']

        # Construct paths and identifiers
        img_path = os.path.join(self.full_frame_dir, f"{frame_name}{self.img_extension}")
        unique_iceberg_name = f"{frame_name}_{iceberg_id}"

        try:
            # Load full image (cached)
            full_img = self._load_image(img_path)

            # Extract crop coordinates and crop the iceberg
            left, top = detection['bb_left'], detection['bb_top']
            right, bottom = left + detection['bb_width'], top + detection['bb_height']
            iceberg_crop = full_img.crop((left, top, right, bottom))

            # Resize while maintaining aspect ratio
            w, h = iceberg_crop.size
            if w > h:
                new_w, new_h = self.target_size, int(h * self.target_size / w)
            else:
                new_h, new_w = self.target_size, int(w * self.target_size / h)

            resized_crop = iceberg_crop.resize((new_w, new_h), Image.LANCZOS)

            # Pad to square image with black padding
            padded_img = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
            pad_w, pad_h = (self.target_size - new_w) // 2, (self.target_size - new_h) // 2
            padded_img.paste(resized_crop, (pad_w, pad_h))

            # Apply transformations
            tensor_img = self.transform(padded_img)

        except Exception as e:
            # Handle errors gracefully by returning a zero tensor
            print(f"Error processing {unique_iceberg_name} from {img_path}: {e}")
            tensor_img = torch.zeros(3, self.target_size, self.target_size)

        return tensor_img, unique_iceberg_name


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for Vision Transformer.

    Converts input images into sequences of patch embeddings by:
    1. Splitting the image into non-overlapping patches
    2. Projecting each patch to the embedding dimension using a convolution
    3. Flattening and transposing to get the sequence format

    This is the first step in Vision Transformer processing, converting
    2D images into 1D sequences that can be processed by transformer layers.

    Args:
        img_size (int): Input image size (assumed square)
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension for each patch
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # Total number of patches

        # Convolution layer acts as linear projection for each patch
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass of patch embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Patch embeddings of shape (batch_size, n_patches, embed_dim)
        """
        # Apply convolution to create patch embeddings
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)

        # Flatten spatial dimensions to create sequence
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)

        # Transpose to get sequence format
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Implements the core attention mechanism of transformers, allowing the model
    to focus on different parts of the input sequence simultaneously through
    multiple attention heads. Each head learns different types of relationships.

    The attention mechanism computes:
    Attention(Q,K,V) = softmax(QK^T/√d_k)V

    Where Q, K, V are queries, keys, and values derived from the input.

    Args:
        embed_dim (int): Embedding dimension
        n_heads (int): Number of attention heads
        dropout (float): Dropout rate for attention weights
    """

    def __init__(self, embed_dim: int = 384, n_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by n_heads"

        # Linear layers for computing Q, K, V simultaneously
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Generate Q, K, V and reshape for multi-head attention
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)  # Scaling factor to prevent vanishing gradients
        attn = (q @ k.transpose(-2, -1)) / scale  # Compute attention scores
        attn = F.softmax(attn, dim=-1)  # Normalize attention weights
        attn = self.dropout(attn)  # Apply dropout to attention weights

        # Apply attention to values and reshape output
        out = attn @ v  # Weighted combination of values
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.out_proj(out)  # Final linear projection

        return out


class TransformerBlock(nn.Module):
    """
    Transformer encoder block.

    A standard transformer block consisting of:
    1. Multi-head self-attention with residual connection and layer norm
    2. Feed-forward MLP with residual connection and layer norm

    This follows the "Pre-LN" transformer architecture where layer normalization
    is applied before the attention and MLP layers rather than after.

    Args:
        embed_dim (int): Embedding dimension
        n_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension
        dropout (float): Dropout rate
    """

    def __init__(self, embed_dim: int = 384, n_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP with expansion and contraction
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),  # GELU activation works better than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Multi-head attention with residual connection (Pre-LN)
        x = x + self.attn(self.norm1(x))

        # MLP with residual connection (Pre-LN)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for feature extraction.

    Implements a Vision Transformer (ViT) that:
    1. Converts images to patch embeddings
    2. Adds positional embeddings and a class token
    3. Processes through transformer layers
    4. Extracts features from the class token
    5. Projects to final feature dimension with L2 normalization

    The class token serves as a global representation that aggregates
    information from all patches through self-attention.

    Args:
        img_size (int): Input image size
        patch_size (int): Size of patches
        in_channels (int): Number of input channels
        embed_dim (int): Transformer embedding dimension
        depth (int): Number of transformer layers
        n_heads (int): Number of attention heads
        mlp_ratio (float): MLP expansion ratio
        dropout (float): Dropout rate
        feature_dim (int): Final feature dimension
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 384, depth: int = 12, n_heads: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, feature_dim: int = 256):
        super().__init__()

        # Patch embedding layer
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Learnable class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Feature projection head for final embeddings
        self.feature_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, feature_dim),
            nn.BatchNorm1d(feature_dim)  # Batch normalization for stable training
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize learnable parameters with appropriate distributions."""
        # Initialize positional embeddings and class token with truncated normal
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Forward pass of Vision Transformer.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: L2-normalized features of shape (batch_size, feature_dim)
        """
        batch_size = x.shape[0]

        # Convert image to patch embeddings
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)

        # Add class token to the beginning of the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract class token for global representation
        cls_token_final = x[:, 0]  # First token is the class token
        features = self.feature_head(cls_token_final)

        # L2 normalize features for cosine similarity computation
        features = F.normalize(features, p=2, dim=1)

        return features


class SiameseNetwork(nn.Module):
    """
    Siamese Network using Vision Transformer backbone.

    A Siamese network consists of two identical subnetworks (sharing weights)
    that process two inputs separately and produce embeddings that can be
    compared. This architecture is ideal for learning similarity functions.

    The network:
    1. Processes two images through the same ViT backbone
    2. Produces normalized feature embeddings for each
    3. Can compute similarity between the embeddings

    Args:
        config (IcebergEmbeddingsConfig): Configuration containing model parameters
    """

    def __init__(self, config: IcebergEmbeddingsConfig):
        super().__init__()

        # Extract ViT configuration from main config
        vit_config = {
            'img_size': config.img_size,
            'patch_size': config.patch_size,
            'in_channels': config.in_channels,
            'embed_dim': config.embed_dim,
            'depth': config.depth,
            'n_heads': config.n_heads,
            'mlp_ratio': config.mlp_ratio,
            'dropout': config.dropout,
            'feature_dim': config.feature_dim
        }

        # Shared backbone for both inputs
        self.backbone = VisionTransformer(**vit_config)

    def forward(self, img1, img2):
        """
        Forward pass through Siamese network.

        Args:
            img1 (torch.Tensor): First image batch
            img2 (torch.Tensor): Second image batch

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Feature embeddings for both images
        """
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        return feat1, feat2

    def compute_similarity(self, feat1, feat2):
        """
        Compute cosine similarity between feature embeddings.

        Args:
            feat1 (torch.Tensor): First set of features
            feat2 (torch.Tensor): Second set of features

        Returns:
            torch.Tensor: Cosine similarity scores
        """
        return F.cosine_similarity(feat1, feat2, dim=1)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for siamese network.

    Contrastive loss encourages similar pairs to have small distances and
    dissimilar pairs to have large distances (at least a margin). The loss is:

    L = (1/2) * [y * d^2 + (1-y) * max(0, margin - d)^2]

    Where:
    - y = 1 for similar pairs, 0 for dissimilar pairs
    - d = euclidean distance between features
    - margin = minimum distance for dissimilar pairs

    Args:
        margin (float): Margin for negative pairs
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, feat1, feat2, labels):
        """
        Compute contrastive loss.

        Args:
            feat1 (torch.Tensor): First set of features
            feat2 (torch.Tensor): Second set of features
            labels (torch.Tensor): Binary labels (1 for similar, 0 for dissimilar)

        Returns:
            torch.Tensor: Contrastive loss value
        """
        # Compute euclidean distance between feature pairs
        distances = F.pairwise_distance(feat1, feat2)

        # Loss for positive pairs (should be close)
        loss_positive = labels * torch.pow(distances, 2)

        # Loss for negative pairs (should be at least margin apart)
        loss_negative = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        # Combine losses
        loss = 0.5 * (loss_positive + loss_negative)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function using both contrastive and cosine similarity losses.

    This loss combines:
    1. Contrastive loss in euclidean space
    2. Cosine similarity loss in normalized space

    The combination helps the model learn both distance-based and angle-based
    similarities, which can be more robust for similarity learning.

    Total Loss = α * ContrastiveLoss + (1-α) * CosineLoss

    Args:
        margin (float): Margin for contrastive loss
        alpha (float): Weight for combining losses (0-1)
    """

    def __init__(self, margin: float = 1.0, alpha: float = 0.7):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(margin)
        self.alpha = alpha

    def forward(self, feat1, feat2, labels):
        """
        Compute combined loss.

        Args:
            feat1 (torch.Tensor): First set of features
            feat2 (torch.Tensor): Second set of features
            labels (torch.Tensor): Binary labels

        Returns:
            torch.Tensor: Combined loss value
        """
        # Contrastive loss component
        contrastive = self.contrastive_loss(feat1, feat2, labels)

        # Cosine similarity loss component
        cos_sim = F.cosine_similarity(feat1, feat2, dim=1)
        cosine_loss = torch.mean((labels - cos_sim) ** 2)

        # Weighted combination
        total_loss = self.alpha * contrastive + (1 - self.alpha) * cosine_loss
        return total_loss


# ============================================================================
# DATASET
# ============================================================================

class IcebergComparisonDataset(Dataset):
    """
    Dataset for training Siamese networks on iceberg comparison tasks.

    This dataset creates positive and negative pairs from iceberg annotations:
    - Positive pairs: Same iceberg in consecutive frames
    - Negative pairs: Different icebergs in consecutive frames

    The dataset includes:
    - Pair generation with configurable negative sampling ratio
    - Precomputed crop caching for efficiency
    - Proper train/validation splitting considerations

    Args:
        config (IcebergEmbeddingsConfig): Configuration object
        transform (callable, optional): Image transformations
        negative_ratio (float, optional): Ratio of negative to positive pairs
    """

    def __init__(self, config: IcebergEmbeddingsConfig, transform=None, negative_ratio: Optional[float] = None):
        self.config = config
        self.image_dir = os.path.join(DATA_DIR, config.dataset, "images", "raw")
        self.transform = transform
        self.target_size = config.img_size
        self.negative_ratio = negative_ratio or config.negative_ratio_train
        self.annotation_file = os.path.join(DATA_DIR, config.dataset, "annotations", "gt.txt")
        self.image_format = f".{config.image_format}"

        # Load annotations and create pairs
        self.candidates = extract_candidates(self.annotation_file)
        self.matches = extract_matches(self.candidates)
        self.pairs = self._create_pairs()
        self.icebergs_by_frame = load_icebergs_by_frame(self.annotation_file)

        # Precompute crops for efficiency
        self.crop_cache = self._precompute_crops()

    def _create_pairs(self):
        """
        Create positive and negative training pairs.

        Positive pairs are created from ground truth matches between consecutive
        frames. Negative pairs are created by pairing each iceberg with random
        non-matching icebergs from the next frame.

        Returns:
            list: List of pair dictionaries with labels
        """
        positive_pairs = []
        negative_pairs = []

        # Create positive pairs from ground truth matches
        for match in self.matches:
            positive_pair = match.copy()
            positive_pair["label"] = 1  # Label 1 for matching pairs
            positive_pairs.append(positive_pair)

        # Create negative pairs by random sampling
        for match in self.matches:
            # Get all candidates except the true match
            candidates = [c for c in self.candidates[match["next_frame"]] if c != match["id"]]

            # Sample negative candidates based on ratio
            n_negatives = min(len(candidates), int(self.negative_ratio))
            if n_negatives > 0:
                negative_candidates = random.sample(candidates, n_negatives)

                for neg_candidate in negative_candidates:
                    negative_pair = match.copy()
                    negative_pair["negative_candidate"] = neg_candidate
                    negative_pair["label"] = 0  # Label 0 for non-matching pairs
                    negative_pairs.append(negative_pair)

        print(f"Created {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        return positive_pairs + negative_pairs

    def _precompute_crops(self):
        """
        Precompute and cache all required iceberg crops.

        This method loads all full images, crops all icebergs, and stores
        the processed crops in memory. This trades memory for speed during
        training, avoiding repeated image loading and cropping operations.

        Returns:
            dict: Cache mapping (frame_name, iceberg_id) -> processed_crop
        """
        crop_cache = {}

        # Count total crops for progress reporting
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            unique_crops = sum(1 for _ in f)

        print(f"Precomputing {unique_crops} unique crops...")

        # Process each frame and its icebergs
        for frame_name, icebergs in self.icebergs_by_frame.items():
            image_path = os.path.join(self.image_dir, frame_name + self.image_format)

            # Load full image using OpenCV for consistency with bbox coordinates
            full_image = cv2.imread(image_path)

            # Process each iceberg in this frame
            for iceberg_id, iceberg_data in icebergs.items():
                x, y, w, h = iceberg_data['bbox']

                # Crop iceberg region
                crop = full_image[int(y):int(y + h), int(x):int(x + w)]

                # Convert to PIL Image for consistency with transforms
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                # Resize and pad to target size
                processed_crop = self._resize_and_pad(img)

                # Store in cache with composite key
                crop_cache[(frame_name, iceberg_id)] = processed_crop

        return crop_cache

    def _resize_and_pad(self, img):
        """
        Resize image while maintaining aspect ratio and pad to square.

        This method ensures all crops have the same dimensions while preserving
        the original aspect ratio of each iceberg. Smaller dimension is scaled
        to target size, then the image is padded with black pixels.

        Args:
            img (PIL.Image): Input image to resize and pad

        Returns:
            PIL.Image: Resized and padded square image
        """
        w, h = img.size

        # Calculate new dimensions maintaining aspect ratio
        if w > h:
            new_w = self.target_size
            new_h = int(h * self.target_size / w)
        else:
            new_h = self.target_size
            new_w = int(w * self.target_size / h)

        # Resize image
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Calculate padding needed to make square
        pad_w = (self.target_size - new_w) // 2
        pad_h = (self.target_size - new_h) // 2

        # Create padded image with black background
        padded_img = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        padded_img.paste(img, (pad_w, pad_h))

        return padded_img

    def __len__(self):
        """Return total number of pairs in the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Get a pair of images and their similarity label.

        This method retrieves a training pair consisting of two iceberg crops
        and their binary similarity label. For positive pairs, both crops show
        the same iceberg in consecutive frames. For negative pairs, the crops
        show different icebergs.

        Args:
            idx (int): Index of the pair to retrieve

        Returns:
            tuple: (img1, img2, label) where images are tensors and label is float
        """
        pair = self.pairs[idx]
        img1_name = pair["frame"]
        img2_name = pair["next_frame"]
        label = pair["label"]
        img1_id = pair["id"]

        # For positive pairs, use same iceberg ID; for negative pairs, use different ID
        img2_id = img1_id if label == 1 else pair["negative_candidate"]

        # Retrieve precomputed crops from cache
        img1 = self.crop_cache[(img1_name, img1_id)]
        img2 = self.crop_cache[(img2_name, img2_id)]

        # Apply transforms if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ============================================================================
# MAIN ORCHESTRATOR CLASS
# ============================================================================

class IcebergEmbeddingsTrainer:
    """
    Main orchestrator class for the training pipeline.

    This class coordinates all aspects of training a Siamese network for iceberg
    similarity learning:

    - Data loading and preprocessing
    - Model initialization and configuration
    - Training loop with validation
    - Early stopping and model checkpointing
    - Evaluation and metrics computation
    - Results visualization

    The trainer uses dependency injection through factory functions to allow
    for easy customization and testing of different components.

    Key Features:
    - Configurable through IcebergEmbeddingsConfig
    - Early stopping based on validation AUC
    - Automatic model checkpointing
    - Comprehensive training history tracking
    - Built-in evaluation and visualization

    Args:
        config (IcebergEmbeddingsConfig): Training configuration
        model_factory (callable, optional): Factory function for model creation
        dataset_factory (callable, optional): Factory function for dataset creation
        transform_factory (callable, optional): Factory function for transform creation
    """

    def __init__(self,
                 config: IcebergEmbeddingsConfig,
                 model_factory: Optional[Callable] = None,
                 dataset_factory: Optional[Callable] = None,
                 transform_factory: Optional[Callable] = None):
        """Initialize the trainer with configuration and optional factories."""
        self.config = config
        self.dataset = config.dataset
        self.model_path = os.path.join(DATA_DIR, self.dataset, "models", "embedding_model.pth")
        self.annotation_file = os.path.join(DATA_DIR, self.dataset, "annotations", "gt.txt")
        self.image_dir = os.path.join(DATA_DIR, self.dataset, "images", "raw")
        self.embeddings_path = os.path.join(DATA_DIR, self.dataset, "annotations", "iceberg_gt_embeddings.pt")

        # Set up factory functions (use defaults if not provided)
        self.model_factory = model_factory or self._default_model_factory
        self.dataset_factory = dataset_factory or self._default_dataset_factory
        self.transform_factory = transform_factory or self._default_transform_factory

        # Create models directory if it doesn't exist
        os.makedirs(os.path.join(DATA_DIR, self.dataset, "models"), exist_ok=True)

        # Initialize components (will be set up later)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Training history tracking
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'val_aucs': [],
            'best_auc': 0.0,
            'early_stopped': False,
            'final_epoch': 0
        }

    def _default_model_factory(self) -> SiameseNetwork:
        """
        Default factory for creating the Siamese network model.

        Returns:
            SiameseNetwork: Initialized Siamese network with ViT backbone
        """
        return SiameseNetwork(self.config)

    def _default_dataset_factory(self, transform, negative_ratio: float) -> IcebergComparisonDataset:
        """
        Default factory for creating comparison datasets.

        Args:
            transform (callable): Image transformations to apply
            negative_ratio (float): Ratio of negative to positive pairs

        Returns:
            IcebergComparisonDataset: Initialized dataset
        """
        return IcebergComparisonDataset(self.config, transform=transform, negative_ratio=negative_ratio)

    def _default_transform_factory(self) -> tuple:
        """
        Default factory for creating image transforms.

        Creates separate transform pipelines for training and validation:
        - Training: Includes data augmentation (color jitter, flips, rotation)
        - Validation: Only normalization (no augmentation)

        Returns:
            tuple: (train_transform, val_transform)
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

    def setup_data(self):
        """
        Setup data loaders for training and validation.

        Creates datasets with appropriate transforms and negative sampling ratios,
        then wraps them in DataLoaders with specified batch sizes and worker counts.
        """
        print("Setting up data loaders...")
        train_transform, val_transform = self.transform_factory()

        # Create datasets with different negative ratios for train/val
        train_dataset = self.dataset_factory(train_transform, self.config.negative_ratio_train)
        val_dataset = self.dataset_factory(val_transform, self.config.negative_ratio_val)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.config.num_workers
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=self.config.num_workers
        )

    def setup_model(self):
        """
        Setup model, optimizer, scheduler, and loss function.

        Initializes all components needed for training:
        - Model architecture and moves to device
        - Optimizer with weight decay
        - Learning rate scheduler
        - Combined loss function
        """
        print("Setting up model...")

        # Create model and move to device
        self.model = self.model_factory().to(self.device)

        # Print model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Using device: {self.device}")

        # Setup optimizer with AdamW (better than Adam for transformers)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Setup cosine annealing learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs
        )

        # Setup combined loss function
        self.criterion = CombinedLoss(
            margin=self.config.loss_margin,
            alpha=self.config.loss_alpha
        )

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Performs one complete pass through the training dataset:
        1. Sets model to training mode
        2. Iterates through all training batches
        3. Computes forward pass and loss
        4. Performs backpropagation and parameter updates
        5. Returns average loss for the epoch

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()  # Set model to training mode
        total_loss = 0.0

        for img1, img2, labels in self.train_loader:
            # Move data to device
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

            # Zero gradients from previous iteration
            self.optimizer.zero_grad()

            # Forward pass through Siamese network
            feat1, feat2 = self.model(img1, img2)

            # Compute loss
            loss = self.criterion(feat1, feat2, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_epoch(self) -> tuple:
        """
        Validate for one epoch.

        Performs validation on the validation dataset:
        1. Sets model to evaluation mode
        2. Disables gradient computation
        3. Computes predictions and losses
        4. Calculates metrics (loss and AUC)
        5. Returns validation metrics

        Returns:
            tuple: (average_validation_loss, validation_auc)
        """
        self.model.eval()  # Set model to evaluation mode
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

    def train(self):
        """
        Main training loop with early stopping.

        Orchestrates the complete training process:
        - Runs training and validation for each epoch
        - Tracks best model based on validation AUC
        - Implements early stopping to prevent overfitting
        - Updates learning rate schedule
        - Saves model checkpoints
        - Logs progress and metrics
        """
        print("Starting training...")

        best_auc = 0.0
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss = self.train_epoch()
            self.history['train_losses'].append(train_loss)

            # Validation phase
            val_loss, val_auc = self.validate_epoch()
            self.history['val_losses'].append(val_loss)
            self.history['val_aucs'].append(val_auc)

            # Check for improvement and save best model
            if val_auc > best_auc + self.config.min_delta:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                print(f"  ✅ New best model saved! AUC: {best_auc:.4f}")
            else:
                patience_counter += 1
                # Early stopping check
                if patience_counter >= self.config.patience:
                    print(f"  🛑 Early stopping triggered after {self.config.patience} epochs without improvement")
                    self.history['early_stopped'] = True
                    break

            # Update learning rate
            self.scheduler.step()

            # Print progress
            print(f'Epoch {epoch + 1}/{self.config.num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val AUC: {val_auc:.4f}')
            print(f'  Best AUC: {best_auc:.4f}')
            print(f'  Patience: {patience_counter}/{self.config.patience}')
            print('-' * 50)

        # Update final history
        self.history['best_auc'] = best_auc
        self.history['final_epoch'] = epoch + 1

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the trained model on validation set.

        Loads the best saved model and computes comprehensive evaluation metrics
        including AUC, Average Precision, and similarity distributions.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation results and metrics
        """
        print("Evaluating model...")

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

        print(f"Final Results:")
        print(f"AUC: {auc:.4f}")
        print(f"Average Precision: {ap:.4f}")

        return results

    def generate_iceberg_embeddings(self, txt_file, embeddings_path):
        """
        Extract feature vectors for all icebergs defined in an annotation file.

        This method loads a trained Siamese network model and uses it to generate
        feature embeddings for all iceberg detections in the ground truth file.
        The embeddings are cached for efficient future use.

        Args:
            txt_file (str): Path to annotation file containing iceberg detections
            embeddings_path (str): Path where computed embeddings will be saved
        """
        total_start_time = time.time()
        print("Generate iceberg embeddings...")

        # Step 1: Load the trained embedding model
        model = SiameseNetwork(self.config).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        # Step 2: Set up validation transforms (no data augmentation for inference)
        val_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        # Set model to evaluation mode (disables dropout, batch norm training mode, etc.)
        model = model.to(self.device)
        model.eval()

        # Step 3: Parse all detections from annotation file
        all_detections = parse_annotations(txt_file)

        # Sort detections by frame name
        # This groups all crops from the same image together, maximizing cache hits
        # for the DataLoader workers and dramatically improving performance
        all_detections.sort(key=lambda x: x['frame'])

        # Step 4: Create dataset and dataloader for batch processing
        dataset = IcebergEmbeddingsDataset(detections=all_detections,
                                           full_frame_dir=self.image_dir,
                                           transform=val_transform,
                                           target_size=224)

        # The cache works per-worker. With 4 workers, an image might be loaded
        # up to 4 times (once by each worker), but this is still a massive
        # improvement over loading it thousands of times without caching.
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        # Step 5: Extract features for all detections
        features_dict = {}
        print(f"Extracting features from {len(all_detections)} detected icebergs...")

        with torch.no_grad():  # Disable gradient computation for inference
            # Process images in batches for efficiency
            for image_batch, name_batch in dataloader:
                image_batch = image_batch.to(self.device)

                # Extract features using the backbone of the Siamese network
                feature_batch = model.backbone(image_batch)

                # Store features for each iceberg detection
                for i, img_name in enumerate(name_batch):
                    features_dict[img_name] = feature_batch[i].cpu()

        print("Generating embeddings complete.")

        # Step 6: Save computed features for future use
        torch.save(features_dict, embeddings_path)
        print(f"Embeddings saved to {embeddings_path}")

        elapsed_time = timedelta(seconds=int(time.time() - total_start_time))
        print(f"Generating embeddings completed in {elapsed_time}")

    def plot_results(self):
        """
        Plot training history and evaluation results.

        Creates a comprehensive visualization showing:
        1. Training and validation loss curves
        2. Validation AUC progression
        3. Distribution of similarity scores for positive vs negative pairs

        This helps understand model performance and training dynamics.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot training and validation losses
        axes[0].plot(self.history['train_losses'], label='Train Loss')
        axes[0].plot(self.history['val_losses'], label='Val Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot validation AUC over epochs
        axes[1].plot(self.history['val_aucs'], label='Val AUC', color='green')
        axes[1].set_title('Validation AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].legend()
        axes[1].grid(True)

        # Plot similarity score distributions
        if 'similarities' in self.history and 'labels' in self.history:
            # Separate similarities by label
            pos_sim = [s for s, l in zip(self.history['similarities'], self.history['labels']) if l == 1]
            neg_sim = [s for s, l in zip(self.history['similarities'], self.history['labels']) if l == 0]

            # Plot histograms
            axes[2].hist(pos_sim, alpha=0.7, label='Positive pairs', bins=30, color='green')
            axes[2].hist(neg_sim, alpha=0.7, label='Negative pairs', bins=30, color='red')
            axes[2].set_title('Similarity Distribution')
            axes[2].set_xlabel('Cosine Similarity')
            axes[2].set_ylabel('Frequency')
            axes[2].legend()
            axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training and evaluation pipeline.

        This is the main entry point that orchestrates the entire process:
        1. Data setup and preprocessing
        2. Model initialization
        3. Training with early stopping
        4. Final evaluation
        5. Results visualization
        6. Summary reporting

        Returns:
            Dict[str, Any]: Complete training and evaluation results
        """
        print("🚀 Starting Iceberg Similarity Training Pipeline")
        print("=" * 60)

        try:
            # Setup phase
            self.setup_data()
            self.setup_model()

            # Training phase
            self.train()

            # Evaluation phase
            self.evaluate()

            # Generation phase
            self.generate_iceberg_embeddings(self.annotation_file, self.embeddings_path)

            # Visualization phase
            self.plot_results()

            print("✅ Pipeline completed successfully!")

            # Summary reporting
            print(f"\n🎉 Training completed!")
            print(f"Final AUC: {self.history['best_auc']:.4f}")
            print(f"Early stopped: {self.history['early_stopped']}")
            print(f"Final epoch: {self.history['final_epoch']}")

            return self.history

        except Exception as e:
            print(f"❌ Pipeline failed with error: {str(e)}")
            raise


def main():
    # Configuration
    dataset = "hill_2min_2023-08"
    image_format = "JPG"

    # Create configuration and trainer
    config = IcebergEmbeddingsConfig(dataset=dataset, image_format=image_format)
    trainer = IcebergEmbeddingsTrainer(config)

    # Train model
    trainer.run_complete_pipeline()


if __name__ == "__main__":
    main()