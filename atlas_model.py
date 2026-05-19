"""
ATLAS Phase 4 — GraphSAGE GNN Model for Assembly Constraint Prediction
========================================================================

Architecture:
    - 3-layer GraphSAGE encoder (learns structural node embeddings)
    - Link-prediction head (predicts constraint type between face pairs)
    - Multi-class classification: None / Mate / Flush / Insert / Angle

The link predictor concatenates [emb_i || emb_j || |emb_i - emb_j| || emb_i * emb_j]
for each candidate face pair, producing a 4*hidden_dim vector fed through an MLP.

USAGE:
    from atlas_model import AtlasGNN
    model = AtlasGNN(in_channels=21, hidden_channels=128, num_classes=5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, BatchNorm
except ImportError:
    raise ImportError(
        "PyTorch Geometric is required. Install via:\n"
        "  pip install torch-geometric torch-scatter torch-sparse"
    )


class AtlasGNN(nn.Module):
    """
    3-layer GraphSAGE with a link-prediction head for constraint classification.

    Args:
        in_channels:     Input feature dimension (21 for ATLAS)
        hidden_channels: Hidden embedding dimension (default 128)
        num_classes:     Number of constraint classes (5: None/Mate/Flush/Insert/Angle)
        dropout:         Dropout rate during training
    """

    def __init__(
        self,
        in_channels: int = 21,
        hidden_channels: int = 128,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.dropout = dropout
        self.num_classes = num_classes

        # ── GraphSAGE Encoder (3 layers) ─────────────────────────────────────
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)

        # ── Link Prediction Head ─────────────────────────────────────────────
        # Input: [emb_i || emb_j || |emb_i - emb_j| || emb_i * emb_j]
        link_input_dim = hidden_channels * 4

        self.link_predictor = nn.Sequential(
            nn.Linear(link_input_dim, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

        # ── Weight initialization ────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for m in self.link_predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode node features using 3-layer GraphSAGE.

        Args:
            x:          Node feature matrix [N, in_channels]
            edge_index: Structural (B-Rep) edge indices [2, E]

        Returns:
            Node embeddings [N, hidden_channels]
        """
        # Layer 1
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 3
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)

        return h

    def predict_links(
        self, embeddings: torch.Tensor, edge_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict constraint classes for candidate face pairs.

        Args:
            embeddings:  Node embeddings [N, hidden_channels]
            edge_pairs:  Candidate edge indices [2, K] (src, dst)

        Returns:
            Class logits [K, num_classes]
        """
        src = embeddings[edge_pairs[0]]  # [K, H]
        dst = embeddings[edge_pairs[1]]  # [K, H]

        # Symmetric features: concat, abs-diff, element-product
        link_features = torch.cat([
            src,
            dst,
            torch.abs(src - dst),
            src * dst,
        ], dim=-1)  # [K, 4*H]

        return self.link_predictor(link_features)  # [K, num_classes]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass: encode → predict links.

        Args:
            x:                 Node features [N, 21]
            edge_index:        Structural edges [2, E]
            target_edge_index: Candidate constraint edges [2, K]

        Returns:
            Constraint class logits [K, num_classes]
        """
        embeddings = self.encode(x, edge_index)
        logits = self.predict_links(embeddings, target_edge_index)
        return logits

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings without link prediction (for inference)."""
        return self.encode(x, edge_index)
