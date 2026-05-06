"""
Mamba + Hybrid1D backbone model for iterative ANE testing.

Combines:
  - Hybrid1DBackbone: Conv1D feature extraction → Linear projection
  - MambaBlock: Stateful SSM-style block with EMA state buffers
  - StatefulMambaHybrid1D: Full model with state management
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hybrid1D Backbone (ported from model.py)
# ---------------------------------------------------------------------------

class Hybrid1DBackbone(nn.Module):
    """
    Hybrid 1D backbone: Conv1D + Linear layers with SiLU only.
    
    Designed for 1D sequential input or 2D images flattened to 1D.
    Uses Conv1D for feature extraction and Linear layers for projection.
    No BatchNorm or ReLU6 — only SiLU activations as requested.
    
    Args:
        in_channels: number of input channels (default 3 for image-like)
        hidden_dim: intermediate hidden dimension (default 256)
        output_dim: output feature dimension (default 512)
        seq_length: sequence length for Conv1D (default 224, assuming flattened 224×224 image)
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 512,
        seq_length: int = 224,
    ):
        super().__init__()
        self.seq_length = seq_length
        
        # Conv1D feature extraction: downsample with stride
        self.conv1 = nn.Conv1d(
            in_channels, hidden_dim, kernel_size=5, stride=2, padding=2, bias=True
        )
        self.silu1 = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv1d(
            hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2, bias=True
        )
        self.silu2 = nn.SiLU(inplace=True)
        
        # After 2× stride-2 convolutions: seq_length → seq_length/4
        conv_out_length = seq_length // 4
        
        # Linear layers for final projection
        self.fc1 = nn.Linear(hidden_dim * 2 * conv_out_length, hidden_dim * 2)
        self.silu3 = nn.SiLU(inplace=True)
        
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)
        self.silu4 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, seq_length) or (B, 3, 224) for image input
        
        Returns:
            (B, output_dim)
        """
        # Conv1D blocks
        x = self.silu1(self.conv1(x))           # (B, hidden_dim, seq_length/2)
        x = self.silu2(self.conv2(x))           # (B, hidden_dim*2, seq_length/4)
        
        # Flatten for linear layers
        x = x.flatten(1)                        # (B, hidden_dim*2 * seq_length/4)
        
        # Linear projection
        x = self.silu3(self.fc1(x))             # (B, hidden_dim*2)
        x = self.silu4(self.fc2(x))             # (B, output_dim)
        
        return x


# ---------------------------------------------------------------------------
# Mamba Block (iteratively developed)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Mamba-like SSM block with state buffers.
    
    Designed to be edited progressively for ANE testing:
    - Projections for Q, K, V
    - State update pattern (initially: slice_assign)
    - Ready to add: RoPE, trig functions, outer products, etc.
    
    Args:
        d_model: input/output dimension
        d_state: SSM state dimension (default 64)
        headdim: head dimension (default 64)
        num_heads: number of heads (default 8)
        ema_alpha: EMA blending coefficient (default 0.1)
    """
    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 64,
        headdim: int = 64,
        num_heads: int = 8,
        ema_alpha: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.num_heads = num_heads
        self.ema_alpha = ema_alpha
        
        self.d_inner = d_model * 2  # Expansion factor
        
        # Input projection: d_model → 3 * d_inner (Q, K, V)
        self.in_proj = nn.Linear(d_model, 3 * self.d_inner)
        
        # Output projection: d_inner → d_model
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # State buffers (registered but not yet used in forward pass)
        # These are placeholders for progressively adding features
        self.register_buffer("angle_state", torch.zeros(1, num_heads, 16))
        self.register_buffer("k_state", torch.zeros(1, 1, num_heads, d_state))
        self.register_buffer("v_state", torch.zeros(1, num_heads, headdim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_model)
        
        Returns:
            (B, d_model)
        """
        # Project input to Q, K, V
        # (B, d_model) → (B, 3 * d_inner)
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, d_inner)
        
        # Read state buffers to ensure they're part of the computation graph
        # This prevents CoreML from treating them as unused inputs
        angle_scale = self.angle_state.mean()  # (scalar)
        k_scale = self.k_state.mean()  # (scalar)
        v_scale = self.v_state.mean()  # (scalar)
        
        # Apply minimal gating using state information
        # This makes states active in the forward pass
        gate_scale = 1.0 + angle_scale * 0.01 + k_scale * 0.01 + v_scale * 0.01
        v_gated = v * gate_scale
        
        # Update state buffers in-place (slice assignment pattern from Phase 4)
        # Reshape v for state update
        v_for_state = v_gated[:, :self.num_heads * self.headdim].reshape(1, self.num_heads, self.headdim)
        self.v_state[:] = self.v_state * (1.0 - self.ema_alpha) + v_for_state * self.ema_alpha
        
        # Also update k_state and angle_state with dummy patterns
        k_for_state = k[:, :self.num_heads * self.d_state].reshape(1, 1, self.num_heads, self.d_state)
        self.k_state[:] = self.k_state * (1.0 - self.ema_alpha) + k_for_state * self.ema_alpha
        
        self.angle_state[:] = self.angle_state * (1.0 - self.ema_alpha) + \
                              q[:, :self.num_heads * 16].reshape(1, self.num_heads, 16) * self.ema_alpha
        
        # Output projection
        # (B, d_inner) → (B, d_model)
        out = self.out_proj(v_gated)
        
        return out


# ---------------------------------------------------------------------------
# Stateful Mamba + Hybrid1D Composite Model
# ---------------------------------------------------------------------------

class StatefulMambaHybrid1D(nn.Module):
    """
    StatefulMambaHybrid1D: Hybrid1DBackbone → MambaBlock → Classifier
    
    Combines a 1D feature extractor with a stateful Mamba block for ANE testing.
    State buffers persist across inferences via ct.StateType in CoreML export.
    
    Args:
        num_classes: number of output classes
        backbone_in_channels: input channels for Conv1D (default 3)
        backbone_hidden_dim: hidden dim in backbone (default 256)
        backbone_output_dim: backbone output dimension (default 512)
        seq_length: input sequence length (default 224)
        ema_alpha: EMA coefficient for state updates (default 0.1)
        mamba_d_state: SSM state dimension (default 64)
        mamba_headdim: head dimension (default 64)
        mamba_num_heads: number of heads (default 8)
    """
    def __init__(
        self,
        num_classes: int = 1000,
        backbone_in_channels: int = 3,
        backbone_hidden_dim: int = 256,
        backbone_output_dim: int = 512,
        seq_length: int = 224,
        ema_alpha: float = 0.1,
        mamba_d_state: int = 64,
        mamba_headdim: int = 64,
        mamba_num_heads: int = 8,
    ):
        super().__init__()
        
        # Backbone: Conv1D feature extraction
        self.backbone = Hybrid1DBackbone(
            in_channels=backbone_in_channels,
            hidden_dim=backbone_hidden_dim,
            output_dim=backbone_output_dim,
            seq_length=seq_length,
        )
        
        # Mamba block
        self.mamba = MambaBlock(
            d_model=backbone_output_dim,
            d_state=mamba_d_state,
            headdim=mamba_headdim,
            num_heads=mamba_num_heads,
            ema_alpha=ema_alpha,
        )
        
        # Classifier
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(backbone_output_dim, num_classes)
        
        # Store metadata
        self.feature_dim = backbone_output_dim
        self.ema_alpha = ema_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, seq_length) or (B, 3, 224) for 224×224 images
        
        Returns:
            (B, num_classes)
        """
        # Extract features via Hybrid1D backbone
        # (B, 3, 224) → (B, backbone_output_dim)
        features = self.backbone(x)
        
        # Process through Mamba block (with state updates)
        # (B, backbone_output_dim) → (B, backbone_output_dim)
        mamba_out = self.mamba(features)
        
        # Classification
        logits = self.classifier(self.dropout(mamba_out))
        
        return logits


# ---------------------------------------------------------------------------
# Backward compatibility: keep MambaPhase0 for any existing references
# ---------------------------------------------------------------------------

class MambaPhase0(nn.Module):
    """
    Test de survie ANE : Projections + Allocation des immenses buffers d'état.
    AUCUNE mathématique complexe.
    (Kept for backward compatibility)
    """
    def __init__(self, d_model=256, d_state=64, headdim=64, num_rope_angles=16):
        super().__init__()
        self.d_model = d_model
        self.nheads = (d_model * 2) // headdim
        self.d_inner = d_model * 2
        
        # Projections pures
        self.in_proj = nn.Linear(d_model, self.d_inner * 3)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # ── LES ÉTATS ──
        self.register_buffer("angle_state", torch.zeros(1, self.nheads, num_rope_angles))
        self.register_buffer("ssm_state", torch.zeros(1, self.nheads, headdim, d_state))
        self.register_buffer("k_state", torch.zeros(1, 1, self.nheads, d_state))
        self.register_buffer("v_state", torch.zeros(1, self.nheads, headdim))

    def forward(self, u: torch.Tensor):
        # u: (1, d_model)
        x = self.in_proj(u)
        self.angle_state[:] = self.angle_state * 0.99 + 0.01
        self.ssm_state[:] = self.ssm_state * 0.99 + 0.01
        self.k_state[:] = self.k_state * 0.99 + 0.01
        self.v_state[:] = self.v_state * 0.99 + 0.01
        y = x[..., :self.d_inner]
        out = self.out_proj(y)
        return out


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== StatefulMambaHybrid1D — PyTorch sanity check ===\n")
    
    model = StatefulMambaHybrid1D(
        num_classes=1000,
        backbone_in_channels=3,
        backbone_hidden_dim=256,
        backbone_output_dim=512,
        seq_length=224,
        ema_alpha=0.1,
        mamba_d_state=64,
        mamba_headdim=64,
        mamba_num_heads=8,
    )
    
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres: {n_params / 1e6:.2f}M")
    print(f"Feature dim: {model.feature_dim}")
    print(f"EMA alpha: {model.ema_alpha}")
    print(f"Mamba states:")
    print(f"  - angle_state: {model.mamba.angle_state.shape}")
    print(f"  - k_state: {model.mamba.k_state.shape}")
    print(f"  - v_state: {model.mamba.v_state.shape}\n")
    
    # Simulate 5 frames
    print("Simulation 5 frames:")
    for i in range(5):
        x = torch.randn(1, 3, 224)
        with torch.no_grad():
            logits = model(x)
        print(f"  Frame {i}: input {x.shape} → logits {logits.shape} ✓")
    
    print("\n✓ Forward pass OK — state buffers ready for progressive development")