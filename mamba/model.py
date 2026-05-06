"""
Mamba + Hybrid1D backbone model for iterative ANE testing.

Combines:
  - Hybrid1DBackbone: Conv1D feature extraction → Linear projection
  - MambaBlock: Stateful SSM-style block with EMA state buffers
  - StatefulMambaHybrid1D: Full model with state management
"""

import sys
import math
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

    Phase 1: Slicing Explicite & Activations.
    Test de la tolérance de l'ANE aux non-linéarités (softplus, sigmoid, silu, clamp)
    et au découpage asymétrique des tenseurs.

    Phase 2: Discretisation & SSM Recurrence (Outer Products).
    Test de la fusion d'état via matmul et de l'intégration temporelle.

    Phase 3: Le Boss de Fin (Trigonométrie & RoPE).
    Intégration du Rotary Position Embedding complet de Mamba3.

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
        
        self.d_inner = self.num_heads * self.headdim 
        self.num_rope_angles = 16
        self.rotary_dim = self.num_rope_angles * 2 # 32

        self._s0 = self.d_inner
        self._s1 = self._s0 + self.d_inner
        self._s2 = self._s1 + self.d_state
        self._s3 = self._s2 + self.d_state
        self._s4 = self._s3 + self.num_heads
        self._s5 = self._s4 + self.num_heads
        self._s6 = self._s5 + self.num_heads
        self._s7 = self._s6 + self.num_rope_angles

        d_in_proj = self._s7

        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.register_buffer("angle_state", torch.zeros(1, num_heads, self.num_rope_angles))
        self.register_buffer("ssm_state", torch.zeros(1, num_heads, headdim, d_state))
        self.register_buffer("k_state", torch.zeros(1, 1, num_heads, d_state))
        self.register_buffer("v_state", torch.zeros(1, num_heads, headdim))

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Application statique du RoPE, optimisée pour l'ANE (sans `if` dynamique).
        x shape: (1, 8, 64) | cos/sin shape: (1, 8, 16)
        """
        x_rot = x[..., :self.rotary_dim] # Les 32 premiers éléments
        x_pass = x[..., self.rotary_dim:] # Les 32 derniers éléments intacts
        
        x0 = x_rot[..., 0::2] # Shape: (1, 8, 16)
        x1 = x_rot[..., 1::2] # Shape: (1, 8, 16)
        
        xo0 = x0 * cos - x1 * sin
        xo1 = x0 * sin + x1 * cos
        
        out_rot = torch.stack([xo0, xo1], dim=-1).flatten(-2) # Recombine en (1, 8, 32)
        return torch.cat([out_rot, x_pass], dim=-1) # Recolage -> (1, 8, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zxBCdt = self.in_proj(x)
        
        z_raw      = zxBCdt[..., :self._s0]
        x_raw      = zxBCdt[..., self._s0:self._s1]
        B_raw      = zxBCdt[..., self._s1:self._s2]
        C_raw      = zxBCdt[..., self._s2:self._s3]
        dd_dt      = zxBCdt[..., self._s3:self._s4]
        dd_A       = zxBCdt[..., self._s4:self._s5]
        trap_raw   = zxBCdt[..., self._s5:self._s6]
        angles_raw = zxBCdt[..., self._s6:self._s7]

        # 3. Activations pures (Sans clamp = zéro constante infinie dans le graphe)
        A   = -F.softplus(dd_A)
        DT  = F.softplus(dd_dt)
        lam = torch.sigmoid(trap_raw)
        z   = F.silu(z_raw)

        # Discretisation Coefficients
        alpha = torch.exp(A * DT)
        beta  = (1.0 - lam) * DT * alpha
        gamma = lam * DT

        # --- NOUVEAUTÉ : RoPE ---
        # 1. Calcul et accumulation de l'angle
        angles = angles_raw.view(1, 1, self.num_rope_angles).repeat(1, self.num_heads, 1)
        delta_theta = torch.tanh(angles) * math.pi * DT.unsqueeze(-1)
        theta = self.angle_state + delta_theta
        
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # 2. Préparation et Rotation de K et Q
        K_pre = B_raw.view(1, 1, self.d_state).repeat(1, self.num_heads, 1) # (1, 8, 64)
        Q_pre = C_raw.view(1, 1, self.d_state).repeat(1, self.num_heads, 1) # (1, 8, 64)

        K_rot = self.apply_rope(K_pre, cos_theta, sin_theta)
        Q_rot = self.apply_rope(Q_pre, cos_theta, sin_theta)

        # 3. Préparation de V
        V = x_raw.reshape(1, self.num_heads, self.headdim)

        # --- SSM Recurrence ---
        v_prev = self.v_state
        k_prev = self.k_state.squeeze(1)

        outer_curr = torch.matmul(V.unsqueeze(-1), K_rot.unsqueeze(-2))
        outer_prev = torch.matmul(v_prev.unsqueeze(-1), k_prev.unsqueeze(-2))

        g4 = gamma[:, :, None, None]
        b4 = beta[:, :, None, None]
        delta_h = b4 * outer_prev + g4 * outer_curr

        new_h = alpha[:, :, None, None] * self.ssm_state + delta_h

        # --- State Updates (API StateType) ---
        self.ssm_state[:] = new_h
        self.k_state[:] = K_rot.unsqueeze(1)
        self.v_state[:] = V
        self.angle_state[:] = theta

        # --- Output y (Produit tensoriel de sortie) ---
        y_ssm = torch.matmul(new_h, Q_rot.unsqueeze(-1)).squeeze(-1) # (1, 8, 64)
        y_flat = y_ssm.reshape(1, self.d_inner) # (1, 512)
        
        y = y_flat * z
        out = self.out_proj(y)
        
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