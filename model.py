"""
StatefulMobileNet — MobileNetV2-like CNN + CoreML 9.0 stateful feature extractor.

Architecture:
  - Backbone MobileNetV2-style (depthwise separable convs + inverted residuals)
  - Temporal state: running EMA des features (register_buffer → ct.StateType)
  - Output: logits + state mis à jour automatiquement entre les inférences

Usage:
  python model.py          # forward pass PyTorch seul (sanity check)
  python export.py         # export vers .mlpackage avec state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase 1: MLP Backbone
# ---------------------------------------------------------------------------

class MLPBackbone(nn.Module):
    """Simple MLP backbone for Phase 1 testing — linear layers + SiLU."""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        return self.mlp(x)  # (B, output_dim)


class HybridBackbone(nn.Module):
    """Hybrid backbone: CNN + MLP for increased ANE workload."""
    def __init__(self, in_channels: int = 3, hidden_dim: int = 256, output_dim: int = 512):
        super().__init__()
        # Lightweight CNN: 2 conv blocks
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
        )
        # MLP: 2 layers
        self.mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.conv(x)                        # (B, 128, H/4, W/4)
        x = F.adaptive_avg_pool2d(x, 1)         # (B, 128, 1, 1)
        x = x.flatten(1)                        # (B, 128)
        x = self.mlp(x)                         # (B, output_dim)
        return x


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
# Blocs de base MobileNetV2
# ---------------------------------------------------------------------------

class ConvBNReLU6(nn.Sequential):
    """Conv 2D + BN + ReLU6."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1):
        padding = (kernel - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNetV2 style)."""

    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = int(round(in_ch * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_ch, hidden, kernel=1))
        layers += [
            ConvBNReLU6(hidden, hidden, stride=stride, groups=hidden),  # depthwise
            nn.Conv2d(hidden, out_ch, 1, bias=False),                   # pointwise
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Backbone MobileNetV2-like
# ---------------------------------------------------------------------------

# (expand_ratio, out_ch, n_blocks, stride)
MV2_CONFIG = [
    (1,  16, 1, 1),
    (6,  24, 2, 2),
    (6,  32, 3, 2),
    (6,  64, 4, 2),
    (6,  96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def _make_backbone(width_mult=1.0):
    in_ch = _round(32 * width_mult)
    layers = [ConvBNReLU6(3, in_ch, stride=2)]

    for t, c, n, s in MV2_CONFIG:
        out_ch = _round(c * width_mult)
        for i in range(n):
            layers.append(InvertedResidual(in_ch, out_ch, stride=s if i == 0 else 1, expand_ratio=t))
            in_ch = out_ch

    last_ch = _round(1280 * width_mult)
    layers.append(ConvBNReLU6(in_ch, last_ch, kernel=1))
    return nn.Sequential(*layers), last_ch


def _round(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ---------------------------------------------------------------------------
# StatefulMobileNet  ← le modèle principal
# ---------------------------------------------------------------------------

class StatefulMobileNet(nn.Module):
    """
    MobileNetV2-like avec un état temporel (EMA des feature maps).

    Le buffer `feature_state` sera converti en ct.StateType lors de l'export.
    Il accumule une moyenne exponentielle des features globales entre les frames,
    ce qui simule un contexte temporel persistant — utile pour les streams vidéo
    ou les séquences haptiques (ICANSII context !).

    Args:
        num_classes: nombre de classes de sortie
        width_mult:  facteur de largeur MobileNetV2 (0.5, 0.75, 1.0...)
        ema_alpha:   coefficient EMA (0 = état figé, 1 = pas de mémoire)
        feature_dim: dimension du vecteur de features global (après GAP)
        backbone_type: "cnn" (Phase 0), "mlp" (Phase 1), "hybrid" (Phase 1.5), "hybrid1d" (Conv1D + Linear)
        input_dim:   pour MLP, dimension d'entrée vectorielle
        phase2:      si True, reshape feature_state en 4D (1, nheads, headdim, d_state)
    """

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        ema_alpha: float = 0.1,
        feature_dim: int | None = None,
        backbone_type: str = "cnn",
        input_dim: int = 256,
        phase2: bool = False,
        phase3: bool = False,
        num_states: int = 1,
        phase4: bool = False,
        phase4_pattern: str = "slice_assign_with_cast",
        phase5: bool = False,
        phase5_pattern: str = "matmul",
        phase6a: bool = False,
        phase6b: bool = False,
        phase6c: bool = False,
    ):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.backbone_type = backbone_type
        self.input_dim = input_dim
        self.phase2 = phase2
        self.phase3 = phase3
        self.num_states = num_states
        self.phase4 = phase4
        self.phase4_pattern = phase4_pattern
        self.phase5 = phase5
        self.phase5_pattern = phase5_pattern
        self.phase6a = phase6a
        self.phase6b = phase6b
        self.phase6c = phase6c
        
        # Phase 4 validation
        valid_patterns = [
            "addition",
            "mul",
            "copy",
            "clone",
            "slice_assign_with_cast",
            "slice_assign_no_cast",
        ]
        if self.phase4 and self.phase4_pattern not in valid_patterns:
            raise ValueError(
                f"Invalid phase4_pattern '{self.phase4_pattern}'. "
                f"Valid patterns: {valid_patterns}"
            )
        
        # Phase 5 validation
        valid_phase5_patterns = ["einsum", "matmul"]
        if self.phase5 and self.phase5_pattern not in valid_phase5_patterns:
            raise ValueError(
                f"Invalid phase5_pattern '{self.phase5_pattern}'. "
                f"Valid patterns: {valid_phase5_patterns}"
            )
        
        # Phase 6: Trigonometry & RoPE bisection
        # phase6a ⇒ tanh + softplus only
        # phase6b ⇒ + cos/sin accumulation (implies 6a)
        # phase6c ⇒ + full pairwise rotation (implies 6b and 6a)
        if self.phase6c:
            self.phase6b = True
            self.phase6a = True
        elif self.phase6b:
            self.phase6a = True
        
        # Phase 6: Auto-enable ALL previous phases (2, 3, 4, 5) with optimal settings
        if self.phase6a or self.phase6b or self.phase6c:
            self.phase2 = True
            self.phase3 = True
            self.phase4 = True
            self.phase5 = True
            self.num_states = 3
            self.phase4_pattern = "mul"
            self.phase5_pattern = "matmul"
        
        # Phase 5: Auto-enable ALL previous phases (2, 3, 4) with optimal settings
        elif self.phase5:
            self.phase2 = True
            self.phase3 = True
            self.phase4 = True
            self.num_states = 3  # Phase 5 uses 3 states: angle, k, v
            self.phase4_pattern = "mul"  # Use optimal pattern from Phase 4
        
        # Phase 4: Auto-enable Phase 2 and Phase 3 with 3 states
        elif self.phase4:
            self.phase2 = True
            self.phase3 = True
            self.num_states = 3  # Phase 4 uses 3 states: angle, k, v

        # Backbone
        if backbone_type == "cnn":
            self.backbone, last_ch = _make_backbone(width_mult)
        elif backbone_type == "mlp":
            # Phase 1: MLP backend
            hidden_dim = 512
            self.backbone = MLPBackbone(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            )
            last_ch = hidden_dim
        elif backbone_type == "hybrid":
            # Phase 1.5: CNN + MLP (higher ANE workload)
            hidden_dim = 256
            output_dim = 512
            self.backbone = HybridBackbone(
                in_channels=3,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            )
            last_ch = output_dim
        elif backbone_type == "hybrid1d":
            # Hybrid 1D: Conv1D + Linear with SiLU only
            hidden_dim = 256
            output_dim = 512
            seq_length = 224  # Default for 224×224 flattened images or 1D sequences
            self.backbone = Hybrid1DBackbone(
                in_channels=3,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                seq_length=seq_length,
            )
            last_ch = output_dim
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # Projection vers feature_dim si spécifié
        self.feature_dim = feature_dim or last_ch
        if self.feature_dim != last_ch:
            self.proj = nn.Linear(last_ch, self.feature_dim)
        else:
            self.proj = nn.Identity()

        # Phase 2: Projection vers 4D state (1, nheads, headdim, d_state)
        # Contrainte: nheads * headdim * d_state = feature_dim (pour que la fusion soit possible)
        # Exemple: feature_dim=1280 → (1, 8, 20, 8) = 8*20*8 = 1280
        #          feature_dim=512  → (1, 8, 8, 8) = 8*8*8 = 512
        if self.phase2:
            state_nheads = 8
            d_inner = self.feature_dim // state_nheads  # e.g., 1280 / 8 = 160 or 512 / 8 = 64
            
            # Factorize d_inner into headdim * d_state
            # Simple heuristic: try to balance them
            state_headdim = int(d_inner ** 0.5)  # approx sqrt
            # Round to nearest divisor
            while d_inner % state_headdim != 0:
                state_headdim -= 1
            state_dstate = d_inner // state_headdim
            
            # Validation
            assert state_headdim * state_dstate == d_inner, \
                f"headdim*d_state ({state_headdim}*{state_dstate}) must equal feature_dim/nheads ({d_inner})"
            
            self.state_shape = (1, state_nheads, state_headdim, state_dstate)
            self.state_flatdim = self.feature_dim  # Keep same as feature_dim for fusion
            self.state_proj = nn.Identity()  # No projection needed
        else:
            self.state_shape = (1, self.feature_dim)

        # Classifier
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # ── STATE ──────────────────────────────────────────────────────────
        # register_buffer → persistant dans le modèle, traceable par TorchScript
        # Nom "feature_state" = celui qu'on passera à ct.StateType(name=...)
        # Phase 3: Multiple state buffers REPLACE feature_state
        # Phase 2: 4D feature_state
        # Phase 0/1: 2D feature_state
        
        if not self.phase3:
            # Only register feature_state if NOT Phase 3
            if self.phase2:
                self.register_buffer(
                    "feature_state",
                    torch.zeros(*self.state_shape, dtype=torch.float32),
                )
            else:
                self.register_buffer(
                    "feature_state",
                    torch.zeros(1, self.feature_dim, dtype=torch.float32),
                )
        
        # Phase 3: Multiple state buffers
        # Shapes per plan.md: angle_state (1, 8, 16), k_state (1, 1, 8, 64), v_state (1, 8, 64)
        if self.phase3:
            self.state_shapes = {}
            
            # State 1: angle_state (1, 8, 16) = 128 elements
            self.register_buffer(
                "angle_state",
                torch.zeros(1, 8, 16, dtype=torch.float32),
            )
            self.state_shapes["angle_state"] = (1, 8, 16)
            # Projection layer for angle_state
            self.angle_proj = nn.Linear(self.feature_dim, 8 * 16)
            
            # State 2: k_state (1, 1, 8, 64) = 512 elements
            self.register_buffer(
                "k_state",
                torch.zeros(1, 1, 8, 64, dtype=torch.float32),
            )
            self.state_shapes["k_state"] = (1, 1, 8, 64)
            # Projection layer for k_state
            self.k_proj = nn.Linear(self.feature_dim, 1 * 8 * 64)
            
            # State 3: v_state (1, 8, 64) = 512 elements
            self.register_buffer(
                "v_state",
                torch.zeros(1, 8, 64, dtype=torch.float32),
            )
            self.state_shapes["v_state"] = (1, 8, 64)
            # Projection layer for v_state
            self.v_proj = nn.Linear(self.feature_dim, 8 * 64)
            
            # State 4 (optional): dv_state (1, 8, 64) = 512 elements (for phase 3.2)
            if self.num_states >= 4:
                self.register_buffer(
                    "dv_state",
                    torch.zeros(1, 8, 64, dtype=torch.float32),
                )
                self.state_shapes["dv_state"] = (1, 8, 64)
                # Projection layer for dv_state
                self.dv_proj = nn.Linear(self.feature_dim, 8 * 64)
            
            # Phase 5: SSM state for Mamba-style outer product fusion
            # ssm_state: (1, 8, 64, 64) = 32768 elements — the main fusion state
            if self.phase5:
                self.register_buffer(
                    "ssm_state",
                    torch.zeros(1, 8, 64, 64, dtype=torch.float32),
                )
                self.state_shapes["ssm_state"] = (1, 8, 64, 64)
                # Scalars for trapezoid mixing (a, b, g coefficients)
                self.register_buffer("a_coeff", torch.tensor(0.9, dtype=torch.float32))
                self.register_buffer("b_coeff", torch.tensor(0.5, dtype=torch.float32))
                self.register_buffer("g_coeff", torch.tensor(0.5, dtype=torch.float32))
            
            # Phase 6: Trigonometry & RoPE bisection
            # Phase 6a: tanh + softplus gating
            # Phase 6b: + cos/sin accumulation
            # Phase 6c: + full pairwise rotation
            if self.phase6a:
                # Constants for Phase 6 (locked per Mamba3)
                self.num_angles = 16
                self.rotary_dim = 2 * self.num_angles  # 32
                
                # Phase 6a projection layers
                self.theta_proj = nn.Linear(self.feature_dim, 8 * self.num_angles)  # (8, 16)
                self.dt_proj = nn.Linear(self.feature_dim, 8)  # (8,) for nheads
                
                # Phase 6b: cos/sin states (if enabled)
                if self.phase6b:
                    self.register_buffer(
                        "theta_state",
                        torch.zeros(1, 8, 16, dtype=torch.float32),
                    )
                    self.state_shapes["theta_state"] = (1, 8, 16)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _update_states_phase4(
        self,
        angle_feats: torch.Tensor,
        k_feats: torch.Tensor,
        v_feats: torch.Tensor,
    ) -> None:
        """
        Phase 4: Apply different state write patterns to test which breaks ANE.
        
        Patterns:
        - "addition": new allocation
        - "mul": in-place mul_.add_()
        - "copy": in-place copy_()
        - "clone": detach + clone
        - "slice_assign_with_cast": explicit to(float16)
        - "slice_assign_no_cast": implicit dtype preservation
        """
        alpha = self.ema_alpha
        feats_detached_angle = angle_feats.detach()
        feats_detached_k = k_feats.detach()
        feats_detached_v = v_feats.detach()
        
        if self.phase4_pattern == "addition":
            # New allocation: state = state + (1 - α) * features
            self.angle_state = self.angle_state + (1.0 - alpha) * feats_detached_angle
            self.k_state = self.k_state + (1.0 - alpha) * feats_detached_k
            self.v_state = self.v_state + (1.0 - alpha) * feats_detached_v
        
        elif self.phase4_pattern == "mul":
            # In-place mul_.add_() (baseline method)
            self.angle_state.mul_(1.0 - alpha).add_(feats_detached_angle * alpha)
            self.k_state.mul_(1.0 - alpha).add_(feats_detached_k * alpha)
            self.v_state.mul_(1.0 - alpha).add_(feats_detached_v * alpha)
        
        elif self.phase4_pattern == "copy":
            # In-place copy_()
            new_angle = feats_detached_angle * alpha + self.angle_state * (1.0 - alpha)
            new_k = feats_detached_k * alpha + self.k_state * (1.0 - alpha)
            new_v = feats_detached_v * alpha + self.v_state * (1.0 - alpha)
            
            self.angle_state.copy_(new_angle)
            self.k_state.copy_(new_k)
            self.v_state.copy_(new_v)
        
        elif self.phase4_pattern == "clone":
            # Detach + clone
            new_angle = self.angle_state.detach().clone() + feats_detached_angle
            new_k = self.k_state.detach().clone() + feats_detached_k
            new_v = self.v_state.detach().clone() + feats_detached_v
            
            self.angle_state = new_angle
            self.k_state = new_k
            self.v_state = new_v
        
        elif self.phase4_pattern == "slice_assign_with_cast":
            # Explicit cast to float16 via slice assignment
            new_angle = feats_detached_angle * alpha + self.angle_state * (1.0 - alpha)
            new_k = feats_detached_k * alpha + self.k_state * (1.0 - alpha)
            new_v = feats_detached_v * alpha + self.v_state * (1.0 - alpha)
            
            self.angle_state[:] = new_angle.to(torch.float16)
            self.k_state[:] = new_k.to(torch.float16)
            self.v_state[:] = new_v.to(torch.float16)
        
        elif self.phase4_pattern == "slice_assign_no_cast":
            # Slice assignment without explicit cast (trust buffer dtype)
            new_angle = feats_detached_angle * alpha + self.angle_state * (1.0 - alpha)
            new_k = feats_detached_k * alpha + self.k_state * (1.0 - alpha)
            new_v = feats_detached_v * alpha + self.v_state * (1.0 - alpha)
            
            self.angle_state[:] = new_angle
            self.k_state[:] = new_k
            self.v_state[:] = new_v

    def _update_states_phase5(
        self,
        angle_feats: torch.Tensor,
        k_feats: torch.Tensor,
        v_feats: torch.Tensor,
    ) -> None:
        """
        Phase 5: Mamba-style outer product state fusion using trapezoid rule.
        
        Computes outer products of V × K (previous and current) and blends them
        with the SSM state using trapezoid mixing rule:
            delta_h = b * outer_prev + g * outer_curr
            new_h = a * ssm_state + delta_h
        
        Patterns:
        - "einsum": torch.einsum("bhp,bhs->bhps", V, K)
        - "matmul": torch.matmul(V.unsqueeze(-1), K.unsqueeze(-2))
        
        State shapes:
        - ssm_state: (1, nheads=8, headdim=64, d_state=64)
        - k_state: (1, 1, nheads=8, d_state=64) → squeezed to (1, 8, 64)
        - v_state: (1, nheads=8, headdim=64)
        """
        # Extract the current step features
        # V: (1, 8, 64) — current V from backbone
        # K: (1, 1, 8, 64) → squeeze to (1, 8, 64) for current K
        V = v_feats.detach()  # (1, 8, 64)
        K = k_feats.detach().squeeze(1)  # (1, 1, 8, 64) → (1, 8, 64)
        
        # Previous step features from state
        Vp = self.v_state.detach()  # (1, 8, 64)
        Kp = self.k_state.detach().squeeze(1)  # (1, 1, 8, 64) → (1, 8, 64)
        
        # === OUTER PRODUCT COMPUTATION ===
        # Option 1: einsum-based (user can add .float() after each tensor if needed for precision)
        # Option 2: matmul-based (vectorized unsqueeze)
        
        if self.phase5_pattern == "einsum":
            # Use torch.einsum for outer product: (B, H, P) x (B, H, S) → (B, H, P, S)
            # Note: If precision issues arise, add .float() after V, K, Vp, Kp above
            outer_prev = torch.einsum("bhp,bhs->bhps", Vp, Kp)  # (1, 8, 64, 64)
            outer_curr = torch.einsum("bhp,bhs->bhps", V, K)    # (1, 8, 64, 64)
        
        elif self.phase5_pattern == "matmul":
            # Alternative: torch.matmul with unsqueeze
            # Shapes: V.unsqueeze(-1) = (1, 8, 64, 1), K.unsqueeze(-2) = (1, 8, 1, 64)
            # Note: If precision issues arise, add .float() after V, K, Vp, Kp above
            outer_prev = torch.matmul(
                Vp.unsqueeze(-1),  # (1, 8, 64, 1)
                Kp.unsqueeze(-2),  # (1, 8, 1, 64)
            )  # → (1, 8, 64, 64)
            outer_curr = torch.matmul(
                V.unsqueeze(-1),   # (1, 8, 64, 1)
                K.unsqueeze(-2),   # (1, 8, 1, 64)
            )  # → (1, 8, 64, 64)
        
        else:
            raise ValueError(f"Unknown phase5_pattern: {self.phase5_pattern}")
        
        # === TRAPEZOID MIXING RULE ===
        # delta_h = b * outer_prev + g * outer_curr
        # new_h = a * ssm_state + delta_h
        # Note: If precision issues arise, add .float() before / after operations:
        #   delta_h = self.b_coeff.float() * outer_prev.float() + ...
        delta_h = self.b_coeff * outer_prev + self.g_coeff * outer_curr  # (1, 8, 64, 64)
        new_h = self.a_coeff * self.ssm_state + delta_h  # (1, 8, 64, 64)
        
        # === STATE UPDATE (in-place, using Phase 4 "mul" pattern) ===
        # mul_.add_() for minimal CoreML overhead
        self.ssm_state.mul_(1.0 - self.ema_alpha).add_(new_h * self.ema_alpha)
        
        # Also update k_state and v_state with Phase 4 "mul" pattern (in-place)
        self.k_state.mul_(1.0 - self.ema_alpha).add_(k_feats.detach() * self.ema_alpha)
        self.v_state.mul_(1.0 - self.ema_alpha).add_(v_feats.detach() * self.ema_alpha)

    def _update_states_phase6(
        self,
        feats: torch.Tensor,
        k_feats: torch.Tensor,
        v_feats: torch.Tensor,
    ) -> None:
        """
        Phase 6: Trigonometry & RoPE bisection test.
        
        - Phase 6a: tanh + softplus gating only
        - Phase 6b: + cos/sin accumulation  
        - Phase 6c: + full pairwise rotation
        
        Reference: Mamba3 RoPE pipeline (lines 326-353 of mamba3_siso_portable.py)
        """
        import math
        
        # ===== PHASE 6A: Tanh + softplus =====
        # Project features to angles and dt
        angles_flat = self.theta_proj(feats)  # (B, 8*16)
        angles = angles_flat.view(1, 8, self.num_angles)  # (1, 8, 16)
        
        dt_flat = self.dt_proj(feats)  # (B, 8)
        dt = F.softplus(dt_flat).view(1, 8)  # (1, 8)
        
        # Compute gating: gate = tanh(angles) * dt
        # (1, 8, 16) * (1, 8, 1) → (1, 8, 16)
        gate = torch.tanh(angles) * dt.unsqueeze(-1)  # (1, 8, 16)
        
        # ===== PHASE 6B: Cos/Sin accumulation =====
        if self.phase6b:
            # Compute delta_theta = tanh(angles) * π * dt
            delta_theta = torch.tanh(angles) * math.pi * dt.unsqueeze(-1)  # (1, 8, 16)
            
            # Accumulate theta: theta_t = theta_{t-1} + delta_theta
            theta = self.theta_state + delta_theta  # (1, 8, 16)
            
            # Compute cos and sin (these are the suspected ANE-breakers)
            cos_theta = torch.cos(theta)  # (1, 8, 16)
            sin_theta = torch.sin(theta)  # (1, 8, 16)
            
            # Update theta_state (using Phase 4 mul pattern)
            self.theta_state.mul_(1.0 - self.ema_alpha).add_(theta.detach() * self.ema_alpha)
            
            # ===== PHASE 6C: Full pairwise rotation =====
            if self.phase6c:
                # Apply pairwise RoPE rotation to K
                # Reshape K to pairs: (1, 8, 64) → (1, 8, 32, 2)
                rotary_dim = self.rotary_dim
                k_feats_sq = k_feats.squeeze(1)  # (1, 8, 64)
                k_rot = k_feats_sq[..., :rotary_dim].reshape(1, 8, self.num_angles, 2)  # (1, 8, 16, 2)
                
                k0 = k_rot[..., 0]  # (1, 8, 16)
                k1 = k_rot[..., 1]  # (1, 8, 16)
                
                # Rotation: (k0', k1') = (k0*cos - k1*sin, k0*sin + k1*cos)
                ko0 = k0 * cos_theta - k1 * sin_theta  # (1, 8, 16)
                ko1 = k0 * sin_theta + k1 * cos_theta  # (1, 8, 16)
                
                # Stack and flatten: (1, 8, 16, 2) → (1, 8, 32)
                k_rotated = torch.stack([ko0, ko1], dim=-1).flatten(-2)  # (1, 8, 32)
                
                # Concatenate non-rotated part (if any)
                if rotary_dim < 64:
                    k_rotated = torch.cat([k_rotated, k_feats_sq[..., rotary_dim:]], dim=-1)  # (1, 8, 64)
                
                # Update k_state with rotated K (reshape back to (1, 1, 8, 64) for compatibility)
                k_rotated_expanded = k_rotated.unsqueeze(1)  # (1, 1, 8, 64)
                self.k_state.mul_(1.0 - self.ema_alpha).add_(k_rotated_expanded.detach() * self.ema_alpha)
            else:
                # Phase 6b: Just apply simple gating to K (no full rotation)
                # gate shape: (1, 8, 16), k_feats shape: (1, 1, 8, 64)
                # Apply gate as a scalar per head by averaging over angles
                gate_scalar = gate.mean(dim=-1)  # (1, 8) — average gate over angles
                k_feats_sq = k_feats.squeeze(1)  # (1, 8, 64)
                k_gated = k_feats_sq * gate_scalar.unsqueeze(-1)  # (1, 8, 64)
                self.k_state.mul_(1.0 - self.ema_alpha).add_(k_gated.unsqueeze(1).detach() * self.ema_alpha)
        else:
            # Phase 6a only: Use gate to scale outer product (no cos/sin yet)
            # We'll apply gate scaling in the forward pass when needed
            pass
        
        # Update v_state and angle_state (if present) with Phase 4 mul pattern
        self.v_state.mul_(1.0 - self.ema_alpha).add_(v_feats.detach() * self.ema_alpha)
        
        # Update angle_state for consistency with Phase 3
        angle_feats_flat = self.angle_proj(feats)  # (B, 8*16)
        angle_feats = angle_feats_flat.view(1, 8, 16)  # (1, 8, 16)
        self.angle_state.mul_(1.0 - self.ema_alpha).add_(angle_feats.detach() * self.ema_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor
               - CNN mode: image (B, 3, H, W)
               - MLP mode: vector (B, input_dim)
               - Hybrid mode: image (B, 3, H, W)

        Returns:
            logits: (B, num_classes)

        Side-effect:
            self.feature_state mis à jour in-place (EMA)
            → CoreML lit/écrit automatiquement l'état entre les prédictions
        """
        # Backbone
        if self.backbone_type == "cnn":
            feats = self.backbone(x)                    # (B, last_ch, h, w)
            feats = F.adaptive_avg_pool2d(feats, 1)     # (B, last_ch, 1, 1)
            feats = feats.flatten(1)                    # (B, last_ch)
        elif self.backbone_type in ["mlp", "hybrid", "hybrid1d"]:
            feats = self.backbone(x)                    # already (B, hidden_dim)

        feats = self.proj(feats)                        # (B, feature_dim)

        # ── STATE UPDATE (EMA) ────────────────────────────────────────────
        if self.phase3:
            # Phase 3: Multiple state buffers — each with EMA update derived from features
            # Project features to each state dimension
            angle_feats_flat = self.angle_proj(feats)   # (B, 8*16)
            angle_feats = angle_feats_flat.view(1, 8, 16)  # (1, 8, 16)
            
            k_feats_flat = self.k_proj(feats)           # (B, 1*8*64)
            k_feats = k_feats_flat.view(1, 1, 8, 64)    # (1, 1, 8, 64)
            
            v_feats_flat = self.v_proj(feats)           # (B, 8*64)
            v_feats = v_feats_flat.view(1, 8, 64)       # (1, 8, 64)
            
            # For Phase 6, apply trigonometry & RoPE bisection
            if self.phase6a or self.phase6b or self.phase6c:
                self._update_states_phase6(feats, k_feats, v_feats)
            # For Phase 5, apply Mamba-style outer product fusion
            elif self.phase5:
                self._update_states_phase5(angle_feats, k_feats, v_feats)
            # For Phase 4, apply the specified state write pattern
            elif self.phase4:
                self._update_states_phase4(angle_feats, k_feats, v_feats)
            else:
                # Phase 3 baseline: EMA updates in-place
                self.angle_state.mul_(1.0 - self.ema_alpha).add_(
                    angle_feats.detach() * self.ema_alpha
                )
                self.k_state.mul_(1.0 - self.ema_alpha).add_(
                    k_feats.detach() * self.ema_alpha
                )
                self.v_state.mul_(1.0 - self.ema_alpha).add_(
                    v_feats.detach() * self.ema_alpha
                )
            
            # For fusion, use simple sum of state global norms (since shapes differ)
            state_norm = (
                self.angle_state.norm().detach() +
                self.k_state.norm().detach() +
                self.v_state.norm().detach()
            )
            if self.num_states >= 4:
                state_norm = state_norm + self.dv_state.norm().detach()
            
            # Phase 5: Include ssm_state in fusion
            if self.phase5:
                state_norm = state_norm + self.ssm_state.norm().detach()
            
            # Phase 6: Include theta_state in fusion (if enabled)
            state_count = self.num_states + (1 if self.phase5 else 0)
            if self.phase6b:
                state_norm = state_norm + self.theta_state.norm().detach()
                state_count += 1
            
            # Broadcast state_norm to (1, feature_dim) for fusion
            state_contribution = torch.ones(1, self.feature_dim, device=feats.device) * (state_norm / state_count)
            fused = feats + state_contribution.detach()
        elif self.phase2:
            # Phase 2: 4D state (1, nheads, headdim, d_state) where flatdim = feature_dim
            state_feats_flat = self.state_proj(feats)   # (B, feature_dim)
            state_feats = state_feats_flat.view(1, *self.state_shape[1:])  # (1, nheads, headdim, d_state)
            
            # EMA update in-place on 4D state
            self.feature_state.mul_(1.0 - self.ema_alpha).add_(
                state_feats.detach() * self.ema_alpha
            )
            
            # For classifier, reshape state back to 2D for fusion
            state_global = self.feature_state.reshape(1, -1)  # (1, feature_dim)
            fused = feats + state_global
        else:
            # Phase 0/1: 2D state (1, feature_dim)
            # Mise à jour in-place du buffer — c'est ce que CoreML intercepte
            # pour gérer le state entre les appels à predict()
            self.feature_state.mul_(1.0 - self.ema_alpha).add_(
                feats.detach() * self.ema_alpha
            )
            fused = feats + self.feature_state          # simple addition — pas de grad leak
        # ──────────────────────────────────────────────────────────────────

        # Classification
        out = self.dropout(fused)
        logits = self.classifier(out)
        return logits


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import sys

    print("=== StatefulMobileNet — PyTorch sanity check ===\n")

    backbone_type = sys.argv[1] if len(sys.argv) > 1 else "cnn"
    
    if backbone_type == "mlp":
        model = StatefulMobileNet(
            num_classes=1000,
            backbone_type="mlp",
            input_dim=256,
            ema_alpha=0.1
        )
        input_shape = (1, 256)
        print(f"Backbone: MLP (Phase 1)")
    elif backbone_type == "hybrid":
        model = StatefulMobileNet(
            num_classes=1000,
            backbone_type="hybrid",
            ema_alpha=0.1
        )
        input_shape = (1, 3, 224, 224)
        print(f"Backbone: Hybrid CNN+MLP (Phase 1.5)")
    else:
        model = StatefulMobileNet(
            num_classes=1000,
            width_mult=1.0,
            backbone_type="cnn",
            ema_alpha=0.1
        )
        input_shape = (1, 3, 224, 224)
        print(f"Backbone: CNN (Phase 0)")

    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres: {n_params / 1e6:.2f}M")
    print(f"Feature dim: {model.feature_dim}")
    print(f"EMA alpha: {model.ema_alpha}")
    print(f"State shape: {model.feature_state.shape}\n")

    # Simulate 5 frames consécutives
    print("Simulation 5 frames:")
    for i in range(5):
        x = torch.rand(*input_shape)
        t0 = time.time()
        with torch.no_grad():
            logits = model(x)
        dt = (time.time() - t0) * 1000
        state_norm = model.feature_state.norm().item()
        print(f"  Frame {i+1}: logits {logits.shape}, "
              f"top-1={logits.argmax().item():4d}, "
              f"state_norm={state_norm:.4f}, "
              f"latency={dt:.1f}ms")

    print("\n✓ Forward pass OK — state s'accumule bien entre les frames")