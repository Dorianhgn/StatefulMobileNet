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
        backbone_type: "cnn" (Phase 0), "mlp" (Phase 1), "hybrid" (Phase 1.5)
        input_dim:   pour MLP, dimension d'entrée vectorielle
    """

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        ema_alpha: float = 0.1,
        feature_dim: int | None = None,
        backbone_type: str = "cnn",
        input_dim: int = 256,
    ):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.backbone_type = backbone_type
        self.input_dim = input_dim

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
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # Projection vers feature_dim si spécifié
        self.feature_dim = feature_dim or last_ch
        if self.feature_dim != last_ch:
            self.proj = nn.Linear(last_ch, self.feature_dim)
        else:
            self.proj = nn.Identity()

        # Classifier
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # ── STATE ──────────────────────────────────────────────────────────
        # register_buffer → persistant dans le modèle, traceable par TorchScript
        # Nom "feature_state" = celui qu'on passera à ct.StateType(name=...)
        self.register_buffer(
            "feature_state",
            torch.zeros(1, self.feature_dim, dtype=torch.float32),
        )
        # ──────────────────────────────────────────────────────────────────

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
        elif self.backbone_type in ["mlp", "hybrid"]:
            feats = self.backbone(x)                    # already (B, hidden_dim)

        feats = self.proj(feats)                        # (B, feature_dim)

        # ── STATE UPDATE (EMA) ────────────────────────────────────────────
        # Mise à jour in-place du buffer — c'est ce que CoreML intercepte
        # pour gérer le state entre les appels à predict()
        self.feature_state.mul_(1.0 - self.ema_alpha).add_(
            feats.detach() * self.ema_alpha
        )
        # ──────────────────────────────────────────────────────────────────

        # Fusion features courantes + état accumulé
        fused = feats + self.feature_state          # simple addition — pas de grad leak

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