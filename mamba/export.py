"""
mamba/export.py — Export StatefulMambaHybrid1D to CoreML 9.0 (.mlpackage)

Features:
  ✓ Hybrid1D backbone (Conv1D + Linear)
  ✓ MambaBlock state buffers (angle_state, k_state, v_state)
  ✓ ct.StateType for persistent state across inferences
  ✓ mlprogram format, iOS18+, FLOAT16 precision
  ✓ Stateful API: make_state() + predict(state=…)
  ✓ Optional numerical verification PyTorch vs CoreML

Usage:
  python export.py
  python export.py --classes 10 --seq-length 224
  python export.py --ema-alpha 0.05 --no-verify
"""

import argparse
import os
import time

import numpy as np
import torch
import coremltools as ct

from model import StatefulMambaHybrid1D


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Export StatefulMambaHybrid1D to CoreML")
    
    # Model architecture
    p.add_argument("--num-classes", type=int, default=1000,
                   help="Number of output classes")
    p.add_argument("--seq-length", type=int, default=224,
                   help="Sequence length for Conv1D input")
    p.add_argument("--backbone-hidden-dim", type=int, default=256,
                   help="Hidden dimension in Hybrid1D backbone")
    p.add_argument("--backbone-output-dim", type=int, default=512,
                   help="Output dimension from Hybrid1D backbone")
    
    # Mamba block
    p.add_argument("--mamba-d-state", type=int, default=64,
                   help="SSM state dimension")
    p.add_argument("--mamba-headdim", type=int, default=64,
                   help="Head dimension")
    p.add_argument("--mamba-num-heads", type=int, default=8,
                   help="Number of heads")
    
    # Training/export
    p.add_argument("--ema-alpha", type=float, default=0.1,
                   help="EMA alpha coefficient")
    p.add_argument("--out-dir", default="./exported_model",
                   help="Output directory for .mlpackage")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip numerical verification PyTorch vs CoreML")
    
    return p.parse_args()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 70)
    print("StatefulMambaHybrid1D → CoreML 9.0 Export")
    print("=" * 70)
    
    # [1/5] Build model
    print(f"\n[1/5] Build StatefulMambaHybrid1D")
    print(f"      Backbone: Hybrid1D Conv1D + Linear")
    print(f"      Mamba block: {args.mamba_num_heads} heads, d_state={args.mamba_d_state}")
    print(f"      Classes: {args.num_classes}, Seq length: {args.seq_length}")
    
    model = StatefulMambaHybrid1D(
        num_classes=args.num_classes,
        backbone_in_channels=3,
        backbone_hidden_dim=args.backbone_hidden_dim,
        backbone_output_dim=args.backbone_output_dim,
        seq_length=args.seq_length,
        ema_alpha=args.ema_alpha,
        mamba_d_state=args.mamba_d_state,
        mamba_headdim=args.mamba_headdim,
        mamba_num_heads=args.mamba_num_heads,
    )
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {n_params / 1e6:.2f}M")
    print(f"      Feature dim: {model.feature_dim}")
    print(f"      EMA alpha: {model.ema_alpha}")
    
    # [2/5] Check buffers
    print(f"\n[2/5] Verify state buffers (TorchScript)")
    buffers = dict(model.named_buffers())
    required_buffers = ["mamba.angle_state", "mamba.k_state", "mamba.v_state", "mamba.ssm_state"]
    
    for buf in required_buffers:
        assert buf in buffers, f"ERROR: '{buf}' not in named_buffers()!"
    
    print(f"      Buffers registered: ✓")
    print(f"        - mamba.angle_state: {model.mamba.angle_state.shape}")
    print(f"        - mamba.k_state: {model.mamba.k_state.shape}")
    print(f"        - mamba.v_state: {model.mamba.v_state.shape}")
    print(f"        - mamba.ssm_state: {model.mamba.ssm_state.shape}")
    # [3/5] TorchScript trace
    print(f"\n[3/5] torch.jit.trace")
    shape = (1, 3, args.seq_length)
    example_input = torch.rand(shape)
    
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_input,))
    print(f"      Trace OK ✓")
    
    # [4/5] CoreML conversion
    print(f"\n[4/5] CoreML 9.0 Conversion")
    print(f"      → convert_to=mlprogram")
    print(f"      → minimum_deployment_target=iOS18 (required for StateType)")
    print(f"      → compute_precision=FLOAT16 (ANE-friendly)")
    
    # Define state types
    states_list = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=model.mamba.angle_state.shape),
            name="mamba.angle_state",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=model.mamba.k_state.shape),
            name="mamba.k_state",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=model.mamba.v_state.shape),
            name="mamba.v_state",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=model.mamba.ssm_state.shape),
            name="mamba.ssm_state",
        ),
    ]
    
    print(f"      → states=[StateType('mamba.angle_state'), StateType('mamba.k_state'), StateType('mamba.v_state')]")
    
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="x",
                shape=shape,
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32)
        ],
        states=states_list,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    dt_convert = time.time() - t0
    print(f"      Conversion OK ✓  ({dt_convert:.1f}s)")
    
    # [5/5] Save and metadata
    print(f"\n[5/5] Save model")
    
    mlmodel.short_description = (
        f"StatefulMambaHybrid1D — Hybrid1D Conv1D+Linear + Mamba block + EMA state "
        f"(angle, k, v) (CoreML 9.0)"
    )
    mlmodel.author = "Dorian — Mamba + Hybrid1D ANE test"
    mlmodel.version = "1.0"
    mlmodel.input_description["x"] = f"1D sequence (1, 3, {args.seq_length}), float32"
    mlmodel.output_description["logits"] = f"Class logits ({args.num_classes} classes)"
    
    model_name = (
        f"StatefulMambaHybrid1D_seq{args.seq_length}_c{args.num_classes}"
        f"_alpha{args.ema_alpha}"
    )
    out_path = os.path.join(args.out_dir, f"{model_name}.mlpackage")
    mlmodel.save(out_path)
    
    print(f"      Saved → {out_path}")
    
    # Size on disk
    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(out_path)
        for f in files
    ) / 1e6
    print(f"      Size: {total_size:.1f} MB")
    
    # ---------------------------------------------------------------------------
    # Numerical verification (optional)
    # ---------------------------------------------------------------------------
    if not args.no_verify:
        print("\n" + "─" * 70)
        print("Numerical Verification: PyTorch vs CoreML")
        print("─" * 70)
        
        # Reset states
        model.mamba.angle_state.zero_()
        model.mamba.k_state.zero_()
        model.mamba.v_state.zero_()
        model.mamba.ssm_state.zero_()
        
        # Create CoreML state
        coreml_state = mlmodel.make_state()
        
        n_frames = 5
        print(f"\nRunning {n_frames} frames with random inputs...\n")
        
        for i in range(n_frames):
            x_np = np.random.rand(1, 3, args.seq_length).astype(np.float32)
            x_pt = torch.from_numpy(x_np)
            
            # PyTorch inference
            with torch.no_grad():
                logits_pt = model(x_pt).numpy()
            
            # CoreML inference
            logits_coreml = mlmodel.predict(
                {"x": x_np},
                state=coreml_state
            )["logits"]
            
            # Compute difference
            diff = np.abs(logits_pt - logits_coreml).max()
            print(f"  Frame {i}: max diff = {diff:.2e}")
    
    print("\n" + "=" * 70)
    print("✓ Export complete!")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    export(args)
