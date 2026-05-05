import argparse
import os
import time

import numpy as np
import torch
import coremltools as ct

from model import StatefulMobileNet


def parse_args():
    p = argparse.ArgumentParser(description="Export tiny StatefulMobileNet to CoreML 9.0")
    p.add_argument("--width", type=float, default=0.25,
                   help="width_mult MobileNetV2 (0.25 = tiny)")
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--input-size", type=int, default=64)
    p.add_argument("--out-dir", default="./exported_model")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip numerical verification")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    H = W = args.input_size
    shape = (1, 3, H, W)

    # Build model
    print(f"\n[1/4] Build tiny StatefulMobileNet (width={args.width}, classes={args.classes})")
    model = StatefulMobileNet(
        num_classes=args.classes,
        width_mult=args.width,
        ema_alpha=args.ema_alpha,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Params: {n_params / 1e6:.2f}M")
    print(f"      Feature dim: {model.feature_dim}")
    print(f"      State shape: {model.feature_state.shape}")

    # Trace TorchScript
    print("\n[2/4] torch.jit.trace")
    example_input = torch.rand(shape)
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_input,))
    print("      Trace OK ✓")

    # Convert to CoreML 9.0
    print("\n[3/4] Conversion CoreML 9.0")
    print("      → convert_to=mlprogram")
    print("      → minimum_deployment_target=iOS18 (StateType support)")
    print("      → compute_precision=FLOAT16 (ANE-friendly)")
    print("      → states=[ct.StateType('feature_state')]")

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
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, model.feature_dim)
                ),
                name="feature_state",
            )
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    dt_convert = time.time() - t0
    print(f"      Conversion OK ✓  ({dt_convert:.1f}s)")

    # Metadata
    mlmodel.short_description = "Tiny StatefulMobileNet — CoreML 9.0 with persistent state"
    mlmodel.author = "Test ANE"
    mlmodel.version = "1.0"
    mlmodel.input_description["x"] = f"RGB image (1, 3, {H}, {W}), float32"
    mlmodel.output_description["logits"] = f"Class logits ({args.classes} classes)"

    # Save
    model_name = (
        f"TinyStatefulMobileNet_w{args.width}_c{args.classes}"
        f"_{H}x{W}_alpha{args.ema_alpha}"
    )
    out_path = os.path.join(args.out_dir, f"{model_name}.mlpackage")
    mlmodel.save(out_path)
    print(f"\n[4/4] Sauvegardé → {out_path}")

    # File size
    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(out_path)
        for f in files
    ) / 1e6
    print(f"      Taille: {total_size:.1f} MB")

    # Numerical verification
    if not args.no_verify:
        print("\n" + "─" * 60)
        print("Vérification numérique PyTorch vs CoreML")
        print("─" * 60)

        model.feature_state.zero_()
        coreml_state = mlmodel.make_state()

        n_frames = 3
        max_diffs = []

        for i in range(n_frames):
            x_np = np.random.rand(1, 3, H, W).astype(np.float32)
            x_pt = torch.from_numpy(x_np)

            with torch.no_grad():
                pt_out = model(x_pt).numpy()

            ct_out = mlmodel.predict({"x": x_np}, state=coreml_state)
            ct_logits = ct_out["logits"]

            max_diff = np.abs(pt_out - ct_logits).max()
            max_diffs.append(max_diff)
            print(f"  Frame {i+1}: max|PyTorch - CoreML| = {max_diff:.6f}")

        mean_diff = np.mean(max_diffs)
        print(f"\n  Mean diff: {mean_diff:.6f}")
        if mean_diff < 1e-2:
            print("  ✓ OK — FP16 precision expected")
        else:
            print("  ⚠ Check precision or unsupported ops")

    print("\n" + "═" * 60)
    print(f"Output: {out_path}")
    print("═" * 60)


if __name__ == "__main__":
    main()