"""
export.py — Exporte StatefulMobileNet vers CoreML 9.0 (.mlpackage)

Features CoreML testées :
  ✓ ct.StateType        → state persistant entre les inférences
  ✓ mlprogram           → format moderne (vs neuralnetwork)
  ✓ minimum_deployment_target=iOS18  → requis pour StateType
  ✓ compute_precision=FLOAT16        → poids en fp16 (ANE-friendly)
  ✓ make_state() + predict(state=…)  → API de prédiction stateful
  ✓ Vérification numérique PyTorch vs CoreML

Usage:
  python export.py
  python export.py --width 0.5 --classes 10 --no-verify
"""

import argparse
import os
import time

import numpy as np
import torch
import coremltools as ct

from model import StatefulMobileNet


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="cnn", choices=["cnn", "mlp", "hybrid"],
                   help="Backbone type: 'cnn' (Phase 0), 'mlp' (Phase 1), 'hybrid' (Phase 1.5)")
    p.add_argument("--width", type=float, default=1.0,
                   help="width_mult MobileNetV2 (0.5 / 0.75 / 1.0) — CNN only")
    p.add_argument("--input-dim", type=int, default=256,
                   help="Input dimension for MLP — MLP only")
    p.add_argument("--classes", type=int, default=1000)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--input-size", type=int, default=224,
                   help="Image size — CNN / Hybrid only")
    p.add_argument("--phase2", action="store_true",
                   help="Enable Phase 2: 4D state (1, 8, 64, 64)")
    p.add_argument("--out-dir", default="./exported_model")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip la vérification numérique")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Determine input shape based on backbone type
    if args.backbone == "mlp":
        shape = (1, args.input_dim)
        input_name = "x_vec"
    else:  # cnn or hybrid
        H = W = args.input_size
        shape = (1, 3, H, W)
        input_name = "x"

    # 1. Instancier et préparer le modèle
    if args.backbone == "mlp":
        phase_label = "Phase 2" if args.phase2 else "Phase 1"
        print(f"\n[1/5] Build StatefulMobileNet {phase_label} (MLP, input_dim={args.input_dim}, classes={args.classes})")
        model = StatefulMobileNet(
            num_classes=args.classes,
            backbone_type="mlp",
            input_dim=args.input_dim,
            ema_alpha=args.ema_alpha,
            phase2=args.phase2,
        )
    elif args.backbone == "hybrid":
        phase_label = "Phase 2" if args.phase2 else "Phase 1.5"
        print(f"\n[1/5] Build StatefulMobileNet {phase_label} (Hybrid CNN+MLP, classes={args.classes})")
        model = StatefulMobileNet(
            num_classes=args.classes,
            backbone_type="hybrid",
            ema_alpha=args.ema_alpha,
            phase2=args.phase2,
        )
    else:
        phase_label = "Phase 2" if args.phase2 else "Phase 0"
        print(f"\n[1/5] Build StatefulMobileNet {phase_label} (CNN, width={args.width}, classes={args.classes})")
        model = StatefulMobileNet(
            num_classes=args.classes,
            backbone_type="cnn",
            width_mult=args.width,
            ema_alpha=args.ema_alpha,
            phase2=args.phase2,
        )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Params: {n_params / 1e6:.2f}M")
    print(f"      Feature dim: {model.feature_dim}")
    print(f"      State shape: {model.feature_state.shape}")

    # 2. Vérification du buffer (doit apparaître dans named_buffers)
    print("\n[2/5] Vérification des buffers TorchScript")
    buffers = dict(model.named_buffers())
    assert "feature_state" in buffers, \
        "ERROR: 'feature_state' absent de named_buffers() !"
    print(f"      named_buffers: {list(buffers.keys())} ✓")

    # 3. Trace TorchScript
    print("\n[3/5] torch.jit.trace")
    example_input = torch.rand(shape)
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_input,))
    print("      Trace OK ✓")

    # 4. Conversion CoreML
    print("\n[4/5] Conversion CoreML 9.0")
    print("      → convert_to=mlprogram")
    print("      → minimum_deployment_target=iOS18 (requis pour StateType)")
    print("      → compute_precision=FLOAT16 (ANE-friendly)")
    print("      → states=[ct.StateType('feature_state')]")

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name=input_name,
                shape=shape,
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32)
        ],
        # ── LE CŒUR DE CoreML 9.0 ──────────────────────────────────────
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=model.feature_state.shape
                ),
                name="feature_state",  # DOIT matcher le nom du register_buffer
            )
        ],
        # ─────────────────────────────────────────────────────────────────
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    dt_convert = time.time() - t0
    print(f"      Conversion OK ✓  ({dt_convert:.1f}s)")

    # Métadonnées
    phase_label = "Phase 2" if args.phase2 else ""
    state_desc = "4D state (1, 8, 64, 64)" if args.phase2 else "2D state (1, feature_dim)"
    
    if args.backbone == "mlp":
        mlmodel.short_description = f"StatefulMobileNet Phase 1{' + Phase 2' if args.phase2 else ''} — MLP + EMA state ({state_desc}) (CoreML 9.0)"
        input_desc = f"Vector input ({args.input_dim}-dim)"
    elif args.backbone == "hybrid":
        mlmodel.short_description = f"StatefulMobileNet Phase 1.5{' + Phase 2' if args.phase2 else ''} — Hybrid CNN+MLP + EMA state ({state_desc}) (CoreML 9.0)"
        input_desc = f"RGB image (1, 3, {args.input_size}, {args.input_size}), float32"
    else:
        mlmodel.short_description = f"StatefulMobileNet Phase 0{' + Phase 2' if args.phase2 else ''} — MobileNetV2 + EMA state ({state_desc}) (CoreML 9.0)"
        input_desc = f"RGB image (1, 3, {args.input_size}, {args.input_size}), float32"
    
    mlmodel.author = "Dorian — test CoreML stateful API"
    mlmodel.version = "1.0"
    mlmodel.input_description[input_name] = input_desc
    mlmodel.output_description["logits"] = f"Class logits ({args.classes} classes)"

    # 5. Sauvegarde
    phase2_suffix = "_Phase2_4Dstate" if args.phase2 else ""
    if args.backbone == "mlp":
        model_name = (
            f"StatefulMobileNet_Phase1_MLP_d{args.input_dim}_c{args.classes}_alpha{args.ema_alpha}{phase2_suffix}"
        )
    elif args.backbone == "hybrid":
        model_name = (
            f"StatefulMobileNet_Phase15_Hybrid_{args.input_size}x{args.input_size}_c{args.classes}_alpha{args.ema_alpha}{phase2_suffix}"
        )
    else:
        model_name = (
            f"StatefulMobileNet_Phase0_w{args.width}_c{args.classes}"
            f"_{args.input_size}x{args.input_size}_alpha{args.ema_alpha}{phase2_suffix}"
        )
    out_path = os.path.join(args.out_dir, f"{model_name}.mlpackage")
    mlmodel.save(out_path)
    print(f"\n[5/5] Sauvegardé → {out_path}")

    # Taille sur disque
    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(out_path)
        for f in files
    ) / 1e6
    print(f"      Taille: {total_size:.1f} MB")

    # ---------------------------------------------------------------------------
    # Vérification numérique PyTorch vs CoreML
    # ---------------------------------------------------------------------------
    if not args.no_verify:
        print("\n" + "─" * 60)
        print("Vérification numérique PyTorch vs CoreML")
        print("─" * 60)

        # Reset states
        model.feature_state.zero_()

        # Créer le state CoreML
        coreml_state = mlmodel.make_state()

        n_frames = 5
        max_diffs = []

        for i in range(n_frames):
            if args.backbone == "mlp":
                x_np = np.random.rand(1, args.input_dim).astype(np.float32)
            else:  # cnn or hybrid
                H = W = args.input_size
                x_np = np.random.rand(1, 3, H, W).astype(np.float32)
            
            x_pt = torch.from_numpy(x_np)

            # PyTorch forward
            with torch.no_grad():
                pt_out = model(x_pt).numpy()

            # CoreML forward (avec state persistant)
            ct_out = mlmodel.predict({input_name: x_np}, state=coreml_state)
            ct_logits = ct_out["logits"]

            max_diff = np.abs(pt_out - ct_logits).max()
            max_diffs.append(max_diff)
            print(f"  Frame {i+1}: max|PyTorch - CoreML| = {max_diff:.6f}")

        mean_diff = np.mean(max_diffs)
        print(f"\n  Diff moyenne: {mean_diff:.6f}")
        if mean_diff < 1e-2:
            print("  ✓ Vérification OK — FP16 precision légèrement dégradée, normal")
        else:
            print("  ⚠ Diff élevée — vérifier la précision ou les ops non supportés")

    # ---------------------------------------------------------------------------
    # Résumé final
    # ---------------------------------------------------------------------------
    print("\n" + "═" * 60)
    print("RÉSUMÉ")
    print("═" * 60)
    print(f"  Modèle    : {model_name}")
    print(f"  Format    : mlprogram (CoreML 9.0)")
    print(f"  State     : feature_state {list(model.feature_state.shape)} — EMA α={args.ema_alpha}")
    print(f"  Deployment: iOS18+ / macOS15+")
    print(f"  Précision : FLOAT16 (weights)")
    print(f"  Taille    : {total_size:.1f} MB")
    print(f"  Output    : {out_path}")
    print()
    print("Prochaines étapes :")
    print("  1. Ouvrir dans Xcode → Performance tab → générer un rapport ANE")
    print("  2. Tester compute_units=CPU_AND_NE pour forcer l'ANE")
    print("  3. Comparer avec ct.target.iOS26 (nouvelles ops CoreML 9.0)")
    print("  4. Ajouter ct.optimize.coreml pour quantification int8")
    print("═" * 60)

    return mlmodel, out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    export(args)