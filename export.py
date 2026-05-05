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
  python export.py --backbone cnn --phase4 --phase4-pattern slice_assign_with_cast
"""

import argparse
import os
import time
import yaml

import numpy as np
import torch
import coremltools as ct

from model import StatefulMobileNet


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "phase4_config.yaml") -> dict:
    """Load Phase 4 config from YAML if it exists, otherwise return empty dict."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


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
    p.add_argument("--phase3", action="store_true",
                   help="Enable Phase 3: Multiple state buffers")
    p.add_argument("--num-states", type=int, default=1,
                   help="Number of states for Phase 3 (3 or 4)")
    p.add_argument("--phase4", action="store_true",
                   help="Enable Phase 4: Test different state write patterns")
    p.add_argument("--phase4-pattern", type=str,
                   default="slice_assign_with_cast",
                   choices=[
                       "addition",
                       "mul",
                       "copy",
                       "clone",
                       "slice_assign_with_cast",
                       "slice_assign_no_cast",
                   ],
                   help="State write pattern for Phase 4")
    p.add_argument("--phase5", action="store_true",
                   help="Enable Phase 5: Mamba-style outer product state fusion")
    p.add_argument("--phase5-pattern", type=str,
                   default="matmul",
                   choices=["einsum", "matmul"],
                   help="Outer product pattern for Phase 5 (einsum or matmul-based)")
    p.add_argument("--phase6a", action="store_true",
                   help="Enable Phase 6a: tanh + softplus gating only")
    p.add_argument("--phase6b", action="store_true",
                   help="Enable Phase 6b: + cos/sin accumulation (implies 6a)")
    p.add_argument("--phase6c", action="store_true",
                   help="Enable Phase 6c: + full pairwise rotation (implies 6a, 6b)")
    p.add_argument("--out-dir", default="./exported_model")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip la vérification numérique")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load Phase 4 config if it exists, CLI args override config values
    config = load_config("phase4_config.yaml")
    phase4_cfg = config.get("phase4", {})
    
    # Apply config values (but CLI args take precedence)
    if args.phase6a or args.phase6b or args.phase6c:
        # When any phase6 is passed via CLI, enable it (auto-enables all previous: 2, 3, 4, 5)
        args.phase2 = True
        args.phase3 = True
        args.phase4 = True
        args.phase5 = True
        args.num_states = 3
        args.phase4_pattern = "mul"  # Phase 6 uses optimal Phase 4 pattern
        args.phase5_pattern = "matmul"  # Phase 6 uses optimal Phase 5 pattern
    elif args.phase5:
        # When phase5 is passed via CLI, enable it (auto-enables 2, 3, 4)
        args.phase2 = True
        args.phase3 = True
        args.phase4 = True
        args.num_states = 3
        args.phase4_pattern = "mul"  # Phase 5 uses optimal Phase 4 pattern
    elif args.phase4:
        # When phase4 is passed via CLI, enable it
        phase4_enabled = True
        phase4_pattern = args.phase4_pattern
        # Auto-enable Phase 2 and Phase 3
        args.phase2 = True
        args.phase3 = True
        args.num_states = 3
    elif phase4_cfg.get("enabled", False):
        # Config file specifies phase4 enabled
        args.phase4 = True
        phase4_pattern = phase4_cfg.get("state_write_pattern", "slice_assign_with_cast")
        args.phase4_pattern = phase4_pattern
        # Auto-enable Phase 2 and Phase 3
        phase3_cfg = phase4_cfg.get("phases_enabled", {})
        args.phase2 = phase3_cfg.get("phase2", True)
        args.phase3 = phase3_cfg.get("phase3", True)
        args.num_states = phase3_cfg.get("num_states", 3)
    else:
        args.phase4 = False

    # Determine input shape based on backbone type
    if args.backbone == "mlp":
        shape = (1, args.input_dim)
        input_name = "x_vec"
    else:  # cnn or hybrid
        H = W = args.input_size
        shape = (1, 3, H, W)
        input_name = "x"

    # 1. Instancier et préparer le modèle
    phase_suffix = ""
    phase4_suffix = ""
    phase5_suffix = ""
    phase6_suffix = ""
    if args.phase6a or args.phase6b or args.phase6c:
        if args.phase6c:
            phase6_suffix = " + Phase 6c (RoPE rotation)"
        elif args.phase6b:
            phase6_suffix = " + Phase 6b (cos/sin)"
        else:
            phase6_suffix = " + Phase 6a (tanh/softplus)"
    elif args.phase5:
        phase5_suffix = f" + Phase 5 ({args.phase5_pattern})"
    elif args.phase4:
        phase4_suffix = f" + Phase 4 ({args.phase4_pattern})"
    if args.phase3:
        phase_suffix = f" + Phase 3.{1 if args.num_states == 3 else 2} ({args.num_states} states)"
    elif args.phase2:
        phase_suffix = " + Phase 2"
    
    if args.backbone == "mlp":
        phase_label = f"Phase 1{phase_suffix}{phase4_suffix}{phase5_suffix}{phase6_suffix}"
        print(f"\n[1/5] Build StatefulMobileNet {phase_label} (MLP, input_dim={args.input_dim}, classes={args.classes})")
        model = StatefulMobileNet(
            num_classes=args.classes,
            backbone_type="mlp",
            input_dim=args.input_dim,
            ema_alpha=args.ema_alpha,
            phase2=args.phase2,
            phase3=args.phase3,
            num_states=args.num_states,
            phase4=args.phase4,
            phase4_pattern=args.phase4_pattern,
            phase5=args.phase5,
            phase5_pattern=args.phase5_pattern,
            phase6a=args.phase6a,
            phase6b=args.phase6b,
            phase6c=args.phase6c,
        )
    elif args.backbone == "hybrid":
        phase_label = f"Phase 1.5{phase_suffix}{phase4_suffix}{phase5_suffix}{phase6_suffix}"
        print(f"\n[1/5] Build StatefulMobileNet {phase_label} (Hybrid CNN+MLP, classes={args.classes})")
        model = StatefulMobileNet(
            num_classes=args.classes,
            backbone_type="hybrid",
            ema_alpha=args.ema_alpha,
            phase2=args.phase2,
            phase3=args.phase3,
            num_states=args.num_states,
            phase4=args.phase4,
            phase4_pattern=args.phase4_pattern,
            phase5=args.phase5,
            phase5_pattern=args.phase5_pattern,
            phase6a=args.phase6a,
            phase6b=args.phase6b,
            phase6c=args.phase6c,
        )
    else:
        phase_label = f"Phase 0{phase_suffix}{phase4_suffix}{phase5_suffix}{phase6_suffix}"
        print(f"\n[1/5] Build StatefulMobileNet {phase_label} (CNN, width={args.width}, classes={args.classes})")
        model = StatefulMobileNet(
            num_classes=args.classes,
            backbone_type="cnn",
            width_mult=args.width,
            ema_alpha=args.ema_alpha,
            phase2=args.phase2,
            phase3=args.phase3,
            num_states=args.num_states,
            phase4=args.phase4,
            phase4_pattern=args.phase4_pattern,
            phase5=args.phase5,
            phase5_pattern=args.phase5_pattern,
            phase6a=args.phase6a,
            phase6b=args.phase6b,
            phase6c=args.phase6c,
        )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Params: {n_params / 1e6:.2f}M")
    print(f"      Feature dim: {model.feature_dim}")
    
    if args.phase3:
        print(f"      Phase 3 states: angle_state {model.angle_state.shape}, k_state {model.k_state.shape}, v_state {model.v_state.shape}", end="")
        if args.num_states >= 4:
            print(f", dv_state {model.dv_state.shape}", end="")
        if args.phase5:
            print(f", ssm_state {model.ssm_state.shape}", end="")
        if args.phase6b:
            print(f", theta_state {model.theta_state.shape}", end="")
        print()
    else:
        print(f"      State shape: {model.feature_state.shape}")

    # 2. Vérification du buffer (doit apparaître dans named_buffers)
    print("\n[2/5] Vérification des buffers TorchScript")
    buffers = dict(model.named_buffers())
    
    if args.phase3:
        required_buffers = ["angle_state", "k_state", "v_state"]
        if args.num_states >= 4:
            required_buffers.append("dv_state")
        if args.phase5:
            required_buffers.extend(["a_coeff", "b_coeff", "g_coeff", "ssm_state"])
        if args.phase6b:
            required_buffers.append("theta_state")
        for buf in required_buffers:
            assert buf in buffers, f"ERROR: '{buf}' absent de named_buffers() !"
    else:
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
    
    # Build states list
    states_list = []
    
    if args.phase3:
        # Phase 3: Use only angle, k, v (and dv if num_states >= 4)
        states_list.append(
            ct.StateType(
                wrapped_type=ct.TensorType(shape=model.angle_state.shape),
                name="angle_state",
            )
        )
        states_list.append(
            ct.StateType(
                wrapped_type=ct.TensorType(shape=model.k_state.shape),
                name="k_state",
            )
        )
        states_list.append(
            ct.StateType(
                wrapped_type=ct.TensorType(shape=model.v_state.shape),
                name="v_state",
            )
        )
        if args.num_states >= 4:
            states_list.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=model.dv_state.shape),
                    name="dv_state",
                )
            )
        # Phase 5: Add ssm_state
        if args.phase5:
            states_list.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=model.ssm_state.shape),
                    name="ssm_state",
                )
            )
        # Phase 6: Add theta_state (if phase6b enabled)
        if args.phase6b:
            states_list.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=model.theta_state.shape),
                    name="theta_state",
                )
            )
        print(f"      → states=[ct.StateType('angle_state'), ct.StateType('k_state'), ct.StateType('v_state')", end="")
        if args.num_states >= 4:
            print(", ct.StateType('dv_state')", end="")
        if args.phase5:
            print(", ct.StateType('ssm_state')", end="")
        if args.phase6b:
            print(", ct.StateType('theta_state')]")
        else:
            print("]")
    else:
        # Phase 0/1/2: Use feature_state
        states_list.append(
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=model.feature_state.shape
                ),
                name="feature_state",  # DOIT matcher le nom du register_buffer
            )
        )
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
        states=states_list,
        # ─────────────────────────────────────────────────────────────────
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    dt_convert = time.time() - t0
    print(f"      Conversion OK ✓  ({dt_convert:.1f}s)")

    # Métadonnées
    phase_desc = ""
    if args.phase3:
        phase_desc = f" + Phase 3.{1 if args.num_states == 3 else 2} ({args.num_states} states)"
    elif args.phase2:
        phase_desc = " + Phase 2"
    
    state_desc = ""
    if args.phase3:
        state_desc = f"Multiple states: angle (1,8,16), k (1,1,8,64), v (1,8,64)" + (", dv (1,8,64)" if args.num_states >= 4 else "")
    elif args.phase2:
        state_desc = "4D state (1, 8, 64, 64)"
    else:
        state_desc = "2D state (1, feature_dim)"
    
    if args.backbone == "mlp":
        mlmodel.short_description = f"StatefulMobileNet Phase 1{phase_desc} — MLP + EMA state ({state_desc}) (CoreML 9.0)"
        input_desc = f"Vector input ({args.input_dim}-dim)"
    elif args.backbone == "hybrid":
        mlmodel.short_description = f"StatefulMobileNet Phase 1.5{phase_desc} — Hybrid CNN+MLP + EMA state ({state_desc}) (CoreML 9.0)"
        input_desc = f"RGB image (1, 3, {args.input_size}, {args.input_size}), float32"
    else:
        mlmodel.short_description = f"StatefulMobileNet Phase 0{phase_desc} — MobileNetV2 + EMA state ({state_desc}) (CoreML 9.0)"
        input_desc = f"RGB image (1, 3, {args.input_size}, {args.input_size}), float32"
    
    mlmodel.author = "Dorian — test CoreML stateful API"
    mlmodel.version = "1.0"
    mlmodel.input_description[input_name] = input_desc
    mlmodel.output_description["logits"] = f"Class logits ({args.classes} classes)"

    # 5. Sauvegarde
    phase_suffix = ""
    if args.phase3:
        phase_suffix = f"_Phase31_{args.num_states}states" if args.num_states == 3 else f"_Phase32_{args.num_states}states"
    elif args.phase2:
        phase_suffix = "_Phase2_4Dstate"
    
    # Phase 4 naming: append pattern name
    phase4_suffix = ""
    if args.phase4:
        phase4_suffix = f"_Phase4_{args.phase4_pattern}"
    
    if args.backbone == "mlp":
        model_name = (
            f"StatefulMobileNet_Phase1_MLP_d{args.input_dim}_c{args.classes}_alpha{args.ema_alpha}{phase_suffix}{phase4_suffix}"
        )
    elif args.backbone == "hybrid":
        model_name = (
            f"StatefulMobileNet_Phase15_Hybrid_{args.input_size}x{args.input_size}_c{args.classes}_alpha{args.ema_alpha}{phase_suffix}{phase4_suffix}"
        )
    else:
        model_name = (
            f"StatefulMobileNet_Phase0_w{args.width}_c{args.classes}"
            f"_{args.input_size}x{args.input_size}_alpha{args.ema_alpha}{phase_suffix}{phase4_suffix}"
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
        if args.phase3:
            model.angle_state.zero_()
            model.k_state.zero_()
            model.v_state.zero_()
            if args.num_states >= 4:
                model.dv_state.zero_()
        else:
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
    
    if args.phase3:
        state_summary = f"angle_state {list(model.angle_state.shape)}, k_state {list(model.k_state.shape)}, v_state {list(model.v_state.shape)}"
        if args.num_states >= 4:
            state_summary += f", dv_state {list(model.dv_state.shape)}"
        print(f"  States    : {state_summary} — EMA α={args.ema_alpha}")
    else:
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