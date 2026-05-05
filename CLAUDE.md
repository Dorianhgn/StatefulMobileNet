# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the `ane_export` conda environment for all Python work here:

```bash
conda activate ane_export
```

Key versions in that env (from `requirements.txt`):
- Python 3.12
- `torch==2.5.1`
- `torchvision==0.20.1`
- `timm>=1.0.0`
- `coremltools==9.0`
- `numpy>=1.26,<2.0`

## Common Commands

```bash
# Sanity check PyTorch forward pass (CNN)
python model.py

# Sanity check MLP backbone (Phase 1)
python model.py mlp

# Export minimal (Phase 0 — CNN, fast)
python export.py --width 0.5 --classes 10

# Export MLP (Phase 1)
python export.py --backbone mlp --input-dim 256 --classes 1000

# Export full MobileNetV2 (Phase 0, 1000 classes)
python export.py --width 1.0 --classes 1000

# Skip numerical verification (faster iteration)
python export.py --no-verify
```

## Architecture

Two modules only:

- **`model.py`** — `StatefulMobileNet(nn.Module)`: CNN (Phase 0) or MLP (Phase 1) backbone, followed by an EMA temporal state stored in `register_buffer("feature_state", ...)`. The buffer is what CoreML maps to `ct.StateType`. State update: `state = (1 - α)·state + α·features` (in-place `mul_` + `add_`).
- **`export.py`** — `torch.jit.trace` → `ct.convert(..., states=[ct.StateType(name="feature_state", ...)])`, targeting `mlprogram` format, `iOS18+`, `FLOAT16` precision. Includes optional numerical verification (PyTorch vs CoreML, 5 frames, tolerance ~1e-2 for FP16).

The `feature_state` buffer name must stay consistent between `register_buffer(name, ...)` and `ct.StateType(name=...)`.

## Research Context

This repo is a **bisection test playground** to identify which CoreML 9.0 / ANE ops break stateful dispatch. The progression is documented in `plan.md`:

- **Phase 0** — CNN backbone, confirmed 100% ANE baseline.
- **Phase 1** — MLP backbone (vector input `(1, d_model)`), tests non-image domain with `ct.StateType`.
- **Phase 2** — 4D state shape `(1, nheads, headdim, d_state)` matching Mamba's `ssm_state`.
- **Phase 3** — Multiple simultaneous state buffers.
- **Phase 4** — Slice-assignment state write pattern (`state[:] = new_val.to(torch.float16)`) vs in-place `mul_`/`add_`.
- **Phase 5** — `bmm`-based state update (SSM recurrence kernel without trig).
- **Phase 6** — `cos`/`sin` ops (suspected ANE breaker on fp16).
- **Phase 7** — Full composition.

Each phase = one export, verified in Xcode → Performance Report → Compute Unit Mapping on a real device. A CPU% spike identifies the failing op.

**Surgical discipline required**: only one variable changes per phase. Keep each phase's `.mlpackage` and Xcode screenshot in `bisect/pN/`.
