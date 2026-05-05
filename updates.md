## Phase 0 : ok, it works

MobileNetV2 w1.0 (3.5M params) → **100% ANE dispatch** ✓

---

## Phase 1 : MLP (MLP-only backbone)

- Input: (1, 256) vectoriel
- Params: 1.17M
- CoreML size: 2.3 MB
- **Result: Workload too small → CPU fallback** ❌

Hypothesis: Pure MLP with only ~18 linear ops insufficient to justify ANE scheduling. CoreML decides it's cheaper to run on CPU.

---

## Phase 1.5 : Hybrid (CNN 2-layer + MLP)

- Backbone: 2 Conv2d (5x5 stride 2) + BatchNorm + ReLU6 → GAP → 2 Linear layers
- Input: (1, 3, 224, 224) image
- Params: 0.89M (smaller than Phase 0!)
- CoreML size: 1.8 MB
- **Result: 100% ANE dispatch** ✓

Device: iPhone 17 Pro, iOS 26.3.1
- Prediction: 0.28 ms
- Load: 8.00 ms
- Compute Unit Mapping: **Neural Engine: 21 ops, CPU: 0, GPU: 0**

**Insight**: Bottleneck is NOT model size, but workload TYPE. MLP-only (linear algebra) below ANE scheduling threshold. Mixed CNN+MLP (convolutional + linear) crosses threshold → ANE dispatch.

**Next**: Phase 2 — test 4D state shape `(1, nheads, headdim, d_state)` to validate ANE compatibility with Mamba SSM state shapes. 