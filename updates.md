## Phase 0 : Baseline confirmé

MobileNetV2 w1.0 (3.5M params) → **100% ANE dispatch** ✓

**Device**: iPhone 17 Pro, iOS 26.3.1
- Prediction: 0.41 ms (median)
- Load: 11.73 ms
- Compilation: 43.30 ms
- Compute Unit Mapping: **111 ops on ANE, 0 on CPU** ✓

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

**Device**: iPhone 17 Pro, iOS 26.3.1
- Prediction: 0.28 ms (median)
- Load: 7.45 ms
- Compilation: 19.72 ms
- Compute Unit Mapping: **23 ops on ANE, 0 on CPU** ✓

**Insight**: Bottleneck is NOT model size, but workload TYPE. MLP-only (linear algebra) below ANE scheduling threshold. Mixed CNN+MLP (convolutional + linear) crosses threshold → ANE dispatch.

---

## Phase 2 : 4D State Shape (1, nheads, headdim, d_state)

Test if ANE accepts Mamba-like state shapes for stateful dispatch.

### Phase 0 + Phase 2 (CNN)

- State shape: (1, 8, 20, 8) = 1280 elements
- State update: EMA reshape → 4D → mul_/add_ in-place → reshape back 2D
- **Result: 100% ANE dispatch maintained** ✓

**Device**: iPhone 17 Pro, iOS 26.3.1
- Prediction: 0.41 ms (median) — **same as Phase 0**
- Load: 11.51 ms
- Compilation: 44.33 ms
- Compute Unit Mapping: **111 ops on ANE, 0 on CPU** ✓

**Observation**: Adding 4D state reshaping does NOT break ANE dispatch. Performance identical to Phase 0 baseline.

### Phase 1.5 + Phase 2 (Hybrid)

- State shape: (1, 8, 8, 8) = 512 elements
- **Result: 100% ANE dispatch maintained** ✓
- Prediction: 0.28 ms (median) — **same as Phase 1.5**
- Compute Unit Mapping: **23 ops on ANE, 0 on CPU** ✓

### Phase 1 + Phase 2 (MLP)

- State shape: (1, 8, 8, 8) = 512 elements
- **Result: CPU fallback** ❌ (same as Phase 1 — workload still too small)

---

## Conclusions Phase 2

✅ **4D state shape is ANE-compatible**. CoreML handles reshape ops + state management without performance penalty.

✅ **Phase 2 + Phase 0 = Phase 2 + Phase 1.5 = Phase 0/1.5 perf**: no regression.

📊 **Summary table** (all Phase 2 enabled):

| Phase | Backbone | Ops | Prediction | ANE % | State Shape |
|-------|----------|-----|-----------|-------|-------------|
| **0+2** | CNN | 111 | 0.41 ms | 100% | (1, 8, 20, 8) |
| **1.5+2** | Hybrid | 23 | 0.28 ms | 100% | (1, 8, 8, 8) |
| **1+2** | MLP | ~0 | ? | 0% | (1, 8, 8, 8) |

**Next**: Phase 3 — test multiple state buffers (angle_state, k_state, v_state) to see if ANE schedules >1 simultaneous states. 