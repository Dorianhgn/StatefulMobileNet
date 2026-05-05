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

---

## Phase 3 : Multiple State Buffers

Test if ANE accepts multiple independent state buffers with different shapes for stateful dispatch.

### Phase 3.1 : 3 States (angle, k, v)

**State definition**:
- `angle_state` (1, 8, 16) = 128 elements
- `k_state` (1, 1, 8, 64) = 512 elements
- `v_state` (1, 8, 64) = 512 elements
- Total: 1152 elements, projected from feature_dim via independent Linear layers

**Projections**:
- `angle_proj`: Linear(feature_dim → 128)
- `k_proj`: Linear(feature_dim → 512)
- `v_proj`: Linear(feature_dim → 512)

**State update**: EMA independent per state: `state = (1-α)·state + α·proj(features).reshape(state_shape)`

**CoreML export fix**: Removed unused `feature_state` buffer from Phase 3 model. Only phase0/1/2 register `feature_state`. Phase 3 exclusively uses (angle/k/v/dv) states. This eliminated CoreML `identity` op error.

### Phase 0 + Phase 3.1 (CNN)

- **Result: CoreML export successful** ✓
- CoreML size: 10.0 MB
- Numerical verification: max|PyTorch - CoreML| = ~1e-4 (FP16 appropriate)

**Device**: iPhone 17 Pro, iOS 26.3.1
- Prediction: 0.45 ms (median)
- Compute Unit Mapping: **140 ops on ANE, 0 on CPU** ✓
- ANE dispatch: 100%

### Phase 1.5 + Phase 3.1 (Hybrid)

- **Result: CoreML export successful** ✓
- CoreML size: 3.0 MB
- Numerical verification: max|PyTorch - CoreML| = 0.000827 (FP16 expected)

**Device**: iPhone 17 Pro, iOS 26.3.1
- Prediction: 0.31 ms (median)
- Compute Unit Mapping: **50 ops on ANE, 0 on CPU** ✓
- ANE dispatch: 100%

---

### Phase 3.2 : 4 States (angle, k, v, dv)

Added fourth state buffer:
- `dv_state` (1, 8, 64) = 512 elements
- `dv_proj`: Linear(feature_dim → 512)

### Phase 0 + Phase 3.2 (CNN)

- **Result: CoreML export successful** ✓
- CoreML size: 11.3 MB

**Device**: iPhone 17 Pro, iOS 26.3.1
- Prediction: 0.47 ms (median)
- Compute Unit Mapping: **152 ops on ANE, 0 on CPU** ✓
- ANE dispatch: 100%

### Phase 1.5 + Phase 3.2 (Hybrid)

- **Result: CoreML export successful** ✓
- CoreML size: 3.5 MB
- Numerical verification: max|PyTorch - CoreML| = 0.000652 (FP16 expected)

**Device**: iPhone 17 Pro, iOS 26.3.1
- Prediction: 0.31 ms (median)
- Compute Unit Mapping: **62 ops on ANE, 0 on CPU** ✓
- ANE dispatch: 100%

---

## Conclusions Phase 3

✅ **Multiple simultaneous states are CoreML-compatible**. 3-state and 4-state configurations both export successfully.

✅ **Architectural fix verified**: Conditional state buffer registration (phase3 exclusive to angle/k/v/dv) eliminates CoreML MIL backend errors.

✅ **Device validation complete** : All Phase 3.1/3.2 exports tested on iPhone 17 Pro Performance Report. ANE dispatch confirmed for all CNN and Hybrid variants.

📊 **Phase 3 device testing results**:

| Phase | Backbone | States | Ops (ANE) | Prediction | ANE % |
|-------|----------|--------|-----------|-----------|-------|
| **0+3.1** | CNN | 3 (a,k,v) | 140 | 0.45 ms | 100% ✓ |
| **1.5+3.1** | Hybrid | 3 (a,k,v) | 50 | 0.31 ms | 100% ✓ |
| **0+3.2** | CNN | 4 (a,k,v,d) | 152 | 0.47 ms | 100% ✓ |
| **1.5+3.2** | Hybrid | 4 (a,k,v,d) | 62 | 0.31 ms | 100% ✓ |

**Key insight**: Adding a 4th state buffer (dv) increases op count but maintains 100% ANE dispatch. Multiple stateful buffers do NOT break ANE scheduling.

**Next**: Phase 4+ roadmap for advanced state patterns (slice assignment, advanced recurrence kernels).