# ANE Testing Options: `einops.rearrange` & `torch.flip`

## Overview

Two new export options have been added to test whether `einops.rearrange` and `torch.flip` operations can pass through to the Apple Neural Engine (ANE) after CoreML export.

## New Command-Line Arguments

### `--use-rearrange`
Uses `einops.rearrange` for feature transformation instead of `.view()` and `.unsqueeze()`.

**When enabled:**
- Features (1, 512) are reshaped to 4D: (1, 512, 1, 1)
- `rearrange` is used for all transformations
- Reshaped back to (1, 512) before passing to Mamba block

**Export:**
```bash
python export.py --use-rearrange
```

Output model name: `StatefulMambaHybrid1D_..._rearrange.mlpackage`

### `--use-flip`
Applies `torch.flip` to reverse a spatial dimension of the features.

**When enabled:**
- Features reshaped to 4D: (1, 512, 1, 1)
- `torch.flip` applied on dim=3 (W dimension)
- Reshaped back to (1, 512) before passing to Mamba block

**Export:**
```bash
python export.py --use-flip
```

Output model name: `StatefulMambaHybrid1D_..._flip.mlpackage`

## Usage Examples

### Base Mode (No Transformations)
```bash
python export.py --num-classes 1000 --seq-length 224
```
Output: `StatefulMambaHybrid1D_seq224_c1000_alpha0.1.mlpackage`

### With Rearrange
```bash
python export.py --num-classes 1000 --seq-length 224 --use-rearrange
```
Output: `StatefulMambaHybrid1D_seq224_c1000_alpha0.1_rearrange.mlpackage`

### With Flip
```bash
python export.py --num-classes 1000 --seq-length 224 --use-flip
```
Output: `StatefulMambaHybrid1D_seq224_c1000_alpha0.1_flip.mlpackage`

### With Both
```bash
python export.py --num-classes 1000 --seq-length 224 --use-rearrange --use-flip
```
Output: `StatefulMambaHybrid1D_seq224_c1000_alpha0.1_rearrange_flip.mlpackage`

## Implementation Details

### Feature Transformation Pipeline

```
Backbone Features (1, 512)
        ↓
    [Optional: Apply transformations]
        ├─ If use_rearrange: rearrange((1,512) → (1,512,1,1) → (1,512))
        └─ If use_flip: flip on spatial dimension
        ↓
    Mamba Block Input (1, 512)
```

### Transformation Flow

1. **Base (no options)**
   ```python
   features_4d = features.view(1, 512, 1, 1)
   # ... (flip if enabled)
   features = features_4d.view(1, 512)
   ```

2. **With rearrange**
   ```python
   features_4d = rearrange(features, 'b d -> b d 1 1')  # (1,512) → (1,512,1,1)
   if use_flip:
       features_4d = torch.flip(features_4d, dims=[3])
   features = rearrange(features_4d, 'b d h w -> b (d h w)')  # → (1, 512)
   ```

## Testing on ANE

After export, these models can be tested on a real iOS device:

1. Open the `.mlpackage` in Xcode
2. Build & run on device
3. Check Performance Report → Compute Unit Mapping
4. Look for CPU% spike to identify if rearrange/flip operations break ANE dispatch

### Expected Observations

- **Base model**: Should see high ANE% (ideally 95%+)
- **With rearrange**: Check if ANE% remains high or drops (indicates ANE compatibility)
- **With flip**: Same diagnostic approach
- **With both**: Tests combined operation compatibility

## Future: STSS (Spatial-Temporal Selective Scan)

These options prepare the ground for implementing multi-directional Mamba scans:
- Row forward/reverse scans using rearrange + flip
- Column forward/reverse scans using rearrange + flip
- Independent Mamba processing for each direction
- Aggregation of outputs

The diagnostic models help identify which operations are ANE-safe before building the full STSS module.
