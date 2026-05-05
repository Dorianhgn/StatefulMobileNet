# StatefulMobileNet — CoreML 9.0 Test Playground

MobileNetV2-like CNN avec état temporel persistant via l'API `ct.StateType` de CoreML 9.0.

## Structure

```
stateful_mobilenet/
├── model.py          ← Architecture PyTorch (MobileNetV2 + EMA state)
├── export.py         ← Export CoreML 9.0 avec ct.StateType
└── README.md
```

## Quickstart

```bash
# Sanity check PyTorch
python model.py

# Export minimal (rapide)
python export.py --width 0.5 --classes 10

# Export full MobileNetV2 (1000 classes, 224x224)
python export.py --width 1.0 --classes 1000

# Sans vérification numérique (plus rapide)
python export.py --no-verify
```

## Ce que ça teste dans CoreML 9.0

### `ct.StateType` (iOS18+)
```python
states=[
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1, feature_dim), dtype=np.float32),
        name="feature_state",   # doit matcher register_buffer()
    )
]
```
Le state est persistant entre les appels à `predict()` — pas besoin de le passer manuellement.

### Inférence stateful côté Python
```python
state = mlmodel.make_state()
for frame in video_stream:
    out = mlmodel.predict({"x": frame}, state=state)
    # state mis à jour automatiquement (EMA des features)
```

### Inférence stateful côté Swift (iOS/macOS)
```swift
let state = try model.makeState()
let out = try model.prediction(input: input, using: state)
```

## Prochains tests à rajouter

| Feature CoreML 9.0 | Comment l'activer |
|---|---|
| `int8` input/output | `ct.TensorType(dtype=np.int8)` + `minimum_deployment_target=iOS26` |
| `iOS26` target | `minimum_deployment_target=ct.target.iOS26` |
| Quantification int8 weights | `ct.optimize.coreml.linear_quantize_weights(mlmodel, mode="linear_symmetric")` |
| Multiple states | Ajouter plusieurs `register_buffer` + plusieurs `ct.StateType` |
| `AllowLowPrecisionAccumulationOnGPU` | `pass_pipeline` custom |

## Architecture

```
Input (1, 3, 224, 224)
        │
  ┌─────▼──────────────────────┐
  │  Backbone MobileNetV2-like  │
  │  (InvertedResidual blocks)  │
  └─────────────────────────────┘
        │
  Global Avg Pool → (1, 1280)
        │
  Linear proj → (1, feature_dim)
        │
  ┌─────▼──────────────────────┐
  │  EMA State Update           │  ← ct.StateType "feature_state"
  │  state = (1-α)·state + α·f  │     persistant entre les frames
  └─────────────────────────────┘
        │ features + state
  Dropout → Linear → logits (1, num_classes)
```