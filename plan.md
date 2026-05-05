Excellent résultat. Donc on tient enfin une carte propre : ton env est sain, l'API stateful iOS18 dispatch correctement sur ANE quand le modèle a la bonne forme. Tu peux maintenant faire de la bissection scientifique. Ta proposition est exactement la bonne méthodo. Voici la feuille de route.

**Principe directeur :** une seule variable change par phase. Tu gardes le scaffolding StatefulMobileNet (image input, classifier output) jusqu'à ce qu'il devienne incompatible avec ce que tu testes — à ce moment-là tu changes le scaffolding aussi mais tu repars d'un modèle minimal qui dispatch ANE dans le nouveau domaine. Tu vérifies à chaque phase dans Xcode Performance Report que tout reste sur ANE. Première phase qui casse → tu as ton coupable.

**Phase 0 — Baseline confirmé.** StatefulMobileNet tel quel, 100% ANE. C'est ton ancre. Garde le `.mlpackage` de référence. Tu reviens dessus à chaque test pour vérifier que tu n'as pas pété l'env.

**Phase 1 — Sortir du domaine image.** Remplace le backbone CNN par un MLP pur : entrée `(1, d_model)` au lieu de `(1, 3, 224, 224)`, deux ou trois `nn.Linear` avec SiLU entre, et la même mécanique d'état EMA `(1, feature_dim)`. C'est le test "Linear/MLP en domaine vectoriel passe-t-il sur ANE avec stateful ?". Mamba est essentiellement un MLP+récurrence sur des tenseurs non-image, donc cette phase doit valider que ce domaine est viable. Si ça casse ici, on a un signal énorme : il faut tout reformuler en `(B, C, 1, S)` à la ml-ane-transformers et utiliser des Conv2d 1x1. Si ça passe, on continue.

**Phase 2 — Complexifier la forme de l'état.** Garde le MLP de la phase 1 mais transforme l'état d'une simple `(1, feature_dim)` en `(1, nheads, headdim, d_state)` = `(1, 8, 64, 64)`, soit la forme exacte du `ssm_state` de Mamba. Garde une mise à jour EMA bête (un scalaire de decay), pas encore de récurrence complexe. Cela isole *uniquement* la question : ANE accepte-t-il un état 4D de cette forme ? Les exemples KV-cache d'Apple sont en `(B, H, S, D)`, donc *a priori* oui, mais il vaut mieux le vérifier explicitement.

**Phase 3 — Multiplier les états.** Ajoute un deuxième, puis un troisième, puis un quatrième buffer registered, avec des shapes différentes correspondant à `angle_state` `(1, 8, 16)`, `k_state` `(1, 1, 8, 64)`, `v_state` `(1, 8, 64)`. EMA simple sur chacun. Test : ANE dispatch-il avec 4 states simultanés ? Les exemples Apple en montrent souvent 1 ou 2, jamais 4 à ma connaissance.

**Phase 4 — Pattern d'écriture d'état.** C'est *peut-être* le coupable principal. StatefulMobileNet utilise `state.mul_(...).add_(...)` (in-place via méthodes). Le `StepWrapper` de Mamba utilise `self.ssm_state[:] = new_ssm.to(torch.float16)` (slice assignment avec cast). Refais le modèle de la phase 3 en passant à ce dernier pattern. Si tout reste ANE → bonne nouvelle. Si ça casse → tu sais qu'il faut refactorer le wrapper pour des `mul_`/`add_` ou un `copy_` direct sans cast intercalé.

**Phase 5 — Outer product via bmm.** Sur le modèle de la phase 4, remplace une mise à jour EMA scalaire par un `state ← α·state + bmm(x_reshape, k_reshape).reshape(...)`. C'est le motif du SSM update sans la trigonométrie. Test : le bmm avec gymnastique de reshape passe-t-il sur ANE en update d'état 4D ?


**Phase 6 — Trigonometry & RoPE bisection**

Context: continuing ANE dispatch bisection. Phases 0–5 all pass at 100% ANE on Hybrid backbone (1.5 + 2 + 3.1 + 4 mul + 5 matmul). Phase 6 tests the suspected ANE-breaker: trig ops and Mamba's RoPE rotation pattern. Subdivide into 6a/6b/6c so we know exactly which op breaks if any does.

Reference the original RoPE pipeline from `mamba3_siso_portable.py` lines 326–353:
```python
delta_theta = torch.tanh(angles) * torch.pi * DT.unsqueeze(-1)
theta = angle_state + delta_theta
cos = torch.cos(theta)
sin = torch.sin(theta)
# rope_pairwise(): reshape to pairs, mul/sub/add with cos/sin, stack, flatten, cat
```

**Phase 6a — `tanh` only.** Add a `theta_proj = nn.Linear(feature_dim, nheads * num_angles)` and a `dt_proj = nn.Linear(feature_dim, nheads)`. In forward, compute:
```python
angles = self.theta_proj(features).reshape(B, nheads, num_angles)
dt = F.softplus(self.dt_proj(features))  # (B, nheads)
gate = torch.tanh(angles) * dt.unsqueeze(-1)  # (B, nheads, num_angles)
# Use 'gate' to scale the outer product result before state fusion:
# delta_h = delta_h * gate.mean(-1, keepdim=True).unsqueeze(-1)  (or similar broadcast)
```
Goal: confirm tanh + softplus dispatch ANE. Expected pass.

**Phase 6b — `cos/sin` accumulation.** On top of 6a, add the angle accumulation and trig:
```python
delta_theta = torch.tanh(angles) * math.pi * dt.unsqueeze(-1)
theta = self.angle_state + delta_theta  # uses existing Phase 3 angle_state
cos = torch.cos(theta)
sin = torch.sin(theta)
# Update angle_state (Phase 4 mul pattern):
self.angle_state.mul_(0.0).add_(theta)  # or copy_ — keep consistent with Phase 4
# Use cos/sin to modulate K BEFORE the outer product (no rotation yet — just gating):
k_mod = k * cos.mean(-1, keepdim=True)  # simple scalar gating per head
# Continue with outer product on k_mod instead of k
```
Goal: isolate whether cos/sin themselves break ANE, independent of the rotation geometry. **This is THE critical test.**

**Phase 6c — full pairwise rotation.** On top of 6b, replace the scalar gating with the actual RoPE rotation:
```python
# Pairwise rotation on K (rotary_dim = 2 * num_angles)
k_rot = k[..., :rotary_dim].reshape(B, nheads, num_angles, 2)
k0, k1 = k_rot[..., 0], k_rot[..., 1]
ko0 = k0 * cos - k1 * sin
ko1 = k0 * sin + k1 * cos
k_rotated = torch.stack([ko0, ko1], dim=-1).flatten(-2)
if rotary_dim < d_state:
    k_rotated = torch.cat([k_rotated, k[..., rotary_dim:]], dim=-1)
# Use k_rotated in the outer product
```
Goal: confirm the geometric reshape/stack/flatten/cat pattern doesn't break ANE.

**Implementation rules:**
- Add three CLI flags: `--phase6a`, `--phase6b`, `--phase6c` (each implies the previous)
- No `.float()` casts anywhere — trust `compute_precision=FLOAT16`. If a precision warning appears, document it but don't preemptively cast.
- Keep `num_angles=16`, `rotary_dim=32` (matches Mamba3 locked config)
- Same export script, same Hybrid backbone, same Phase 4 mul pattern for state updates
- For each sub-phase: export `.mlpackage`, run on iPhone 17 Pro Performance Report, document in `updates.md` using the same table format as Phase 5 (ANE Ops / CPU Ops / Median Prediction / Median Compile / ANE % / Precision)

**Decision logic to record at the end:**
- 6a passes, 6b fails → cos/sin is the bottleneck. Next: composite op for `aten::cos`/`aten::sin` lowering to fp32-localized version, OR polynomial approximation
- 6b passes, 6c fails → reshape/stack/flatten geometry is the bottleneck. Next: replacement using simpler `mul`+`add` on full slices instead of pair-stacking
- All three pass → trig is innocent, move to Phase 7 (full Mamba composition with all elements + softplus discretization + sigmoid trap + RMSNorm on B/C)

**Phase 7 — Composition complète.** À ce stade tu sais déjà ce qui passe et ce qui casse. Tu réassembles step-by-step les autres ingrédients (RMSNorm custom, `expand`/`split`, softplus, sigmoid, le double bmm, le pattern complet de la récurrence). Les phases qui ont passé seul pourraient échouer combinées — c'est rare mais possible.

**Comment tu sais que tu es à la bonne phase :** chaque phase produit un `.mlpackage`, tu l'ouvres dans Xcode → Performance Report → device réel → Compute Unit Mapping. Tu cherches le pourcentage CPU. Tant qu'il reste autour de 0%, tu avances. Le moment où il bondit à 100%, tu as identifié l'op fautive — et tu peux la regarder précisément dans le tab Structure pour voir comment elle apparaît dans le MIL.

**Discipline qui paie :** chaque phase = un fichier Python, un export, un screenshot du Performance Report, dans un même dossier `bisect/p1`, `bisect/p2`, etc. Quand tu trouveras le coupable, tu auras la preuve documentée.

Mon pari personnel sur où ça va casser : phase 4 (le pattern de cast en slice assignment) ou phase 6 (cos/sin fp16). Si ce sont les deux, tu sais quoi faire — refactor du wrapper pour la phase 4, et split CPU-trig à la Gemini *uniquement pour la trigo* pour la phase 6, le reste restant entièrement ANE-stateful. Et là on aura répondu à la vraie question.