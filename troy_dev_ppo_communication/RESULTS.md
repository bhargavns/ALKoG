# Phase-2 Symbol-Grounding Results

**Last updated:** 2026-09-13
**Scope:** PPO over distance-gated KG symbol triples — architecture shootout,
value-loss normalization, a symbol-forcing conflict task, transfer to the
real SAM perception pipeline, swapping KG concept matching for a supervised
category head while keeping the triple structure, and a DIAL-style
differentiable communication channel measured against matched controls.

---

## TL;DR

1. **Value normalization is the headline win.** The shared symbol trunk was being
   trained 75–330× more by the critic than the actor (measured). Normalizing the
   value loss to the advantage scale fixed this and took the winning oracle model
   from **0.66 → 0.98** food, and *resurrected* the transformer from **0.28 → 0.957**.
2. **On the clean oracle, architecture barely matters** once value-normalized:
   slot 0.98, transformer 0.957. The pre-norm shootout that ranked slot ≫ transformer
   was **confounded** by value dominance.
3. **None of the oracle wins transferred to the real SAM pipeline.** There the
   **transformer fails to learn** (0.36, random-level) while **slot learns** to the
   old 0.52 baseline. Normalization is neutral-to-positive on real perception; the
   transformer is a regression. **Production reverted to slot + normalization.**
4. **The real bottleneck is perception noise**, not the value/actor gradient balance
   or the policy architecture — confirmed by measurement: SAM proposes the food in only
   **0.309** of frames where it is visible.
5. **The two-agent communication phase produced nothing under RIAL** (§11). With a
   symbol-emitting transmitter and a blind receiver learning independently across a
   sampled discrete channel, 450 iterations / 4,712 episodes left the receiver's
   explained variance at **+0.00007** and its policy at uniform. Neither side has a
   gradient until the other moves first; DIAL is the indicated fix, and this run is
   its control arm.
6. **Replacing KG concept matching with a supervised category head takes the real
   pipeline from 0.52 → 0.925 food (held out), with zero loose-lion deaths** (§7). The
   triple structure is untouched — `(lion, inside, cage)` still comes from 2D box
   geometry, only *identity* changed. The win came from **identity stability**, not
   better detection: SAM's recall is unchanged, but every detection now lands on a
   stable, correct symbol instead of a splintered node. **Single seed — not yet
   established.**
7. **DIAL fixes the gradient and does not move the task** (§12). A differentiable
   noisy-sigmoid channel carries real information — receiver explained variance
   **0.080 vs 0.00005** for an architecture-matched control whose transmitter was
   provably frozen — but food rate differs by **0.13 percentage points** between them
   over ~4,685 episodes each, and receiver policy entropy never leaves uniform. The
   channel transmits; the actor never steers. Gradient flow was never the binding
   constraint.
8. **Giving the receiver the transmitter's orientation did not help either** (§13,
   partial). With `receiver_yaw_mode=shared`, all three DIAL variants reached
   iteration ~276/450 before being stopped, and every task metric matches the
   independent-yaw runs at the same iteration count: food 0.36–0.37, receiver entropy
   still at uniform. The unobserved-yaw problem was not the binding constraint.

---

## 1. Setup

**Task.** A kinematic arena (Discrete(4): forward/back/left/right, no momentum). The
agent must reach the green **food** (+10) while avoiding a **loose lion** (−50 on
contact within 0.8). A **caged lion** (inside the blue cage) is harmless. The
caged/loose flag is **absent from the observation** — the agent can only obtain it
through the KG symbol triples.

**Key constants.** `FOOD_REWARD=+10`, `LION_PENALTY=−50`, `STEP_PENALTY=−0.01`,
`FOOD_SHAPING=+0.5·Δdist` (potential-based), `FOOD_REACH=LION_CATCH=0.8`, arena ±7,
max 300 steps.

**Policy input.** `[x/8, y/8, cos yaw, sin yaw]` (4) + up to `k=6` perceived triples
`[src‖rel‖dst]`, each token a 4-dim symbol + slot positional embedding, node tokens
gated by a learned 2→4 `dist_proj` of the agent-frame offset. Actor and critic are
separate 1-hidden-layer (38) MLPs over a **shared** `SymbolTable` trunk.

**Perception.** Oracle = perfect, always-present triples with exact deltas (plus a
"distractor" variant that injects random cage-aspect entries so slot order churns).
Real = SAM boxes → ResNet-18 embeddings → KG concept match → geometric relations,
re-run every episode reset and every 75 steps, with per-episode anchor memory.

**Metrics.** `food_rate` = fraction of episodes reaching food (headline).
`caged/loose_food_rate`, `loose_death_rate` split by scenario. Entropy near ln4≈1.386
means an uncommitted (≈random) policy. Windowed averages below use the **last ⅙** of
each run unless noted.

---

## 2. Oracle architecture shootout (no normalization)

The original comparison of aggregation architectures on the **distractor** oracle.
`food0`→`foodF` shows start vs final food rate.

| Architecture | iters | food0 → foodF | return | loose_food | loose_death | entropy | expl-var |
|---|---|---|---|---|---|---|---|
| **slot attention** | 300 | 0.31 → **0.66** | +3.75 | 0.57 | 0.15 | 1.30 | 0.51 |
| sum pooling (Deep-Sets) | 300 | 0.34 → 0.42 | −3.24 | 0.36 | 0.27 | 1.35 | 0.33 |
| flat concat | 120 | 0.30 → 0.31 | −3.57 | 0.28 | 0.19 | 1.35 | 0.52 |
| transformer v1 (CLS-only) | 300 | 0.33 → 0.28 | −1.65 | 0.17 | 0.09 | 1.06 | 0.60 |
| transformer v2 (CLS+meanpool) | 300 | 0.30 → 0.28 | −3.98 | 0.28 | 0.20 | 1.25 | 0.52 |

*Clean (non-distractor) oracle, for reference (from prior README):* slot 0.82,
sum **0.95**, flat 0.87, transformer 0.35.

**Read at the time:** slot attention was the only architecture robust to distractor
churn; the transformer "failed." **This conclusion was later shown to be confounded**
(see §4).

---

## 3. Diagnostic: the critic dominates the shared symbol trunk

Measured the gradient deposited on the shared `SymbolTable` by the actor-side loss
(`pi_loss − ent_coef·ent`) vs the value-side loss (`vf_coef·v_loss`), on real
production rollout buffers. `v/a` = ‖g_value‖ / ‖g_actor‖; cosine = alignment of the
two gradients.

| Model | v/a @ init | v/a @ it25 | v/a @ it50 | cos @ it50 | `dist_proj` gate v/a |
|---|---|---|---|---|---|
| slot | 8.2× | 127× | 75× | +0.10 | 141× |
| transformer | 21× | 1438× | 333× | +0.74 | 228× |

**Cause (fixable):** advantages are normalized to unit std, but value targets are raw
returns (~8–13), so the value gradient is 75–330× larger. The `dist_proj` direction
gate — which carries the food-direction signal — was ~99% value-trained. The
transformer had the **worst** dominance (333×), which is exactly why it failed hardest.

**Fix:** `normalize_value_loss` — divide the value-loss residual by the batch return
std so the critic gradient matches the unit-std advantage scale (critic still predicts
raw values; GAE/bootstrap unaffected). After the fix, slot's trunk ratio dropped
**127× → 9.7×** and the residual gradient flipped from orthogonal (cos ≈ 0) to aligned
(**+0.82**).

---

## 4. Value normalization on the oracle

| Config | iters | food0 → foodF | return | caged_food | loose_food | loose_death | entropy |
|---|---|---|---|---|---|---|---|
| slot, no-norm | 300 | 0.31 → 0.66 | +3.75 | 0.73 | 0.57 | 0.15 | 1.30 |
| **slot + norm** | 300 | 0.30 → **0.98** | +12.33 | 1.00 | 0.97 | 0.03 | **0.65** |
| transformer v2, no-norm | 300 | 0.30 → 0.28 | −3.98 | 0.30 | 0.28 | 0.20 | 1.25 |
| **transformer v2 + norm** | 300 | 0.30 → **0.957** | +11.24 | 1.00 | 0.91 | 0.06 | 0.99 |

**Findings:**
- Normalization took slot **0.66 → 0.98** (essentially solved) and made the policy
  commit (entropy 1.30 → 0.65).
- It **resurrected the transformer, 0.28 → 0.957** — the "dead end" verdict was wrong;
  the transformer's blocker was value starvation, not its aggregation. The transformer
  has a long flat plateau (~150 it, the known transformer-under-PPO slowness) then a
  sharp takeoff.
- **The whole no-norm shootout (§2) was confounded.** Once normalized, slot (0.98) and
  transformer (0.957) are ~tied; architecture rankings from the no-norm era are not
  trustworthy.

---

## 5. The conflict layout (symbol-forcing task)

**Motivation.** On the base task the loose lion is a static, always-avoidable obstacle,
so a heuristic ("always avoid" or "always approach") scores well without using the
symbol. The conflict layout co-locates the food with the lion in 50% of episodes and
flips termination so a loose lion in range kills *before* the food is collected:

- **food-on-caged-lion** → safe to grab (approach).
- **food-on-loose-lion** → death trap (must avoid, forgoing the food).

Only a policy that reads the caged/loose symbol *and* the food-lion spatial relation
gets both high food and low death. Scripted-policy check confirmed the structure:
always-approach dies 100% in the near-loose cell; symbol-user survives; near-caged food
is fully grabbable. Optimal food rate ≈ 0.75 (near-loose food is unwinnable).

| Config (oracle, +norm) | food | caged_food | near_caged_food | near_loose_death | near_loose_food | entropy |
|---|---|---|---|---|---|---|
| slot + norm + conflict | 0.574 | 1.00 | **1.00** | 0.16 | 0.00 | 1.15 |
| transformer + norm + conflict | 0.586 | 0.99 | 0.96 | 0.21 | 0.00 | 1.24 |

**Findings:**
- Both architectures **prove symbol use**: `near_caged_food` ≈ 1.0 (approaching food on a
  caged lion is impossible without the symbol).
- Both **only partially solve** the compositional task: over-cautious in far-loose
  (leaving safe food) and imperfect trap avoidance (`near_loose_death` ~0.16–0.21).
- **Architecture-independent** (slot 0.574 ≈ transformer 0.586) — the compositional gap
  is a *training* limitation, not an aggregation one.

---

## 6. Real SAM pipeline

The real perception pipeline (SAM + ResNet + relation heuristics). The conflict layout
uses a 1.5-unit food-lion offset (exact overlap breaks SAM via occlusion).

| Config | iters | food0 → foodF | caged_food | loose_food | loose_death | return | entropy | expl-var | learned? |
|---|---|---|---|---|---|---|---|---|---|
| **old slot, no-norm (baseline)** | 300 | learned to **~0.52** | 0.55 | 0.43 | 0.28 | −1.98 | 1.27 | 0.19 | **yes** |
| transformer + norm + conflict | 300 | 0.31 → 0.36 | 0.47 | 0.25 | 0.31 | −4.94 | **1.35** | 0.16 | **no** |
| transformer + norm (no conflict) | 300 | 0.32 → 0.36 | 0.45 | 0.25 | 0.29 | −4.90 | **1.35** | 0.19 | **no** |
| **slot + norm (no conflict)** | 300 | 0.38 → **0.52** | 0.62 | 0.45 | 0.26 | −1.39 | 1.27 | 0.35 | **yes** |

*(Baseline: the earlier slot run climbed 0.27→0.52 and its continuation plateaued
~0.49–0.52.)*

**Findings (fully disentangled):**
- **Transformer + norm does not learn** on real perception — food flat ~0.36
  (random-walk level), entropy pinned at ~1.35 (≈97% of max) the whole run. Same result
  with and without the conflict layout, so **conflict is exonerated** as the cause.
- **Slot + norm learns** to 0.52 — matching the old baseline, with entropy dropping
  1.35→1.27 and expl-var rising 0.06→0.36. It was still climbing at iteration 300.
- Therefore: **normalization is fine on real perception** (neutral-to-slightly-positive);
  **the transformer is the regression.** Its self-attention over a noisy, variable SAM
  token set is less robust than slot attention's fixed content-addressed role slots —
  a difference the perfect-perception oracle could not reveal.
- **Production reverted to slot; normalization kept on.**

---

## 7. Soft-category perception (2026-08-19)

Merged from `bhargav-softcat`, which forked from `86ff78c` in parallel and shares none
of §2–§6's work. Its own policy (`SoftCategoryActorCritic`) replaces triples with five
fixed category slots plus **two hardcoded scalars** (lion-inside-cage, lion-near-cage) —
it hands the policy the caged/loose bit rather than making it read a relation. That
policy is **not** used here. What was taken is only its *perception front-end*.

**What `perception=softcat` is.** A supervised head (`ProposalCategoryHead`: LayerNorm →
128 → GELU → Dropout → 6) over the frozen SAM+ResNet embedding, trained on MuJoCo
segmentation labels. Its `argmax` replaces `kg.match_batch_sims()` as the concept id;
its max probability replaces the match cosine. **Everything downstream is unchanged** —
`union_boxes_by_concept`, `relations_from_boxes`, `ground_delta`, `assemble_triples_geo`,
`AnchorMemory`, `SymbolTable`, `SlotAttention`, `PPOTrainer`. Triples stay `[k=6, 3]`
`(src, rel, dst)` into a 76-dim policy input, and `inside`/`near` are still inferred
from 2D box overlap. One node per category, so a caged and a loose lion share a symbol
and the distinction must travel through `(lion, inside, cage)`.

### Category head: perfect, and that is the point

3,040 proposals from 1,600 frames (200 scenes × 2 views × 4 cameras), 3-object world.

| Split | Proposals | Accuracy | Macro-F1 |
|---|---|---|---|
| Stored validation (grouped by frame) | 600 | 1.0000 | 1.0000 |
| Live holdout (unseen seeds) | 141 | 1.0000 | 1.0000 |

**The classification task is trivial and the score is not a result.** Ablation: the
64-dim colour histogram *alone* scores 1.0000 (CNN-only 0.9950); class-mean colour
histograms are near-orthogonal (pairwise cosine 0.001–0.17). The objects are colour-coded,
so "classify this proposal" is "what colour is this blob." The head trained in 3 seconds.

**SAM proposal recall is the real measurement:** cage 0.925, lion 0.576, **food 0.309**.
This directly confirms §7.4's inference — the food sphere is simply not proposed most of
the time. Fragmentation is not a problem (cage 3.28 proposals/frame, lion 2.30, food 1.04,
all classified correctly).

### The result: 0.52 → 0.925 held out

300 iterations then `init_from` to 600, horizon 2048, k=6, `perceive_every=75`, slot +
`normalize_value_loss=True`, 3-object world, seed 0. Converged: slope over the final 100
iterations **−0.008 food/100it** (vs +0.213 at iteration 300).

| Iters | food | caged | loose | loose_death | return | entropy |
|---|---|---|---|---|---|---|
| 1–50 | 0.371 | 0.381 | 0.315 | 0.319 | −4.24 | 1.348 |
| 251–300 | 0.653 | 0.763 | 0.532 | 0.215 | +2.01 | 1.227 |
| 351–400 | 0.822 | 0.903 | 0.742 | 0.096 | +7.66 | 1.172 |
| **551–600** | **0.857** | 0.895 | 0.817 | 0.088 | +8.38 | 1.125 |

**Held-out evaluation** — 40 episodes at unseen seeds (777000 + i·97), stochastic policy:

| | in-sample (551–600) | **held out** | §6 baseline |
|---|---|---|---|
| food_rate | 0.857 | **0.925** | 0.52 |
| caged_food | 0.895 | **0.958** (24 eps) | 0.62 |
| loose_food | 0.817 | **0.875** (16 eps) | 0.45 |
| loose_death | 0.088 | **0.000** | 0.26 |
| mean_return | +8.38 | **+11.42** | −1.39 |

It scores *higher* out of sample than in, so the gain is not memorisation of seed 0's
replayed layout stream. **Zero deaths across 16 loose-lion episodes** while still
collecting food in 87.5% of them is the strongest number here: it requires actually
reading `(lion, inside, cage)` vs `(lion, near, cage)`, which is the compositional
behaviour §5 was built to force and only half-achieved on the oracle.

### Why it works — and a prediction that was wrong

Predicted before running: softcat ≈ oracle_id ≈ 0.52, because SAM recall (food 0.31) is
the bottleneck and a perfect classifier cannot label a box SAM never proposed. **The
recall premise was right; the conclusion was wrong.**

What that missed: the KG path does not only *miss* the food, it also assigns detections
to splintered or wrong nodes — the failure the `Oracle.py` false-merge/false-split
diagnostics exist to measure. The classifier removes that entirely (156/156 agreement
with the MuJoCo oracle on real proposals), so every detection that *does* occur lands on
a stable, correct symbol. **Identity consistency mattered more than detection frequency.**
This also explains the shape of the gain: `caged_food` moved first and most, because
reliably identifying lion and cage is what makes the `inside` triple fire consistently.

### Oracle perception, for reference

`perception=oracle_id` (SAM boxes, ground-truth identity) and `perception=oracle_full`
(boxes from the segmentation image) were added as `lib/OracleGrounding.py`, same
4-tuple contract, to separate detection error from identity error:

| | food detected | `(lion,inside,cage)` when caged | when loose |
|---|---|---|---|
| oracle_id | 0.47–0.60 | 0.60 | 0.00 |
| oracle_full | **1.00** | **0.87** | 0.07 |

`oracle_id` barely improves detection — confirming KG matching was never the detection
bottleneck. `oracle_full`'s residual 0.13 miss / 0.07 false positive are **not**
perception: they are the projective 2D `inside` heuristic (a distant food overlapping a
near cage in the image plane registers as inside). That error source is masked in the
SAM path only because SAM misses far more.

A 200-iteration `oracle_full` run reached only 0.36 food — but at horizon **1024**, half
the batch size, so it is not comparable and should not be read as a policy ceiling.

### Caveats

- **Single seed (0).** Seeds 1–2 not yet run, so seed variance is unmeasured. §9 shows
  this project has already had headline conclusions overturned once a hidden confound
  surfaced (there, value dominance) — treat 0.925 as provisional until replicated.
- The `oracle_full` comparison above is at half horizon and is not like-for-like.
- Trivially separable colours mean the category head would not survive two categories
  sharing a colour — the classifier is untested as a classifier.

---

## 8. Conclusions

1. **Value normalization is a genuine, keepable improvement.** Transformative on the
   oracle (0.66→0.98), harmless-to-helpful on the real pipeline. Rooted in a measured
   pathology (critic swamps actor 75–330×) with an identified root cause (normalized
   advantages vs unnormalized value targets).
2. **The oracle is a poor predictor of real-pipeline learning.** It equated slot and
   transformer and hid the transformer's fragility to perception noise. Architecture/
   training changes must be validated on the real pipeline, not just the oracle.
3. **Slot attention's inductive bias wins under noise.** Fixed content-addressed role
   slots are more robust to unreliable SAM input than the transformer's attention.
4. **The real ceiling is perception, not policy** — and §7 confirms it, then partly
   removes it. SAM's food recall is 0.309 (measured), but the larger recoverable loss was
   *concept-identity* noise, not detection: replacing KG matching with a trained category
   head took the real pipeline from 0.52 to **0.925** held out, with the triple structure
   untouched.
5. **Supervised category identity beats unsupervised concept discovery on this world**
   (§7). That is a departure from the project's emergent-symbol premise and should be
   read as a measurement, not an endorsement: the classification task here is
   colour-trivial, and the win is in identity *stability*, not in perception generally.

## 9. Corrections logged (honesty ledger)

- "Transformer is a dead end" (§2) — **wrong**; it was value-starved (§4).
- "Aggregation/dilution is the transformer's blocker" — **wrong**; value dominance was.
- "Transformer's cross-attention should beat slot on the compositional task" — **wrong**;
  they tied (§5).
- "Switch production to transformer + carry norm to real" (oracle-motivated) — the
  **transformer half was wrong** for the real pipeline (§6); norm was fine.
- "softcat will land at ~0.52 because SAM recall is the bottleneck" (§7) — **wrong**.
  The recall premise held, but identity stability dominated; it reached 0.925 held out.
- "The category head's 100% accuracy shows the approach works" — **wrong**, and caught
  before it propagated: colour histogram alone scores 1.0000, so the score measures the
  world, not the method (§7).
- "Shrinking the symbol vocabulary 10 → 4 will help the communication bootstrap"
  (§11) — **wrong**. Proposed as the cheap lever; both probes left receiver
  explained-var pinned at zero. The blocker is the absence of a gradient, not the
  size of the code space, and 4x4 undirected search is as undirected as 10x4.
- "Two consecutive SAM passes give identical static detections" — stated firmly
  from **6 episodes**; at 20 episodes 7/20 differed (box jitter). The operative
  conclusion survived at 30 episodes (0/30 gained an object), but the confidence
  was not supported by the sample when first asserted.
- "Sharing one helper means the trainer and the video recorder cannot drift apart"
  (§11 tooling) — **wrong**, and caught in review: the per-step anchor refresh was
  still copy-pasted and receiver-stripping lived in neither shared path, so a video
  could have shown an input the policy never saw. Fixed by giving both callers one
  `TransmitterInput` built from a single factory, then verifying identical output
  from identical state rather than asserting it.
- "DIAL means a straight-through Gumbel-softmax channel" (§10, above) —
  **wrong method attributed**. Foerster et al. use `sigmoid(z + Gaussian noise)` in
  training and `1[z > 0]` at execution. Gumbel-softmax was never their mechanism.
  §12 implements the actual one.
- "σ=2 is the wrong reference, the probe fails 2/3 seeds there" — **wrong, and it
  was a budget artifact**. At 300 updates σ=0.5 looked dominant; at 1,000 updates
  across 8 seeds the two sigmas tie *exactly* on hard accuracy (0.844 each). The
  apparent margin was small-sample noise on accuracies quantized to steps of 0.25.
  Both the original preference for σ=0.5 and the subsequent reversal to σ=2 were
  overconfident reads of three seeds.
- "The critic is stealing the transmitter's gradient; detaching it on the critic
  branch should free the actor to steer" — **wrong**. `critic_detach=1` zeroed the
  message-head value gradient exactly as designed and lowered receiver explained
  variance as predicted (0.052 → 0.015), but produced **no control gain**. Critic
  dominance was real and measured (6.7–15×) and was not the binding constraint.
- "Channel saturation climbing to 0.90 is reassuring for the σ=0.5 choice" —
  **incomplete, and stated as if it were purely good news**. It did retire the
  analog-exploitation risk, but the same saturation predicts gradient collapse:
  median message-head actor gradient falls 37× from the unsaturated to the
  saturated regime (§12). Saturation was simultaneously the good news and the bad.

## 10. Current best config & open questions

**Best real-pipeline config:** `perception=softcat` + slot attention +
`normalize_value_loss=True`, 600 iterations — **0.925 food, 0.000 loose-death held out**
(§7). Single seed; the prior KG-matching config sits at ~0.52.

**Communication phase:** RIAL is a measured dead end (§11) and **DIAL is now also a
measured null on task performance** (§12), using the paper's actual noisy-sigmoid
channel rather than the Gumbel-softmax this section previously proposed. Gradient
flow across the channel is verified and information transfer is real; neither becomes
control. The settled design question: one joint optimizer with named parameter groups,
which is what `DIALPPOTrainer` uses.

DIAL_PLAN §9's shared-yaw diagnostic has now also been run (§13, stopped at ~61%
of budget): giving the receiver the transmitter's orientation changed nothing through
276 iterations. That removes the strongest remaining explanation on the *task* side.
With the channel, the critic's influence, and the receiver's frame all ruled out as
single causes, the remaining candidates are the receiver itself (no recurrence, no
previous-action input, a 4-dim transformer over messages only) and the reward
signal's credit-assignment horizon. None has a prespecified experiment yet.

**Open questions / next steps:**
- **Seeds 1–2 for the §7 result** — the single outstanding blocker on treating 0.925
  as established.
- SAM food recall (0.309) is still the largest untouched loss: proposal generation
  (a food-sized `min_area_frac`, or a trainable mask decoder) is the next frontier,
  not the classifier.
- The projective 2D `inside` heuristic misfires even under perfect perception
  (0.13 miss / 0.07 false positive, §7) — a relation-side error independent of SAM.
- Does the §7 gain survive a world where two categories share a colour? The category
  head is currently untested as a classifier.
- The compositional conflict task (§5) is only half-solved even on the oracle — a
  training-side problem (curriculum / entropy annealing / shaping), architecture-agnostic.

## 11. Communication phase: the RIAL baseline is a flat line (2026-08-30)

A second agent was added and the task restructured: one agent perceives but cannot
move, the other moves but cannot see. The only link between them is a discrete
symbol. **450 iterations produced no communication at all**, for a reason that is
structural rather than a matter of budget.

### Setup

**Transmitter.** The existing agent, made immobile, keeping the 4-camera rig and
the whole §7 perception stack. Its input is unchanged in kind -- 4 + (k+1)*12 = 88
dims -- but its head emits one of 4 symbols (`a`-`d`) per step instead of a move.
Because it never moves and lion/cage/food never move, the only varying quantity
in its input is the receiver's position.

**Receiver.** Blind. Its entire input is the symbol stream: 11 tokens x 4 dims =
44, laid out `[CLS][s_1]..[s_n][NULL]x(10-n)`, learned embeddings plus learned
positional encodings, through 3 transformer blocks with 1 head each. The
contextualized CLS feeds a 4-unit hidden layer then 4 action logits. After 10
symbols the window slides. 865 parameters.

**Reward.** Both agents receive the SAME reward -- the ordinary task outcome,
evaluated on the *receiver's* position. Neither has a private objective, so the
transmitter can only score by steering and the receiver only by reading. No
gradient crosses the channel: the symbol is a sampled discrete index.

**Perception cadence.** SAM runs ONCE per episode, at reset. The transmitter
cannot move and the static objects do not move, so their anchors stay exact.
Only the receiver's position updates, read from the simulator every step -- a
deliberate oracle shortcut, because SAM at ~1.1 s/frame cannot run per-step and
a frozen anchor for the one moving object would be uncorrelated with the truth.
The receiver is position-only and never enters a relation.

### The result: nothing, for 450 iterations

450 iterations, 4,712 episodes, ~4.2 h.

| iters | RX ev | RX ent | TX ev | TX ent | Hsym | food | return |
|---|---|---|---|---|---|---|---|
| 1-50 | 0.0001 | 1.370 | 0.184 | 1.293 | 1.311 | 0.350 | -5.07 |
| 201-250 | 0.0000 | 1.380 | 0.255 | 1.215 | 1.280 | 0.422 | -4.71 |
| 401-450 | **-0.0003** | **1.371** | 0.365 | 1.186 | 1.326 | 0.433 | -4.34 |

- **The receiver never learned anything.** `rx_explained_var` mean over all 450
  iterations is **+0.00007**; only **6/450** iterations exceeded `|ev| > 0.01`.
  Its entropy never departed from uniform by more than **0.0705 nats** against
  `ln(4) = 1.3863`.
- **The transmitter never formed a code.** Its dominant symbol changed **169
  times** across 450 iterations, and no symbol held dominance for more than 40%
  of the run (`d` 181, `b` 97, `c` 96, `a` 76). Final-50 mix `a .23 / b .16 /
  c .33 / d .28` -- skewed but unsettled. A random walk in symbol space.
- **`food_rate` drifted 0.350 -> 0.433 and this is NOT improvement.** Slope
  +0.017 per 100 iterations, and the receiver's policy is provably unchanged
  (entropy pinned at uniform), so it cannot have gotten better at navigating.
  Recorded here because the headline metric reads as progress and is not.

### Why: neither agent has a gradient

- Receiver: its gradient is `grad log pi(a|context) * A`. With symbols
  uncorrelated with the world, no *conditional* policy beats the marginal one,
  so it can only learn a context-independent action prior.
- Transmitter: its gradient is `grad log pi(s|obs) * A`. Since the receiver
  ignores symbols, `A` is statistically independent of `s`, so the expected
  gradient is **zero** and what arrives is noise.

This is RIAL (Foerster et al. 2016), and this is RIAL's known failure mode,
reproduced here with 4,712 episodes behind it. **DIAL -- a straight-through
Gumbel-softmax channel, so the receiver's loss backpropagates into the
transmitter -- is the indicated next step.** This run stands as its control arm.

### What did work

`tx_explained_var` climbed **0.184 -> 0.365**, peaking at **+0.81** (iteration 265). The
transmitter's critic genuinely learned to predict episode return from
SAM -> category head -> triples -> distance-gated symbols -> receiver slot. The
perception stack is sound; the channel is what is dead.

Supporting measurements from building the phase:

- **A `receiver` category was added to the §7 vocabulary** (7 classes now; ids
  0-4 unchanged, BACKGROUND 5->6). Dataset regenerated: 3,967 proposals over
  1,600 frames. Proposal recall cage 0.915, **receiver 0.523**, lion 0.495,
  food 0.397. The head scores 1.0000 accuracy / 1.0000 macro-F1 on both the
  stored split and a live holdout -- which, per §7, measures the world's
  colour-triviality, not the method.
- **Receiver body size was set from measurement, not taste.** At a 0.28 body
  sphere SAM proposed it in 0.225 of frames where it was visible; at 0.44,
  0.500, with food/lion/cage recall unchanged. The larger form was kept because
  reliable identification is the point of the phase.
- **A second SAM pass per episode was measured and dropped.** Over 30 episodes
  it added a static object **0 times** (mean known 2.47/3 either way), at ~2 s
  per episode. Extra passes buy box jitter, not information, while the
  transmitter is immobile.
- **Lion recall fell 0.576 -> 0.495** (cage 0.925 -> 0.915) once the receiver
  was added, most likely occlusion from a new 0.44-radius body. Food is
  unchanged within noise. A shift from the §7 baseline worth remembering.

### Caveats

- **Single seed (0).** As everywhere else in this document.
- The receiver's localisation is oracle-fed, so this phase does not test whether
  the transmitter could *find* its partner, only whether it could describe the
  world to it. A colour-threshold tracker was prototyped as the vision-only
  alternative (median position error 0.545 units, 3.3 ms/pass, 1327x faster than
  SAM) and can replace the oracle without touching anything downstream.
- A negative result at one architecture and one reward shape does not establish
  that RIAL cannot work here -- only that it did not, over 450 iterations, with
  both ends initialized at random.

## 12. DIAL: the gradient crosses the channel, the task does not move (2026-09-13)

The channel was made differentiable, as §10 and §11 called for. It works, in the
narrow sense that the receiver's loss now reaches the transmitter and real
information crosses. It produced **no task improvement over an architecture-matched
control**. Gradient flow was not the binding constraint.

### What was implemented

Foerster et al.'s actual DIAL channel, not the Gumbel-softmax §10 proposed (logged in
§9). Training uses `b = sigmoid(z + sigma * noise)` on two real logits; execution uses
`b = 1[z > 0]`, mapping `00/01/10/11` to `a/b/c/d`, so hard execution still transmits
exactly one of the same four symbols §11 used. The receiver reads a continuous
four-corner interpolation of its existing symbol embeddings, exact at the binary
corners, so hard and continuous inference share one representation.

The transmitter's private critic and categorical-message PPO objective are **removed**:
its message output is an internal activation of a single composed policy, trained by
one joint Adam with named parameter groups. Messages are never buffered as
activations — the rollout stores observations, triples, deltas and the Gaussian noise
sample, and every minibatch replays the full ten-message window through the *current*
transmitter. See [DIAL.md](DIAL.md) and [DIAL_PLAN.md](DIAL_PLAN.md).

### Sigma selection: the toy probe does not discriminate

A four-target identification probe (production channel, receiver, window
reconstruction and PPO update; one-step episodes) at 1,000 updates across 8 seeds:

| | mean hard | mean continuous | mean gap | seeds at 1.00 hard |
|---|---:|---:|---:|---:|
| σ=0.5 | 0.844 | 0.995 | 0.151 | 4/8 |
| σ=2.0 | 0.844 | 0.844 | **0.000** | 5/8 |

**Identical on hard accuracy**, which is the deployment condition. Head-to-head σ=0.5
wins 1 seed, loses 2, ties 5. The only real difference is the train/execution gap:
σ=0.5 nearly solves the toy in continuous terms and gives 15 points back to
thresholding, while σ=2.0 has no gap at all on any seed. Neither passes DIAL_PLAN §7's
three-seed 95% gate. The arena below ran at **σ=0.5** with that gate knowingly open.

### The arms

Seed 0, 450 iterations × 2,048 steps, σ=0.5, lr 3e-4, `target_kl=0.03`, value
normalization on, identical environment and budget. Three ran concurrently on one GPU
at 76–87 s/iteration.

| Run | Configuration | Role |
|---|---|---|
| `dial_s0` | `method=dial_ppo` | Treatment — full cross-agent gradient |
| `detached_s0` | `+ detach_channel=1` | Control — same architecture, gradient severed |
| `critic_detach_s0` | `+ critic_detach=1` | Ablation — actor-only gradient shapes the code |
| `baseline_s0` | `method=independent_ppo` | RIAL control, **stopped at iteration 120** |

### The result

Full 450 iterations each, ~4,685 episodes per arm:

| metric | DIAL | detached | delta |
|---|---:|---:|---:|
| food_rate | 0.37481 | 0.37353 | **+0.00128** |
| mean_return | −5.22463 | −5.47177 | +0.24714 |
| loose_death_rate | 0.32583 | 0.33129 | −0.00546 |
| mean_length | 202.65209 | 202.71811 | −0.06602 |
| rx_entropy | 1.36549 | 1.37019 | −0.00469 |
| **rx_explained_var** | **0.08007** | **0.00005** | **+0.08002** |

Food rate differs by **0.13 percentage points** from a control whose transmitter never
received a single gradient. Episode length, loose-lion deaths and receiver entropy are
equally indistinguishable. Over the final 50 iterations DIAL is *worse* on food
(0.411 vs 0.438). Against DIAL_PLAN §8: actor gradients verified across the channel
**yes**; improved task performance over the matched control **no**.

DIAL's own trajectory, showing what did change:

| window | food | return | rx_ent | rx_ev | saturation | actor_grad | value_grad |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–50 | 0.338 | −5.79 | 1.3686 | +0.0096 | 0.103 | 0.003459 | 0.001822 |
| 101–150 | 0.375 | −5.56 | 1.3744 | +0.0656 | 0.891 | 0.001230 | 0.008248 |
| 201–250 | 0.344 | −5.80 | 1.3589 | +0.1221 | 0.861 | 0.001376 | 0.009152 |
| 301–350 | 0.418 | −5.63 | 1.3697 | +0.1379 | 0.642 | 0.006429 | 0.024523 |
| 401–450 | 0.411 | −3.70 | 1.3695 | +0.0278 | 0.564 | 0.009295 | 0.050591 |

### What is genuinely established

- **Information crosses the channel.** `rx_explained_var` 0.080 vs 0.00005 (peak
  0.706 vs 0.0024) — a factor of 1,600. The receiver's critic learned to predict
  return from the message stream, which was categorically impossible under RIAL.
  The transmitter's node symbols moved (L2 delta 0.150–0.269).
- **The control is airtight.** Summed transmitter gradient across all 450 detached
  iterations is **exactly 0.0**, and its node symbols are bit-identical to
  initialization (L2 delta 0.000, cosine +1.000). Any DIAL-vs-detached difference is
  attributable to the channel gradient alone.
- **Optimization was never the problem.** 0/450 KL early stops in either arm; no
  update was ever rejected.
- **The detached arm independently reproduced the §11 null:** mean
  `rx_explained_var` +0.0000519, against the RIAL run's +0.00007. Two different
  architectures, the same dead receiver.

### What did not happen

The receiver never committed to a policy. Its entropy finished **0.0169 nats** from
uniform (`ln 4 = 1.3863`) — less departure than §11's acknowledged failure ever showed
(0.0705). Information reached the critic and never reached the actor.

`critic_detach=1` **disconfirmed the critic-dominance hypothesis.** The measurement
that motivated it was real: value-side gradient at the message head ran 6.7–15× the
actor-side gradient, exactly the §3 pathology at a new location. Detaching it worked
mechanically — message-head value gradient exactly 0, receiver explained variance
down 0.052 → 0.015 as predicted, since the critic can no longer shape the code toward
its own predictability — and produced **no control gain**: food 0.370 vs DIAL's 0.360
and the detached control's 0.364, over the 177 iterations all three shared.

This arm was **stopped deliberately at iteration 405/450** (4,264 episodes) and never
completed. Its terminal full-run means: food 0.391, return −4.90, receiver entropy
1.3673, `rx_explained_var` +0.025 (peak 0.443). Summed message-head value gradient
across all 405 iterations was exactly 0.0, so the flag held throughout. The
explained variance sits between DIAL's 0.080 and the control's 0.00005, as expected
for a code shaped by the actor alone. Food at +0.016 over the other two arms is inside
the placebo's drift range below. Saturation reached 0.976 with actor gradient down
to 0.00087, reproducing the vanishing-gradient pattern a third time.

### The channel strangles its own gradient

Not a sufficient explanation for the null, but a real mechanism worth recording. As
noise pressure drives the logits large — which is what makes hard execution lossless —
the sigmoid flattens and the pathwise derivative dies:

| saturation band | n | median message-head actor_grad |
|---|---:|---:|
| 0.0–0.1 | 41 | 0.002820 |
| 0.7–0.9 | 109 | 0.000360 |
| 0.9–1.0 | 129 | **0.000076** |

`corr(saturation, log10 actor_grad) = −0.555`, reproduced independently in the
critic-detach arm at −0.490. Analytically, `d/dz sigmoid(z)` is 0.235 at |z|=0.5 and
0.012 at |z|=4.4, the run's mean. **DIAL's gradient path decays as its discretization
succeeds** — the noise that buys a clean binary code is the same noise that kills the
derivative.

It is not the whole story. Saturation *peaked* mid-run and then declined
(0.103 → 0.891 → 0.861 → 0.642 → 0.564) and gradients recovered as it fell, yet
`rx_explained_var` dropped over that same stretch (0.138 → 0.028) with the critic
still dominating 5.4× at the end. Saturation-driven vanishing gradient is measured and
real; it does not by itself account for the flat task curve.

### An accidental placebo, and a caveat it converts into a measurement

§11 warned that rising `food_rate` is not evidence of learning. The detached arm
proves it. Its transmitter is provably frozen and its receiver critic provably learned
nothing, and its food rate still drifted **0.359 → 0.438** across the run — the same
shape §11 flagged (0.350 → 0.433). That calibrates the noise floor: DIAL's
0.338 → 0.411 drift sits **inside** the placebo's range. Any future claim from a
food-rate trend on this task has to clear about ±0.08 of drift in a system that cannot
possibly be communicating.

### Caveats

- **Single seed (0).** As everywhere in this document. One seed cannot establish that
  DIAL fails here, only that it did.
- **The three-seed probe gate was open** when the arena ran, failing on one seed at
  every sigma tested. The arena result should not be read as independent of that.
- **σ=0.5 only.** σ=2.0 was never run in the arena, and it is the setting with no
  train/execution gap on the toy probe.
- **Held-out evaluation outstanding.** These are training-time numbers. The
  prespecified measurement is `evaluate_communication.py` on frozen weights with
  `hard`/`constant`/`random` message ablations; identical food rates across those
  three would convert "the policy looks uniform" into "the messages provably carry no
  control signal."
- **`baseline_s0` was stopped at iteration 120** to free a GPU slot for the
  critic-detach arm. Over those 120 iterations its mean `rx_explained_var` was
  −0.0000579, reproducing §11; `transmitter_final.pt` was never written, so the
  iteration-100 checkpoint is its newest loadable weight set.
- **`critic_detach_s0` was stopped at iteration 405**, before completion; the
  iteration-400 checkpoint is its newest loadable weight set.
- **Arms are not wall-clock matched.** `critic_detach_s0` started ~3h after the others
  and saw different GPU contention. Seeding is per-process and deterministic, so this
  affects speed and not correctness, but a headline claim from this arm should be
  re-run as a clean matched pair.

### What this points to

The information is in the channel and the receiver's critic can decode it; the actor
never converts it into action. That pattern points away from the channel and toward
DIAL_PLAN §9, which flagged in advance a problem differentiability cannot remove: the
receiver's yaw is randomized every episode and observed by **neither** agent, and the
receiver has no recurrence and no previous-action input. A four-symbol code therefore
cannot map transmitter-frame directions onto receiver-frame moves without some
inferred calibration history. The next experiment should be that clearly-labelled
fixed/shared-yaw diagnostic, which separates a credit-assignment failure from an
unobservable task — not another channel variant. *(Run in §13: it did not help.)*

Sigma annealing (low for gradient, rising to force discretization) would address the
saturation finding directly. Flagged as a **hypothesis generated by this run**, not a
prespecified one, and it should be declared as such if pursued.

## 13. Shared-yaw diagnostic: orientation was not the blocker (2026-09-13, partial)

§12 pointed at DIAL_PLAN §9 as the likeliest remaining cause. The receiver's yaw was
randomized every episode and observed by neither agent, so a transmitter-frame code
could not map onto receiver-frame moves. This section removes that confound. **It
changed nothing.** These runs were **stopped at 61% of budget**, and the result below is
partial.

### Setup

`receiver_yaw_mode=shared` copies the transmitter's randomly sampled yaw into the
receiver at every reset, and both stay fixed for the episode. "Forward" now means the
same world direction for both agents. Both modes consume the same RNG draws. This
was checked directly: food layouts over five resets are byte-identical between
`shared` and `independent`. So these runs replay exactly the layout sequence the §12
runs saw, with only the receiver's orientation changed. All 18 mechanism tests passed
before launch, and a two-iteration end-to-end smoke run completed.

Seed 0, σ=0.5, 450 × 2,048 planned, all other settings as §12. Three arms ran
concurrently at 75–76 s/iteration:

| Run | Flags | Gradient receiver → transmitter | Verified at iteration 3 |
|---|---|---|---|
| `dial_shared_s0` | *(none)* | actor and critic | actor 0.00714, value 0.00116 |
| `critic_detach_shared_s0` | `critic_detach=1` | actor only | actor 0.00748, value **0** |
| `detached_shared_s0` | `detach_channel=1` | neither | actor **0**, value **0** |

### Why the runs are incomplete

All three stopped within 40 seconds of each other at about 09:56, at iterations 280,
276 and 276. None of the logs has a traceback. The kernel log shows no out-of-memory
kill, and the machine did not reboot. They were launched as background tasks of an
interactive Claude Code session, and they most likely ended when that session did.
**This was a launch error, not a training failure.** Long runs should be started
detached from the session (e.g. `setsid nohup`). Checkpoints exist through iteration
275. Final weights were never written, and `init_from` restores weights only, so the
runs cannot be resumed exactly.

### The result

The comparison covers the first 276 iterations. The §12 independent-yaw runs are truncated to the same
length, and the layout sequence is identical:

| arm | food | return | loose_death | length | rx_entropy | rx_ev | saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| DIAL, shared | 0.361 | −5.52 | 0.323 | 204.3 | 1.3723 | +0.0605 | 0.500 |
| critic-detach, shared | 0.373 | −5.20 | 0.321 | 202.2 | 1.3658 | +0.0143 | 0.312 |
| detached, shared | 0.368 | −5.63 | 0.328 | 199.7 | 1.3742 | −0.0000 | 0.000 |
| DIAL, independent | 0.357 | −5.50 | 0.321 | 205.1 | 1.3642 | +0.0701 | 0.714 |
| critic-detach, independent | 0.383 | −5.16 | 0.334 | 200.1 | 1.3668 | +0.0208 | 0.661 |
| detached, independent | 0.359 | −5.75 | 0.329 | 205.2 | 1.3713 | +0.0000 | 0.000 |

Last 50 iterations of that window (227–276):

| arm | food | return | rx_entropy | rx_ev |
|---|---:|---:|---:|---:|
| DIAL, shared | 0.379 | −5.60 | 1.3803 | +0.1165 |
| critic-detach, shared | 0.380 | −4.93 | 1.3699 | +0.0320 |
| detached, shared | 0.389 | −5.13 | 1.3825 | +0.0000 |
| DIAL, independent | 0.326 | −7.27 | 1.3409 | +0.1215 |
| critic-detach, independent | 0.407 | −4.88 | 1.3527 | +0.0455 |
| detached, independent | 0.338 | −6.49 | 1.3732 | +0.0000 |

- **Food rate:** all six arms fall between 0.357 and 0.383 over the matched window.
  That range is inside the ±0.08 drift that the frozen-transmitter placebo showed in §12.
  The shared-yaw DIAL arm is not ahead of its own detached control (0.361 vs 0.368).
- **The receiver still never commits.** Its largest entropy drop below `ln 4` in the
  shared-yaw arms is 0.064–0.066 nats. With independent yaw it is 0.064–0.117. Removing
  the orientation confound did not free the policy.
- **Information transfer is unchanged in character.** `rx_explained_var` follows the
  same pattern as §12: DIAL highest (0.060), critic-detach lower (0.014), and the detached
  control at exactly zero. The channel still carries what the critic can read and the
  actor does not use.
- **Saturation ran lower under shared yaw** (0.500 vs 0.714 for DIAL; 0.312 vs 0.661
  for critic-detach). It is the only systematic difference between the two conditions.
  Task behaviour shows no matching change.

### What this establishes, and what it doesn't

Three candidate explanations for the §11–§12 nulls have now each been tested separately:
the non-differentiable channel (§12), the critic's influence on the code (§12
`critic_detach`), and the receiver's unobserved orientation (here). None produced a
receiver that steers. What remains are properties of the receiver and the task that no
arm has changed. The receiver has no recurrence, no previous-action input, and only a
4-dimensional transformer over messages. The credit-assignment horizon is 300 steps.

### Caveats

- **Partial: 276–280 of 450 iterations** (~2,900 episodes per arm, 61% of budget).
  §12's arms showed no late departure over their final 40%, but that is not proof these
  would not have.
- **Single seed (0).**
- **Training-time numbers only.** No held-out evaluation has been run on any §12 or §13
  checkpoint.
- **σ=0.5 only**, as in §12.

## Artifacts

- DIAL (§12): `lib/CommunicationChannel.py`, `lib/DIALPPOTrainer.py`,
  `lib/CommunicationEvaluation.py`, `DIALTransmitter` in `lib/SymbolPolicy.py`,
  `distribution_bits`/`_bits_cls` in `lib/ReceiverPolicy.py`; scripts
  `evaluate_communication.py`, `probe_dial.py`,
  `train_communication.py method=dial_ppo|independent_ppo`; tests `tests/test_dial.py`;
  design docs [DIAL.md](DIAL.md), [DIAL_PLAN.md](DIAL_PLAN.md); mechanism-validation
  record `output/dial_validation.json`
- §12 runs: `output/runs/comm_20260912_162455_dial_s0/` (450 it),
  `output/runs/comm_20260912_162459_detached_s0/` (450 it, control),
  `output/runs/comm_20260912_192358_critic_detach_s0/` (stopped at 405 it),
  `output/runs/comm_20260912_162503_baseline_s0/` (stopped at 120 it)
- §13 runs (`receiver_yaw_mode=shared`, all stopped at ~276 it, checkpoints through
  275): `output/runs/comm_20260913_033319_dial_shared_s0/`,
  `output/runs/comm_20260913_033324_critic_detach_shared_s0/`,
  `output/runs/comm_20260913_033329_detached_shared_s0/`
- Communication phase (§11): `lib/ReceiverPolicy.py`, `lib/CommunicationTrainer.py`
  (`TransmitterInput`, `CommunicationTrainer`), `lib/KGWorldEnv.py`
  (`CommunicationKGWorldEnv`), `lib/VideoRecorder.py`
  (`record_communication_episode`, `segment_start_iters`);
  script `train_communication.py`
- §11 run: `output/runs/comm_20260829_205854_comm450_s0/` (450 it, 4,712 episodes)
- Oracle metrics: `experiments/oracle/oracle_distract_metrics*.jsonl`
- Gradient diagnostic: `experiments/oracle/grad_attribution.py`
- Real runs: `output/runs/ppo_2026072*_{xf,slot}_norm_*_s0/`
- Policy code: `lib/SymbolPolicy.py` (slot), `lib/TransformerPolicy.py` (transformer),
  `lib/PPOTrainer.py` (`normalize_value_loss`), `lib/KGWorldEnv.py` (`food_near_lion_*`)
- Soft-category (§7): `lib/ObjectCategories.py`, `lib/CategoryModel.py`,
  `lib/CategoryData.py`, `lib/OracleGrounding.py` (`perceive_scene_softcat`,
  `perceive_scene_oracle`); scripts `generate_category_dataset.py`,
  `train_category_model.py`, `evaluate_category_model.py`;
  `train_ppo.py perception=sam|softcat|oracle_id|oracle_full`
- §7 runs: `output/runs/ppo_20260817_192215_softcat_full_s0/` (0→300),
  `output/runs/ppo_20260818_022213_softcat_cont600_s0/` (300→600, `init_from`),
  `output/runs/ppo_20260817_175227_oraclefull_s0/`; category head
  `output/category_model.pt`, dataset `output/category_dataset.pt`
