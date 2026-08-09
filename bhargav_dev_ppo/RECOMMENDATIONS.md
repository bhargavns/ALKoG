# Perception, Memory, and PPO: Recommendation Document

> Update: the executable five-category supervised soft-slot experiment is in
> [SOFT_CATEGORY_QUICKSTART.md](SOFT_CATEGORY_QUICKSTART.md). This document
> describes the earlier three-object hard-KG baseline and remains useful as
> design history.

## Purpose

This document explains the current `troy_dev_ppo` system, why performance fell after removing direct object-distance inputs, and a practical sequence of changes to make the visual pipeline more reliable.

The objective is **not** to return direct simulator distances to the PPO policy. Instead, the policy should act from image-derived object evidence, visual geometry, and an explicitly uncertainty-aware memory.

---

## 1. Problem statement

The earlier PPO input contained direct distances to food, lion, and cage. Direct distances tell the policy exactly how far each object is, even if the object is not visible. This is useful as a debugging baseline, but it bypasses perception.

The current system removes direct object positions from the phase-2 environment observation. PPO must now determine what is present and where it is through this chain:

1. Render four agent-mounted camera images.
2. Propose object masks and bounding boxes with Segment Anything Model (SAM).
3. Encode each proposed object crop with ResNet-18 plus a color histogram.
4. Match the visual encoding to a stored knowledge-graph (KG) concept.
5. Estimate a ground-plane offset from the bounding box.
6. Convert recognized objects and relations into a fixed PPO input vector.
7. Preserve object locations across missed detections with anchor memory.

Performance falls whenever this chain misses an object, gives it the wrong identity, estimates the wrong location, or retains a stale/incorrect location in memory.

---

## 2. Definitions

### 2.1 PPO

**Proximal Policy Optimization (PPO)** is the reinforcement-learning algorithm that updates a policy while limiting how much the policy is allowed to change after each batch of experience.

- **Policy:** a function that maps an input state to a probability distribution over actions.
- **Actor:** the neural-network part that chooses action probabilities.
- **Critic:** the neural-network part that estimates expected future reward for the current state.
- **Reward:** scalar feedback from the environment. In this environment, reaching food is positive and being caught by a loose lion is negative.

### 2.2 Observation / environment vector

An **observation** is the numeric input supplied by the environment at each step. The phase-2 environment supplies only four values:

$$
\mathbf{o}_t = [x_t/8,\ y_t/8,\ \cos(\theta_t),\ \sin(\theta_t)]
$$

Definitions:

- $t$: the current time step.
- $x_t, y_t$: agent position in world coordinates at time $t$.
- $\theta_t$: agent yaw, or horizontal facing angle.
- $\cos(\theta_t), \sin(\theta_t)$: a continuous representation of angle that avoids the discontinuity between $-\pi$ and $\pi$.
- `/8`: a normalization scale. Dividing values by a fixed number keeps inputs closer to a convenient numerical range for neural-network training.

The phase-2 observation does **not** contain lion, food, or cage coordinates. See [lib/KGWorldEnv.py](lib/KGWorldEnv.py#L200-L236).

### 2.3 Egocentric coordinates

**Egocentric** means expressed relative to the agent's current facing direction.

For an object location relative to the agent, the system uses:

$$
\mathbf{d}_i = [d_{i,f}, d_{i,l}]
$$

Definitions:

- $i$: an object or concept.
- $d_{i,f}$: forward distance component. Positive means in front of the agent.
- $d_{i,l}$: left distance component. Positive means left of the agent.

This is a good coordinate system for actions such as `forward`, `backward`, `left`, and `right`: an object ahead should look the same to the policy regardless of the world yaw.

### 2.4 First-person panorama

A **panorama** here is not one stitched image. It is the collection of four camera frames:

- `cam_front`
- `cam_left`
- `cam_back`
- `cam_right`

The four frames cover directions around the agent. The frames are rendered in [lib/KGWorldEnv.py](lib/KGWorldEnv.py#L186-L195).

### 2.5 Segmentation mask and bounding box

A **segmentation mask** is a pixel-level yes/no image indicating pixels thought to belong to one visual region.

A **bounding box** is the smallest rectangle containing the region, stored as:

$$
(x_0, y_0, x_1, y_1)
$$

Definitions:

- $x_0, y_0$: top-left pixel coordinate.
- $x_1, y_1$: bottom-right pixel coordinate.
- **Box area:** $(x_1-x_0)(y_1-y_0)$.

SAM produces class-agnostic masks. That means it proposes regions without deciding whether the region is food, lion, cage, floor, or a cage bar. Proposal filtering is implemented in [lib/Perception.py](lib/Perception.py#L15-L59).

### 2.6 Visual embedding

A **visual embedding** is a vector of numbers describing an image crop. Similar-looking crops should have similar vectors.

Each proposal currently receives:

1. A 512-dimensional ResNet-18 crop embedding.
2. A 64-dimensional color-histogram embedding over masked pixels.

They are concatenated into a 576-dimensional vector:

$$
\mathbf{e} = [\sqrt{0.5}\,\mathbf{e}_{\mathrm{cnn}}\ \Vert\ \sqrt{0.5}\,\mathbf{e}_{\mathrm{color}}]
$$

Definitions:

- $\mathbf{e}_{\mathrm{cnn}}$: the normalized ResNet-18 feature vector.
- $\mathbf{e}_{\mathrm{color}}$: the normalized color-histogram feature vector.
- $\Vert$: vector concatenation, meaning append one vector after another.
- $
\sqrt{0.5}$: equal weighting so CNN appearance and masked color each contribute half of cosine similarity.

The combined embedding is created in [lib/Perception.py](lib/Perception.py#L87-L144).

### 2.7 Knowledge graph (KG), concept, and cosine similarity

The **knowledge graph** stores visual concepts discovered during phase 1.

- **Concept:** one learned cluster of visually similar detections. A concept is not guaranteed to be a semantic class such as “lion”; it is only a cluster that the visual matcher considers similar.
- **Node:** a concept stored in the KG.
- **Relation edge:** a recorded relation between two concept nodes, such as `inside` or `near`.
- **Cosine similarity:** a score measuring how aligned two normalized vectors are. For normalized embeddings:

$$
\operatorname{cosine}(\mathbf{a},\mathbf{b}) = \mathbf{a}^\top \mathbf{b}
$$

A value near $1$ means strong similarity; a lower value means less similarity. A detection is matched only when its best score is at least the stored threshold, currently $0.80$. See [lib/SymbolicKG.py](lib/SymbolicKG.py#L18-L56).

### 2.8 Relation heuristic

A **relation heuristic** is a manually specified geometric rule rather than a learned relation classifier.

Current relations:

- `inside`: one bounding box substantially overlaps and is smaller than another box.
- `near`: the two bounding-box centers are close in image space.

These rules operate on 2D image boxes, not true 3D geometry. See [lib/Perception.py](lib/Perception.py#L177-L205).

### 2.9 Ground-plane grounding

**Grounding** converts a visual bounding box into an estimated location relative to the agent.

The code casts a ray through the bottom-center of a detected bounding box and intersects it with the ground plane. It returns the agent-to-object offset:

$$
\hat{\mathbf{d}}_i^{\mathrm{world}} = [\hat{\Delta x}_i, \hat{\Delta y}_i]
$$

Definitions:

- `hat` ($\hat{}$): an estimate, not ground truth.
- **World frame:** fixed global $x,y$ coordinates used by the simulator.
- **Ground plane:** the flat floor at height $z=0$.
- **Ray casting:** extending a line from a camera pixel into the 3D scene.

The estimate relies on the correctness of the bounding box. A partial box, a cage-bar box, or a box at the horizon can produce an inaccurate offset. See [lib/KGWorldEnv.py](lib/KGWorldEnv.py#L241-L277).

### 2.10 Symbol and triple

A **symbol** is a trainable 4-dimensional vector associated with a KG node or relation. It is a compact learned representation, not a human-readable label.

A **triple** describes a relation in the form:

$$
(\text{source concept},\ \text{relation},\ \text{destination concept})
$$

Examples:

- `(lion, inside, cage)`
- `(food, PAD, PAD)` for a lone detected food concept without a relation.

`PAD` is a padding marker used when a relation or entire triple slot is absent. The system reserves up to six triple slots per observation. Triple assembly is in [lib/Grounding.py](lib/Grounding.py#L117-L166).

### 2.11 Anchor memory

**Anchor memory** is the current mechanism for remembering a visual detection after a later perception pass misses it.

When concept $i$ is seen, the system combines the current normalized agent position and estimated world-frame offset:

$$
\mathbf{a}_i = \hat{\mathbf{d}}_i^{\mathrm{world}} + [x_t/8, y_t/8]
$$

Definitions:

- $\mathbf{a}_i$: a stored world-frame anchor for concept $i$.
- The object is assumed static within an episode.

At every later step, it recomputes the object's agent-relative offset from the stored anchor and rotates it into the egocentric frame. This prevents the distance vector from becoming stale merely because the agent moved. See [lib/Grounding.py](lib/Grounding.py#L168-L240).

**Limitation:** memory retains a point estimate only. It does not store whether the point came from a confident live detection, a weak detection, or a long-ago detection.

### 2.12 Confidence, freshness, age, and camera agreement

These terms are recommended additions:

- **Confidence:** a numerical estimate of how trustworthy a detection is. A first candidate is the visual cosine-match score.
- **Freshness:** whether the object was detected in the latest perception pass rather than recalled from memory.
- **Age:** number of environment steps since the object was last directly detected.
- **Camera agreement:** number of camera views that support a compatible object identity/location.
- **Uncertainty:** an estimate of how inaccurate the stored object position might be. It can grow as age increases and shrink when corroborated by high-quality visual observations.

---

## 3. How the current policy vector is built

The policy receives a fixed 76-dimensional vector:

$$
\mathbf{z}_t = [\mathbf{o}_t\ \Vert\ \mathbf{g}_t]
$$

Definitions:

- $\mathbf{z}_t$: complete PPO input at time $t$.
- $\mathbf{o}_t$: 4-dimensional environment observation.
- $\mathbf{g}_t$: 72-dimensional flattened KG representation.

### 3.1 Agent component: 4 values

$$
\mathbf{o}_t = [x_t/8,\ y_t/8,\ \cos(\theta_t),\ \sin(\theta_t)]
$$

### 3.2 KG component: 6 triple slots x 3 tokens x 4 values

The KG component has up to $K=6$ triples. Every triple has three token positions:

1. source concept,
2. relation,
3. destination concept.

Each token becomes a 4-dimensional vector, yielding:

$$
6 \times 3 \times 4 = 72
$$

For each concept token, its 4-dimensional symbol is added to a token-position embedding and multiplied elementwise by a learned projection of the egocentric offset:

$$
\mathbf{v}_{i} = (\mathbf{s}_{i} + \mathbf{p}_{\mathrm{slot}}) \odot f(\mathbf{d}_i)
$$

Definitions:

- $\mathbf{v}_i$: encoded node token.
- $\mathbf{s}_i$: learned 4-dimensional KG symbol for concept $i$.
- $\mathbf{p}_{\mathrm{slot}}$: learned 4-dimensional embedding indicating source, relation, or destination position.
- $\odot$: elementwise multiplication.
- $f$: a learned linear map from 2D egocentric offset to four values.

Relation tokens receive symbol plus position embedding but are not distance-gated.

The three tokens are concatenated, then all six triple slots are concatenated and appended to the agent vector. See [lib/SymbolPolicy.py](lib/SymbolPolicy.py#L14-L85).

### 3.3 Important current limitations

1. **Fixed capacity:** only six triple entries are retained.
2. **Fixed ordering:** triples are sorted and flattened; when detections appear/disappear, semantic information can shift between positions.
3. **No explicit confidence:** the policy cannot distinguish a high-confidence live detection from a low-confidence one.
4. **No explicit age/freshness:** memory and current perception look almost identical to the policy.
5. **No position uncertainty:** every anchor is treated as an exact location.
6. **No learned temporal state:** the MLP has no recurrent hidden state that can summarize a sequence of observations.
7. **Relation ambiguity:** `inside` is inferred from 2D bounding-box overlap, which is especially fragile for the lion/cage decision.

---

## 4. Recommended implementation plan

The plan intentionally starts with measurement and small representation changes. Do not redesign PPO before identifying whether the primary failure is missed proposals, bad concept matching, bad relation inference, or bad ground geometry.

### Step 1 — Add perception-quality diagnostics

**Goal:** locate the failure stage objectively.

For each physical object (food, lion, cage) at every perception pass, log:

- whether it is visible in each camera;
- whether SAM proposed a box for it;
- matched KG concept ID;
- visual cosine confidence;
- union-box area;
- camera name;
- estimated ground offset;
- true simulator offset, **only in offline diagnostic logs**;
- offset error, defined as estimated offset minus true offset;
- whether the policy input used a live detection or memory;
- anchor age.

For lion/cage relations, log predicted `inside`/`near` and the true caged/loose state, only for evaluation.

**Why:** without this data, changing thresholds or architecture is guesswork.

**Success criteria:** produce per-object recall, concept consistency, average grounding error, and lion-caged relation accuracy plots.

### Step 2 — Improve proposal recall before PPO tuning

**Goal:** ensure food, lion, and cage have usable proposals often enough.

The current proposal filter discards masks outside the configured bounding-box area range. The minimum fraction is `0.0005`; see [lib/Perception.py](lib/Perception.py#L24-L59).

Run a small controlled sweep, for example:

- lower minimum area threshold;
- vary SAM points-per-side;
- vary SAM confidence/stability thresholds;
- compare the number of accepted proposals with per-object recall from Step 1.

**Do not choose settings only because they create more masks.** Choose settings that improve recall for real objects without producing an unusably large number of floor/background fragments.

**Success criteria:** determine whether object misses are primarily proposal failures or downstream matching failures.

### Step 3 — Preserve confidence and freshness in anchor memory

**Goal:** allow the policy and memory update logic to distinguish reliable evidence from weak/stale evidence.

Extend every stored concept anchor with:

$$
[\text{world position},\ \text{confidence},\ \text{age},\ \text{freshness},\ \text{camera count},\ \text{uncertainty}]
$$

Recommended behavior:

1. Set `age = 0` for a live direct detection.
2. Increase `age` every environment step without a direct detection.
3. Set `freshness = 1` for a live detection and `0` for memory-only recall.
4. Store the best cosine match as a confidence signal.
5. Count compatible detections across cameras as camera agreement.
6. Increase uncertainty as anchor age grows.
7. Update an existing anchor only when new evidence is sufficiently confident and not geometrically inconsistent.
8. Expire anchors that become too old or too uncertain.

**Why:** current memory helps with intermittent SAM misses, but it can also preserve a wrong first observation for an entire episode.

**Success criteria:** PPO receives explicit freshness/age/confidence fields, and low-confidence estimates are no longer treated identically to new high-confidence detections.

### Step 4 — Replace the current flat triple vector with object-centric slots

**Goal:** make the scene representation clearer, more stable, and easier for PPO to use.

Recommended first redesign: retain a fixed maximum number of objects, but encode one slot per object/concept rather than relying exclusively on flattened relation triples.

An object slot could be:

$$
\mathbf{q}_i = [\mathbf{s}_i\ \Vert\ d_{i,f}\ \Vert\ d_{i,l}\ \Vert\ c_i\ \Vert\ a_i\ \Vert\ r_i\ \Vert\ m_i]
$$

Definitions:

- $\mathbf{q}_i$: object slot for concept $i$.
- $\mathbf{s}_i$: learned concept symbol or embedding.
- $d_{i,f}, d_{i,l}$: egocentric forward/left offset.
- $c_i$: visual confidence.
- $a_i$: normalized anchor age.
- $r_i$: freshness flag (`1` for live, `0` for memory-only).
- $m_i$: camera-agreement count or normalized agreement score.

Use separate relation slots:

$$
\mathbf{r}_{ij} = [\mathbf{s}_i\ \Vert\ \mathbf{s}_{\mathrm{relation}}\ \Vert\ \mathbf{s}_j\ \Vert\ c_{ij}]
$$

Definitions:

- $\mathbf{r}_{ij}$: relation slot involving concepts $i$ and $j$.
- $\mathbf{s}_{\mathrm{relation}}$: learned relation symbol.
- $c_{ij}$: relation confidence.

Append encoded object and relation slots to the agent vector.

**Why:** this separates “what/where is an object?” from “what relation may hold between two objects?” It also makes stale memory visible instead of hiding it inside an otherwise identical symbol token.

**Success criteria:** stable object slots across perception passes, clear debug printouts, and policy input fields that can be inspected directly.

### Step 5 — Validate lion/cage relation separately

**Goal:** establish whether `lion inside cage` can be inferred reliably enough to guide safety behavior.

Current `inside` uses 2D box overlap. Evaluate it as a binary classifier:

- predicted: lion is `inside` cage according to box heuristic;
- actual: environment's caged/loose state, used offline only;
- metrics: precision, recall, false-positive rate, false-negative rate.

Definitions:

- **True positive:** caged lion correctly predicted as caged.
- **False positive:** loose lion incorrectly predicted as caged. This is especially dangerous because it encourages unsafe behavior.
- **False negative:** caged lion predicted as loose. This is safer but can reduce food-seeking behavior.

If accuracy is poor, use one of these options:

1. Train a small supervised caged-versus-loose classifier on simulator-rendered lion/cage crops.
2. Use multi-view agreement before accepting an `inside` relation.
3. Treat uncertain relations as unknown and train a conservative behavior around unknown lion state.

**Success criteria:** a caged/loose decision with known error rates. Do not expect PPO to solve a relation that is systematically mislabeled.

### Step 6 — Upgrade memory to a simple belief state

**Goal:** represent location uncertainty rather than one supposedly exact point.

For each concept store:

$$
\mathbf{b}_i = [\mu_f, \mu_l, \sigma_f, \sigma_l, c_i, a_i]
$$

Definitions:

- $\mathbf{b}_i$: belief state for concept $i$.
- $\mu_f, \mu_l$: best estimated egocentric location.
- $\sigma_f, \sigma_l$: uncertainty in forward and left location. Larger values mean less certainty.
- $c_i$: confidence.
- $a_i$: age.

A lightweight first implementation can:

- use high-confidence observations to reduce $\sigma$;
- use low-confidence observations only when close to the old estimate;
- increase $\sigma$ when an anchor is not re-observed;
- expire anchors once uncertainty exceeds a threshold.

**Why:** this directly addresses noisy box geometry and intermittent detection without revealing oracle object positions.

**Success criteria:** visibly incorrect anchors decay instead of remaining equally influential throughout an episode.

### Step 7 — Consider attention-based scene encoding

**Goal:** remove dependence on fixed flattened slot order.

An **attention encoder** lets the policy process a set of object and relation tokens without assuming that token 0 always has a fixed meaning.

For each object token:

$$
\mathbf{u}_i = [\mathbf{s}_i\ \Vert\ d_{i,f}\ \Vert\ d_{i,l}\ \Vert\ c_i\ \Vert\ a_i\ \Vert\ r_i]
$$

Feed valid object/relation tokens to a small Transformer or set-attention module, then pool them into one scene vector. Append the scene vector to agent state before actor and critic heads.

Definitions:

- **Token:** one structured object or relation entry.
- **Attention:** a learned mechanism that weighs the relevance of one token against others.
- **Pooling:** converting a variable-size token set into one fixed-size scene vector.
- **Mask:** a marker telling the encoder which slots are padding and must be ignored.

**Why:** the current flattening in [lib/SymbolPolicy.py](lib/SymbolPolicy.py#L32-L57) can make the input change abruptly when triples are sorted differently or disappear.

**Success criteria:** policy performance becomes less sensitive to detection order and the number of detected relations.

### Step 8 — Add recurrent memory only if needed

**Goal:** let the policy learn temporal patterns that structured anchors do not capture.

A **recurrent neural network (RNN)** carries a hidden state from one step to the next. A practical choice is a gated recurrent unit (GRU):

$$
\mathbf{h}_t = \operatorname{GRU}(\text{scene}_t, \mathbf{h}_{t-1})
$$

Definitions:

- $\mathbf{h}_t$: learned hidden-memory vector at time $t$.
- `GRU`: a recurrent unit designed to retain useful history and forget irrelevant history.
- `scene_t`: the current structured scene encoding.

Use the GRU output for actor and critic predictions.

**Important:** do this after Steps 1–6. A recurrent policy can hide poor perception in a hard-to-debug hidden state, but it cannot reliably correct a detector that consistently gives the wrong object identity.

---

## 5. Recommended order of work

1. **Measure:** implement Step 1 diagnostics and inspect saved visual overlays.
2. **Fix object availability:** run Step 2 proposal-recall experiments.
3. **Make evidence explicit:** implement Step 3 confidence, freshness, age, camera agreement, and anchor expiry.
4. **Improve the representation:** implement Step 4 object-centric and relation-centric slots.
5. **Protect the core safety predicate:** complete Step 5 lion/cage relation evaluation and replace the heuristic if necessary.
6. **Handle geometry uncertainty:** implement Step 6 belief-state anchors.
7. **Remove ordering fragility:** consider Step 7 attention encoding if fixed slots remain unstable.
8. **Add learned temporal memory last:** consider Step 8 GRU only after the visual and structured memory signals are trustworthy.

---

## 6. What not to do

- Do not return direct `dist_to_food`, `dist_to_lion`, or `dist_to_cage` values to PPO as a production solution.
- Do not tune PPO hyperparameters before measuring perception failure rates.
- Do not treat the number of SAM masks as a success metric; use object recall and correct concept identity.
- Do not allow a weak/old anchor to look identical to a fresh high-confidence detection.
- Do not assume a 2D bounding-box containment rule is a reliable proxy for the lion's true cage state without evaluation.

---

## 7. Initial experiment checklist

For the next short experimental cycle:

1. Produce one phase-1 diagnostic run with overlays and per-object metrics.
2. Test three proposal-filter configurations while holding seed/layouts constant.
3. Choose the configuration with the best real-object recall/false-positive tradeoff.
4. Add `confidence`, `age`, `freshness`, and `camera agreement` to stored anchors and PPO input.
5. Re-run a short PPO training run with perception every 15–30 steps rather than 75 steps.
6. Compare food rate, death rate, caged-food rate, loose-food rate, and perception metrics side by side.
7. Only then choose between a simple object-slot MLP redesign and an attention encoder.
