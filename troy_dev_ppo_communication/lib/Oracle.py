"""Ground-truth oracle for phase-1 KG construction.

MuJoCo can render a segmentation image telling us exactly which geom painted
each pixel, so for every SAM detection we know the true object it came from.
That converts "did this new node actually refer to an object that already has
a node?" from a judgment call into a lookup, and lets us name the two failure
modes precisely:

  false_split  a detection created a NEW node, but a node whose true identity
               is the same object already existed -- the threshold was too high
               for this pair.
  false_merge  a detection was absorbed into an EXISTING node that stands for a
               DIFFERENT object -- the threshold was too low for this pair.

The oracle never feeds anything back into the KG; it only observes. Every
detection is also dumped (with its embedding) so clustering can be replayed
offline at other thresholds without re-running SAM -- see
scripts/eval_kg_oracle.py.
"""

import json
import os
from collections import Counter, defaultdict

import cv2
import mujoco
import numpy as np

BACKGROUND = "background"  # floor / skybox: no object in the mask
AMBIGUOUS = "ambiguous"  # mask straddles objects, no clear majority
_NON_OBJECT = (BACKGROUND, AMBIGUOUS)

VERDICT_OK = "ok"
VERDICT_SPLIT = "false_split"
VERDICT_MERGE = "false_merge"
VERDICT_UNMATCHED = "unmatched"  # phase 2 only: matched nothing
VERDICT_SKIPPED = "skipped"  # background/ambiguous detection, not scored


def geom_body_table(model):
    """Map every geom id to its body name (the object identity we score against).

    All 12 cage bars, the rim, and the base share body 'cage'; lion body/head/
    mane share body 'lion'. Body-level labels are therefore the right notion of
    "the same object" for KG nodes, while the geom name is kept alongside so
    part-level splinter can be told apart from viewpoint splinter.
    """
    body_of_geom = np.asarray(model.geom_bodyid, dtype=np.int64)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
        for i in range(model.nbody)
    ]
    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}"
        for i in range(model.ngeom)
    ]
    return body_of_geom, body_names, geom_names


def label_mask(seg, mask, table, min_coverage=0.5):
    """True object behind one SAM mask.

    seg:  HxWx2 int32 from KGWorldEnv.render_panorama(with_segmentation=True)
    mask: boolean HxW segmentation mask for one detection

    Returns (body_name, geom_name, coverage). Pixels belonging to the world body
    (floor, skybox) are treated as background, so a crop that is mostly floor
    with a small lion in it still labels as 'lion' provided the lion covers at
    least min_coverage of the mask.
    """
    body_of_geom, body_names, geom_names = table
    px = int(mask.sum())
    if px == 0:
        return BACKGROUND, BACKGROUND, 0.0

    objid = seg[..., 0][mask]
    objtype = seg[..., 1][mask]
    valid = (objid >= 0) & (objtype == int(mujoco.mjtObj.mjOBJ_GEOM))
    objid = objid[valid]
    if objid.size == 0:
        return BACKGROUND, BACKGROUND, 0.0

    geom_counts = np.bincount(objid, minlength=len(geom_names)).astype(np.float64)
    body_counts = np.bincount(
        body_of_geom, weights=geom_counts, minlength=len(body_names)
    )
    body_counts[0] = 0.0  # body 0 is 'world': floor and skybox
    top_body = int(np.argmax(body_counts))
    coverage = float(body_counts[top_body]) / px
    if body_counts[top_body] == 0:
        return BACKGROUND, BACKGROUND, 0.0
    if coverage < min_coverage:
        return AMBIGUOUS, AMBIGUOUS, coverage

    in_body = body_of_geom == top_body
    masked = np.where(in_body, geom_counts, 0.0)
    return body_names[top_body], geom_names[int(np.argmax(masked))], coverage


# ------------------------------------------------------------------ panels

_PANEL_H = 220
_CAPTION_H = 34
_HEADER_H = 26
_BG = (28, 28, 30)
_TEXT = (235, 235, 235)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fit(img, height=_PANEL_H):
    """Letterbox a BGR image into a fixed-height panel."""
    canvas = np.full((height, height, 3), _BG, np.uint8)
    if img is None or img.size == 0:
        cv2.putText(canvas, "(no exemplar)", (12, height // 2), _FONT, 0.45, _TEXT, 1)
        return canvas
    h, w = img.shape[:2]
    scale = min(height / h, height / w)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    rh, rw = resized.shape[:2]
    y0, x0 = (height - rh) // 2, (height - rw) // 2
    canvas[y0 : y0 + rh, x0 : x0 + rw] = resized
    return canvas


def error_panel(title, items):
    """Side-by-side comparison strip: [(caption, BGR image), ...] under a title."""
    panels = []
    for caption, img in items:
        panel = _fit(img)
        strip = np.full((_CAPTION_H, panel.shape[1], 3), _BG, np.uint8)
        for i, line in enumerate(caption.split("\n")[:2]):
            cv2.putText(strip, line[:34], (6, 14 + 15 * i), _FONT, 0.38, _TEXT, 1)
        panels.append(np.vstack([panel, strip]))
    body = np.hstack(panels)
    header = np.full((_HEADER_H, body.shape[1], 3), _BG, np.uint8)
    cv2.putText(header, title, (6, 18), _FONT, 0.45, _TEXT, 1)
    return np.vstack([header, body])


# ------------------------------------------------------------------ scoring


class NodeIdentityScorer:
    """The split/merge verdict rules, independent of rendering and file output.

    A node has no intrinsic identity -- it *is* the detections it absorbed -- so
    its true label is the majority ground-truth label over those detections.
    Every detection is judged against the node identities as they stood BEFORE
    that detection was absorbed, then folded in. The online oracle and the
    offline replay (scripts/eval_kg_oracle.py) share this class so their numbers
    are directly comparable.
    """

    def __init__(self):
        self.node_labels = defaultdict(Counter)  # node id -> true label histogram
        self.counts = Counter()
        self.label_sims = defaultdict(list)  # "same"/"diff" -> best_sim samples

    def node_label(self, node_id):
        hist = self.node_labels.get(node_id)
        return hist.most_common(1)[0][0] if hist else None

    def nodes_for_label(self, label):
        return [n for n in self.node_labels if self.node_label(n) == label]

    def judge(self, label, node_id, created, best_sim, sims):
        """Verdict for one detection: (verdict, rival_node, rival_sim).

        sims is this detection's similarity to every node as of match time
        (SymbolicKG.match_or_add), so the rival reading is the number the
        matcher actually saw, not a later re-measurement.
        """
        if label in _NON_OBJECT:
            return VERDICT_SKIPPED, None, None
        if node_id is None:
            return VERDICT_UNMATCHED, None, None

        def nearest(rivals):
            rivals = [n for n in rivals if n < len(sims)]
            if not rivals:
                return None, None
            best = max(rivals, key=lambda n: float(sims[n]))
            return best, float(sims[best])

        if created:
            # a node for this same real object may already exist -> splinter
            best, sim = nearest([n for n in self.nodes_for_label(label) if n != node_id])
            if best is None:
                return VERDICT_OK, None, None
            self.label_sims["same"].append(sim)
            return VERDICT_SPLIT, best, sim

        assigned = self.node_label(node_id)
        if assigned is None or assigned == label:
            self.label_sims["same"].append(float(best_sim))
            return VERDICT_OK, None, None

        # absorbed into a node that stands for a different object
        self.label_sims["diff"].append(float(best_sim))
        best, sim = nearest(self.nodes_for_label(label))
        return VERDICT_MERGE, best, sim

    def update(self, node_id, label, verdict):
        self.counts[verdict] += 1
        self.counts[f"label:{label}"] += 1
        if node_id is not None:
            self.node_labels[node_id][label] += 1

    # ------------------------------------------------------------ reporting

    def totals(self):
        return {
            "scored": sum(
                self.counts[v] for v in (VERDICT_OK, VERDICT_SPLIT, VERDICT_MERGE)
            ),
            "ok": self.counts[VERDICT_OK],
            "false_split": self.counts[VERDICT_SPLIT],
            "false_merge": self.counts[VERDICT_MERGE],
            "unmatched": self.counts[VERDICT_UNMATCHED],
            "skipped": self.counts[VERDICT_SKIPPED],
        }

    def label_groups(self, remap=None):
        """{true label: [node ids]} -- more than one node per label is splinter."""
        merged = defaultdict(Counter)
        for node, hist in self.node_labels.items():
            new = node if remap is None else remap.get(node)
            if new is not None:
                merged[new].update(hist)
        groups = defaultdict(list)
        for node in sorted(merged):
            groups[merged[node].most_common(1)[0][0]].append(node)
        return merged, groups

    def purity_lines(self, remap=None, header="-- NODE PURITY"):
        merged, groups = self.label_groups(remap)
        lines = [header]
        for node in sorted(merged):
            hist = merged[node]
            top, top_n = hist.most_common(1)[0]
            detail = ", ".join(f"{k}={v}" for k, v in hist.most_common(4))
            lines.append(
                f"  node {node:<3d} -> {top:<12s} purity "
                f"{top_n / max(1, sum(hist.values())):.2f}  "
                f"({sum(hist.values())} detections: {detail})"
            )
        lines += ["", "-- NODES PER REAL OBJECT (ideal: exactly 1)"]
        for label, nodes in sorted(groups.items()):
            flag = "" if len(nodes) == 1 or label in _NON_OBJECT else "   <-- SPLINTERED"
            lines.append(f"  {label:<12s} {len(nodes)} node(s): {nodes}{flag}")
        return lines

    def separability_lines(self):
        same, diff = self.label_sims["same"], self.label_sims["diff"]
        lines = ["-- SEPARABILITY (best-match similarity by ground truth)"]
        for name, vals in (("same object", same), ("different object", diff)):
            if vals:
                a = np.asarray(vals)
                lines.append(
                    f"  {name:<18s} n={a.size:<5d} mean {a.mean():+.3f}  "
                    f"p5 {np.percentile(a, 5):+.3f}  p50 {np.percentile(a, 50):+.3f}  "
                    f"p95 {np.percentile(a, 95):+.3f}"
                )
        if same and diff:
            overlap = float((np.asarray(same) < np.percentile(diff, 95)).mean())
            lines.append(
                f"  {overlap * 100:.1f}% of same-object matches fall below the 95th "
                "percentile of different-object matches"
            )
            lines.append(
                "  (a large overlap means NO single threshold separates them -- the "
                "embedding is the problem, not the threshold)"
            )
        return lines


# ------------------------------------------------------------------ oracle


class KGOracle:
    """Scores KG node assignments against MuJoCo ground truth.

    Call observe() once per perception frame; call close() to flush the
    detection log, the embedding matrix and the summary.
    """

    def __init__(
        self,
        model,
        out_dir,
        min_coverage=0.5,
        max_images=40,
        save_embeddings=True,
        verbose=False,
    ):
        self.table = geom_body_table(model)
        self.out_dir = out_dir
        self.min_coverage = min_coverage
        self.verbose = verbose
        self.save_embeddings = save_embeddings

        self.split_dir = os.path.join(out_dir, "false_split")
        self.merge_dir = os.path.join(out_dir, "false_merge")
        for d in (out_dir, self.split_dir, self.merge_dir):
            os.makedirs(d, exist_ok=True)
        self.detections_path = os.path.join(out_dir, "detections.jsonl")
        self.embeddings_path = os.path.join(out_dir, "embeddings.npy")
        self.summary_path = os.path.join(out_dir, "summary.txt")
        self._fh = open(self.detections_path, "w")

        # per-image budget so one error type cannot starve the other
        self._budget = {VERDICT_SPLIT: max_images // 2, VERDICT_MERGE: max_images // 2}

        self.scorer = NodeIdentityScorer()
        self.exemplars = {}  # node id -> (mask_px, BGR crop)
        self.records = []
        self._embeddings = []

    # -------------------------------------------------------------- helpers

    @property
    def node_labels(self):
        return self.scorer.node_labels

    @property
    def counts(self):
        return self.scorer.counts

    def node_label(self, node_id):
        """Majority true label of a node, from the detections it has absorbed."""
        return self.scorer.node_label(node_id)

    # -------------------------------------------------------------- observe

    def observe(self, kg, frame, seg, boxes, masks, embeddings, node_ids, info, context):
        """Score one camera frame's detections.

        frame/seg: aligned RGB and segmentation images
        boxes/masks/embeddings/node_ids/info: parallel per-detection lists, info
        entries carrying {"created", "best_sim"} from SymbolicKG.match_batch
        context: {"episode", "step", "cam", ...} merged into every record

        Embeddings are stashed in record order so row i of embeddings.npy is
        detection i of detections.jsonl.
        """
        if self.save_embeddings and len(embeddings):
            self._embeddings.append(embeddings.detach().float().cpu().numpy())

        for det, (box, mask, node_id) in enumerate(zip(boxes, masks, node_ids)):
            meta = info[det]
            label, geom, coverage = label_mask(seg, mask, self.table, self.min_coverage)

            verdict, rival, rival_sim = self.scorer.judge(
                label, node_id, meta["created"], meta["best_sim"], meta["sims"]
            )
            record = {
                "i": len(self.records),
                **context,
                "cam_det": det,
                "box": [int(v) for v in box],
                "mask_px": int(mask.sum()),
                "label": label,
                "geom": geom,
                "coverage": round(coverage, 4),
                "node": node_id,
                "node_label": self.node_label(node_id),
                "created": bool(meta["created"]),
                "best_sim": round(float(meta["best_sim"]), 5),
                "verdict": verdict,
                "rival_node": rival,
                "rival_sim": None if rival_sim is None else round(float(rival_sim), 5),
            }
            self.records.append(record)
            self._fh.write(json.dumps(record) + "\n")

            if verdict in (VERDICT_SPLIT, VERDICT_MERGE):
                self._save_error_image(record, frame, box, node_id, rival)
                if self.verbose:
                    print(
                        f"    ORACLE {verdict}: {label} det -> node {node_id} "
                        f"(node label {record['node_label']}), best_sim "
                        f"{record['best_sim']:+.3f}, rival node {rival} @ {record['rival_sim']}"
                    )

            # exemplar and label stats are updated AFTER judging, so a node is
            # always scored against what it stood for before this detection
            self.scorer.update(node_id, label, verdict)
            if node_id is not None:
                self._update_exemplar(node_id, frame, box, int(mask.sum()))

    def _update_exemplar(self, node_id, frame, box, mask_px):
        """Keep the largest-mask crop per node as its visual representative."""
        if mask_px <= self.exemplars.get(node_id, (0, None))[0]:
            return
        x0, y0, x1, y1 = box
        crop = frame[y0:y1, x0:x1]
        if crop.size:
            self.exemplars[node_id] = (
                mask_px,
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR).copy(),
            )

    def _save_error_image(self, record, frame, box, node_id, rival):
        verdict = record["verdict"]
        if self._budget.get(verdict, 0) <= 0:
            return
        self._budget[verdict] -= 1

        x0, y0, x1, y1 = box
        context_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(context_img, (x0, y0), (x1, y1), (0, 255, 255), 2)
        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2BGR).copy()

        items = [
            (f"new detection: {record['label']}\nbox {x0},{y0},{x1},{y1}", context_img),
            (f"...cropped\ncoverage {record['coverage']:.2f}", crop),
        ]
        if verdict == VERDICT_MERGE:
            title = (
                f"FALSE MERGE ep{record['episode']} step{record['step']} "
                f"cam{record['cam']}: {record['label']} absorbed into node "
                f"{node_id} ({record['node_label']}) at sim {record['best_sim']:+.3f}"
            )
            items.append(
                (
                    f"MERGED INTO node {node_id}\n{record['node_label']} "
                    f"@ sim {record['best_sim']:+.3f}",
                    self.exemplars.get(node_id, (0, None))[1],
                )
            )
            rival_caption = f"CORRECT node {rival}\n{record['label']} @ sim {record['rival_sim']}"
        else:
            title = (
                f"FALSE SPLIT ep{record['episode']} step{record['step']} "
                f"cam{record['cam']}: {record['label']} made new node {node_id}; "
                f"node {rival} is already {record['label']}"
            )
            rival_caption = (
                f"SHOULD HAVE MATCHED node {rival}\n{record['label']} "
                f"@ sim {record['rival_sim']}"
            )
        if rival is not None:
            items.append((rival_caption, self.exemplars.get(rival, (0, None))[1]))

        out_dir = self.merge_dir if verdict == VERDICT_MERGE else self.split_dir
        name = (
            f"ep{record['episode']:02d}_step{record['step']:03d}_cam{record['cam']}"
            f"_det{record['cam_det']}_{record['label']}_n{node_id}.png"
        )
        cv2.imwrite(os.path.join(out_dir, name), error_panel(title, items))

    # -------------------------------------------------------------- close

    def summary(self, kg=None):
        t = self.scorer.totals()
        lines = [
            "=" * 78,
            "KG ORACLE SUMMARY (MuJoCo segmentation ground truth)",
            "=" * 78,
            f"detections logged: {len(self.records)}   scored (real objects): {t['scored']}",
            f"  correct          : {t['ok']}",
            f"  false_split      : {t['false_split']}   "
            "(new node created for an object that already had one)",
            f"  false_merge      : {t['false_merge']}   "
            "(detection absorbed into a node for a different object)",
            f"  unmatched        : {t['unmatched']}",
            f"  skipped          : {t['skipped']}   "
            "(background / ambiguous masks, not scored)",
            "",
            "-- DETECTIONS BY TRUE OBJECT",
        ]
        for key, n in sorted(self.counts.items()):
            if key.startswith("label:"):
                lines.append(f"  {key[6:]:<12s} {n}")

        lines.append("")
        lines += self.scorer.purity_lines(
            header="-- NODE PURITY (a node should be one object; extra nodes = splinter)"
        )
        lines.append("")
        lines += self.scorer.separability_lines()
        if kg is not None:
            lines.append("")
            lines.append(
                f"  current threshold: {kg.similarity_threshold}   nodes: {kg.num_nodes}"
            )
        lines.append("=" * 78)
        return "\n".join(lines)

    def consolidation_report(self, remap):
        """Did consolidate() actually repair the splinter? Scores the FINAL nodes.

        remap is {old_id: new_id or None} as returned by SymbolicKG.consolidate.
        """
        lines = ["", "-" * 78]
        lines += self.scorer.purity_lines(
            remap=remap, header="-- AFTER CONSOLIDATION (final node identities)"
        )
        dropped = sorted(n for n in self.node_labels if remap.get(n) is None)
        if dropped:
            lines.append(f"  pruned pre-consolidation nodes: {dropped}")
        lines.append("-" * 78)
        return "\n".join(lines)

    def close(self, kg=None, remap=None):
        """Flush the log, embedding matrix and summary. Returns the summary text."""
        if not self._fh.closed:
            self._fh.close()
        if self.save_embeddings and self._embeddings:
            np.save(self.embeddings_path, np.concatenate(self._embeddings).astype(np.float16))
        text = self.summary(kg)
        if remap is not None:
            text += "\n" + self.consolidation_report(remap)
        with open(self.summary_path, "w") as f:
            f.write(text + "\n")
        return text
