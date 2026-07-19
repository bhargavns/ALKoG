import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms.functional as TF
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

_DEFAULT_SAM_CHECKPOINT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "sam_vit_b_01ec64.pth")
)

RELATION_NAMES = ["inside", "near"]

# Relation heuristic thresholds (image space)
INSIDE_OVERLAP_FRAC = 0.70  # fraction of A's box covered by B for "A inside B"
INSIDE_AREA_RATIO = 0.75  # A must be meaningfully smaller than B
NEAR_CENTER_FRAC = 0.30  # center distance below this fraction of image width


class BoxProposer:
    """Class-agnostic bounding box proposals via Segment Anything."""

    def __init__(
        self,
        checkpoint=_DEFAULT_SAM_CHECKPOINT,
        model_type="vit_b",
        device="cuda",
        min_area_frac=0.0005,
        max_area_frac=0.35,
    ):
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        sam.eval()
        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
            min_mask_region_area=64,
        )
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac

    def propose(self, frame):
        """frame: HxWx3 uint8 RGB. Returns (boxes, masks): XYXY int boxes and
        matching boolean HxW segmentation masks."""
        h, w = frame.shape[:2]
        img_area = float(h * w)
        boxes, seg_masks = [], []
        with torch.inference_mode():
            masks = self.generator.generate(frame)
        for m in masks:
            x, y, bw, bh = m["bbox"]  # XYWH
            frac = (bw * bh) / img_area
            if frac < self.min_area_frac or frac > self.max_area_frac:
                continue  # specks, floor, sky
            boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
            seg_masks.append(m["segmentation"])
        return boxes, seg_masks


class VisualEncoder:
    """Frozen ImageNet ResNet-18; 512-dim L2-normalized embedding per box crop."""

    OUTPUT_DIM = 512

    def __init__(self, device="cuda"):
        net = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net.to(device)
        self.device = device
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    @torch.inference_mode()
    def encode(self, frame, boxes, pad_frac=0.1):
        """frame: HxWx3 uint8 RGB; boxes: list of (x0,y0,x1,y1). Returns [N,512] tensor."""
        if not boxes:
            return torch.empty(0, self.OUTPUT_DIM, device=self.device)
        h, w = frame.shape[:2]
        img = torch.from_numpy(frame).to(self.device).permute(2, 0, 1).float() / 255.0
        crops = []
        for x0, y0, x1, y1 in boxes:
            px = int((x1 - x0) * pad_frac)
            py = int((y1 - y0) * pad_frac)
            cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
            cx1, cy1 = min(w, x1 + px), min(h, y1 + py)
            crop = img[:, cy0:cy1, cx0:cx1]
            crops.append(TF.resize(crop, [224, 224], antialias=True))
        batch = (torch.stack(crops) - self._mean) / self._std
        emb = self.net(batch)
        return F.normalize(emb, dim=1)


COLOR_HIST_BINS = 4  # per channel -> 64-dim histogram
COLOR_HIST_DIM = COLOR_HIST_BINS**3
EMBEDDING_DIM = VisualEncoder.OUTPUT_DIM + COLOR_HIST_DIM


def mask_color_histogram(frame, mask):
    """L2-normalized RGB histogram over the segmented object pixels only.

    Distant objects produce crops dominated by floor/sky background, which
    washes out the CNN features; the mask-restricted color histogram stays
    distinctive at any distance.
    """
    pixels = frame[mask]  # [N,3] uint8
    idx = (pixels // (256 // COLOR_HIST_BINS)).astype(np.int32)
    flat = idx[:, 0] * COLOR_HIST_BINS**2 + idx[:, 1] * COLOR_HIST_BINS + idx[:, 2]
    hist = np.bincount(flat, minlength=COLOR_HIST_DIM).astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


class PerceptionPipeline:
    """SAM proposals -> CNN features + mask color histogram per detection.

    Embedding = [sqrt(.5)*CNN(crop), sqrt(.5)*colorhist(mask)], both parts
    unit-norm, so cosine similarity = 0.5*cnn_sim + 0.5*color_sim.
    """

    def __init__(self, device="cuda", sam_checkpoint=_DEFAULT_SAM_CHECKPOINT):
        self.proposer = BoxProposer(checkpoint=sam_checkpoint, device=device)
        self.encoder = VisualEncoder(device=device)
        self.device = device

    def detect(self, frame):
        """Returns (boxes, embeddings): XYXY boxes and [N, EMBEDDING_DIM] tensor."""
        boxes, masks = self.proposer.propose(frame)
        cnn = self.encoder.encode(frame, boxes)
        if not boxes:
            return boxes, torch.empty(0, EMBEDDING_DIM, device=self.device)
        hists = torch.stack(
            [
                torch.from_numpy(mask_color_histogram(frame, m)).to(self.device)
                for m in masks
            ]
        )
        w = float(np.sqrt(0.5))
        return boxes, torch.cat([w * cnn, w * hists], dim=1)


def union_boxes_by_concept(concept_ids, boxes):
    """Group same-frame detections by matched concept; union their boxes.

    SAM often returns several masks for one object (individual cage bars, lion
    body vs head). All of them match the same concept node, so the union box is
    the object's true extent -- which is what the relation heuristics need.
    Returns {concept_id: (x0, y0, x1, y1)}.
    """
    grouped = {}
    for cid, (x0, y0, x1, y1) in zip(concept_ids, boxes):
        if cid is None:
            continue
        if cid in grouped:
            gx0, gy0, gx1, gy1 = grouped[cid]
            grouped[cid] = (min(gx0, x0), min(gy0, y0), max(gx1, x1), max(gy1, y1))
        else:
            grouped[cid] = (x0, y0, x1, y1)
    return grouped


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def relations_from_boxes(concept_boxes, image_width):
    """Geometric relation heuristics over one frame's concept->box map.

    'inside': most of A's box lies within B's box and A is smaller than B.
    'near':   box centers are close and neither contains the other.
    Returns list of (src_concept, relation_name, dst_concept). 'near' is
    emitted once per pair with src < dst.
    """
    relations = []
    ids = sorted(concept_boxes.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            box_a, box_b = concept_boxes[a], concept_boxes[b]
            area_a, area_b = _area(box_a), _area(box_b)
            if area_a == 0 or area_b == 0:
                continue
            inter = _intersection(box_a, box_b)
            if inter / area_a >= INSIDE_OVERLAP_FRAC and area_a <= INSIDE_AREA_RATIO * area_b:
                relations.append((a, "inside", b))
                continue
            if inter / area_b >= INSIDE_OVERLAP_FRAC and area_b <= INSIDE_AREA_RATIO * area_a:
                relations.append((b, "inside", a))
                continue
            ca = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
            cb = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
            dist = np.hypot(ca[0] - cb[0], ca[1] - cb[1])
            if dist < NEAR_CENTER_FRAC * image_width:
                relations.append((a, "near", b))
    return relations
