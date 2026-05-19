"""
sweep_bias_baseline.py

Grid-searches the logit bias applied at inference to maximise Overall = (F1 + mIoU) / 2
on the validation set for the baseline model (no QueryMLP, input_dim_gt=2).

Usage (from /workspace inside the Docker container):
    python sweep_bias_baseline.py
"""

import os
import random

import cv2
import numpy as np
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── box utilities ──────────────────────────────────────────────────────────────

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=-1)


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt    = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb    = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh    = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union


def computeIOU(bb_pred, bb_label):
    return torch.diag(
        box_iou(box_cxcywh_to_xyxy(bb_pred),
                box_cxcywh_to_xyxy(bb_label)))


# ── dataset ────────────────────────────────────────────────────────────────────

class BuoyDataset(torch.utils.data.Dataset):
    def __init__(self, yaml_file, mode='val'):
        super().__init__()
        random.seed(0)
        torch.manual_seed(0)
        with open(yaml_file, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        self.data_path = data[mode]
        self.labels  = sorted(os.listdir(os.path.join(self.data_path, "labels")))
        self.images  = sorted(os.listdir(os.path.join(self.data_path, "images")))
        self.queries = sorted(os.listdir(os.path.join(self.data_path, "queries")))
        self.imus    = sorted(os.listdir(os.path.join(self.data_path, "imu")))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.data_path, "images", self.images[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        raw_labels = []
        with open(os.path.join(self.data_path, "labels", self.labels[index])) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if vals:
                    raw_labels.append(vals)
        raw_labels = (torch.tensor(raw_labels, dtype=torch.float32)
                      if raw_labels else torch.zeros((0, 5)))

        raw_queries = []
        with open(os.path.join(self.data_path, "queries", self.queries[index])) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if vals:
                    raw_queries.append(vals)
        queries = torch.tensor(raw_queries, dtype=torch.float32)

        raw_imu = []
        with open(os.path.join(self.data_path, "imu", self.imus[index])) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if vals:
                    raw_imu.append(vals)
        imu_data = (torch.tensor(raw_imu, dtype=torch.float32)
                    if raw_imu else torch.zeros((1, 10)))

        query_ids  = queries[:, 0].tolist()
        label_dict = {}
        if raw_labels.numel() > 0:
            for row in raw_labels:
                label_dict[row[0].item()] = row

        queries_mask   = torch.ones(len(query_ids),  dtype=torch.bool)
        labels_mask    = torch.zeros(len(query_ids), dtype=torch.bool)
        aligned_labels = torch.zeros(len(query_ids), 5, dtype=torch.float32)
        for i, qid in enumerate(query_ids):
            if qid in label_dict:
                aligned_labels[i] = label_dict[qid]
                labels_mask[i]    = True

        return img, queries, aligned_labels, queries_mask, labels_mask, imu_data


# ── collate — baseline (no QueryMLP) ──────────────────────────────────────────

def collate_fn(batch):
    imgs, queries, labels, q_masks, l_masks, imus = zip(*batch)

    new_imgs, proc_queries = [], []
    for img, q in zip(imgs, queries):
        q = q[..., 0:3].clone()   # [id, dist_m, bearing_deg]
        q[:, 1] = q[:, 1] / 1000.0   # normalize dist
        q[:, 2] = q[:, 2] / 180.0    # normalize bearing
        proc_queries.append(q)
        new_imgs.append(img / 255)

    images     = torch.stack(new_imgs, dim=0)
    pad_q      = pad_sequence(proc_queries, batch_first=True,
                              padding_value=0.0)[..., 1:]   # strip id
    pad_mask_q = pad_sequence(list(q_masks), batch_first=True,
                              padding_value=False)
    pad_l      = pad_sequence(list(labels),  batch_first=True,
                              padding_value=0.0)
    pad_mask_l = pad_sequence(list(l_masks), batch_first=True,
                              padding_value=False)

    model_inputs = {"images": images, "queries": pad_q,
                    "queries_mask": pad_mask_q}
    return model_inputs, pad_l, pad_mask_l, pad_mask_q


# ── model loader ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    from models.backbone import Backbone, Joiner
    from models.position_encoding import PositionEmbeddingSine
    from models.transformer import Transformer
    from models.detr import DETR

    hidden_dim = 256
    pe         = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backend    = Backbone('resnet50', True, False, False)
    backbone   = Joiner(backend, pe)
    backbone.num_channels = backend.num_channels

    transformer = Transformer(
        d_model=hidden_dim, dropout=0.1, nhead=8, dim_feedforward=2048,
        num_encoder_layers=6, num_decoder_layers=6,
        normalize_before=True, return_intermediate_dec=True,
    )
    model = DETR(backbone, transformer, input_dim_gt=2,
                 aux_loss=False, use_embeddings=False)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    return model.to(device).eval()


# ── collect raw logits ─────────────────────────────────────────────────────────

def collect(model, loader, device):
    all_logits, all_has_label, all_ious = [], [], []

    with torch.no_grad():
        for model_inputs, pad_l, pad_mask_l, pad_mask_q in tqdm(loader,
                                                                  desc="Collecting"):
            inputs  = {k: v.to(device) for k, v in model_inputs.items()}
            outputs = model(**inputs)

            logits     = outputs['pred_logits'].cpu()
            pred_boxes = outputs['pred_boxes'].cpu()
            labels     = pad_l.cpu()
            mask_q     = pad_mask_q.cpu()
            mask_l     = pad_mask_l.cpu()

            B, Q = logits.shape
            for b in range(B):
                for q in range(Q):
                    if not mask_q[b, q]:
                        continue
                    has_label = bool(mask_l[b, q].item())
                    all_logits.append(logits[b, q].item())
                    all_has_label.append(has_label)
                    if has_label:
                        iou = computeIOU(
                            pred_boxes[b, q].unsqueeze(0),
                            labels[b, q, 1:].unsqueeze(0)).item()
                        all_ious.append(iou)
                    else:
                        all_ious.append(0.0)

    return (np.array(all_logits,    dtype=np.float32),
            np.array(all_has_label, dtype=bool),
            np.array(all_ious,      dtype=np.float32))


# ── metric at a single bias value ──────────────────────────────────────────────

def eval_at_bias(logits, has_label, ious, bias,
                 conf_thresh=0.90, iou_thresh=0.5):
    eps        = 1e-6
    raw_logit  = np.log(logits.clip(eps, 1 - eps) /
                        (1 - logits.clip(eps, 1 - eps)))
    shifted    = 1 / (1 + np.exp(-(raw_logit + bias)))
    predicted  = shifted >= conf_thresh

    tp = fp = fn = 0
    iou_sum = tp_match = 0

    for i in range(len(logits)):
        pred = predicted[i]
        gt   = has_label[i]
        if pred and gt:
            iou_sum  += ious[i]
            tp_match += 1
            if ious[i] > iou_thresh:
                tp += 1
            else:
                fp += 1; fn += 1
        elif pred and not gt:
            fp += 1
        elif not pred and gt:
            fn += 1

    p       = tp / (tp + fp)       if (tp + fp)   > 0 else 0.0
    r       = tp / (tp + fn)       if (tp + fn)   > 0 else 0.0
    f1      = 2 * p * r / (p + r)  if (p + r)     > 0 else 0.0
    miou    = iou_sum / tp_match   if tp_match     > 0 else 0.0
    overall = (f1 + miou) / 2
    return p, r, f1, miou, overall


# ── sweep ──────────────────────────────────────────────────────────────────────

def sweep(logits, has_label, ious):
    print(f"\n{'Bias':>7} | {'Precision':>9} | {'Recall':>6} | "
          f"{'F1':>6} | {'mIoU':>6} | {'Overall':>7}")
    print("-" * 60)

    best_overall, best_bias = -1.0, 0.0
    for bias in np.arange(-3.0, 3.25, 0.25):
        p, r, f1, miou, overall = eval_at_bias(logits, has_label, ious, bias)
        marker = " ◄" if overall > best_overall else ""
        print(f"{bias:>7.2f} | {p:>9.4f} | {r:>6.4f} | "
              f"{f1:>6.4f} | {miou:>6.4f} | {overall:>7.4f}{marker}")
        if overall > best_overall:
            best_overall, best_bias = overall, float(bias)

    print("-" * 60)
    print(f"\n✓  Best bias    : {best_bias:.2f}")
    print(f"   Best Overall : {best_overall:.4f}")
    print(f"\n→  Set BIAS = {best_bias:.2f} in evaluate_baseline.py")
    return best_bias, best_overall


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    CHECKPOINT = "checkpoints_baseline/best.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = BuoyDataset(yaml_file='dataset.yaml', mode='val')
    loader  = DataLoader(dataset, batch_size=8,
                         sampler=torch.utils.data.SequentialSampler(dataset),
                         drop_last=False,
                         collate_fn=collate_fn,
                         num_workers=0)

    print(f"Loading model from {CHECKPOINT} ...")
    model = load_model(CHECKPOINT, device)

    print(f"Running on {len(dataset)} val samples ...")
    logits, has_label, ious = collect(model, loader, device)

    print(f"\nTotal valid queries : {len(logits)}")
    print(f"  with GT label     : {has_label.sum()}")
    print(f"  negatives         : {(~has_label).sum()}")
    print(f"Raw score stats     : min={logits.min():.3f}  max={logits.max():.3f}  "
          f"mean={logits.mean():.3f}  median={np.median(logits):.3f}")
    print(f"Scores >= 0.90      : {(logits >= 0.90).sum()} / {len(logits)}")

    sweep(logits, has_label, ious)
