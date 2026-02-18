from collections import defaultdict
import json
import os
import random
import time

import numpy as np

from tqdm import tqdm

import cv2
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops.boxes import box_area
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import yaml


# -----------------------------------------------------------------------------------
# Replace the following functions with your own implementations. Add imports inside the function definitions as needed.
# Please do not change the function inputs
# See evaluate_example.py at github.com/mkaraaslan-dev/CVPR2026-Transformer for an example implementation

def get_model():
    # this function should return your usable pytorch model
    from models.detr import DETR
    from models.transformer import Transformer
    from models.backbone import Backbone, Joiner
    from models.position_encoding import PositionEmbeddingSine

    path_to_weights = r"path/to/weights.pth"

    hidden_dim = 256  # embedding dim
    enc_layers = 6  # encoding layers
    dec_layers = 6  # decoding layers
    dim_feedforward = 2048  # dim of ff layers in transformer layers
    dropout = 0.1
    nheads = 8  # transformear heads
    pre_norm = True  # apply norm pre or post tranformer layer
    use_embeddings = False
    lr_backbone = 0.1

    position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    return_interm_layers = False
    backend = Backbone('resnet50', lr_backbone > 0, return_interm_layers, False)
    backbone = Joiner(backend, position_embedding)
    backbone.num_channels = backend.num_channels

    transformer = Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )

    model = DETR(
        backbone,
        transformer,
        input_dim_gt=2,
        aux_loss=False,
        use_embeddings=use_embeddings,
    )

    checkpoint = torch.load(path_to_weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)

    return model


def input_collate_fn(img, queries, queries_mask, imu_data):
    # this function takes (unprocessed) input data, and should return input that can be fed into your model s.t.
    # outputs = model(**input_collate_fn(img, queries, queries_mask, imu_data))

    new_imgs = []
    for batch_img, batch_queries in zip(img, queries):
        batch_queries[..., 1] = batch_queries[..., 1] / 1000  # normalize dist between 0-1 (gets clamped later to max 1)
        batch_queries[..., 2] = batch_queries[..., 2] / 180  # normalize angle between -1 to 1
        new_imgs.append(batch_img / 255)

    img = torch.stack(new_imgs, dim=0)
    pad_q = pad_sequence(queries, batch_first=True, padding_value=0.0)[...,
            1:]  # remove index from queries (only for debugging reasons)
    pad_mask_q = pad_sequence(queries_mask, batch_first=True, padding_value=False)

    return {"images": img, "queries": pad_q, "queries_mask": pad_mask_q}


# Please do not change anything other than get_model and input_collate_fn in this script.
# During evaluation, we will copy over these two functions, and only these two, into our own evaluation.py script
# It is the sole responsibility of the participants to ensure that this evaluation script works following the above convention.
# -----------------------------------------------------------------------------------


class BuoyDataset(Dataset):
    def __init__(self, yaml_file, mode='train') -> None:
        # mode: train/test/val
        super().__init__()

        self.yaml_file = yaml_file
        if mode in ["train", "test", "val"]:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode ({mode}) for DataSet")
        self.data_path = None

        random.seed(0)
        torch.manual_seed(0)
        self.processYAML()

        self.labels = sorted(os.listdir(os.path.join(self.data_path, "labels")))
        self.images = sorted(os.listdir(os.path.join(self.data_path, "images")))
        self.queries = sorted(os.listdir(os.path.join(self.data_path, "queries")))
        self.imus = sorted(os.listdir(os.path.join(self.data_path, "imu")))  # YENİ SATIR

        self.checkdataset()

    def processYAML(self):
        if not os.path.exists(self.yaml_file):
            raise ValueError(f"Path to Dataset not found - Incorrect YAML File Path: {self.yaml_file}")
        with open(self.yaml_file, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            if self.mode in data:
                self.data_path = data[self.mode]
                if not os.path.exists(self.data_path):
                    raise ValueError(f"Incorrect path to {self.mode} folder in YAML file: {self.data_path}")
            else:
                raise ValueError(f"YAML file does not contain path to {self.mode} folder")

    def checkdataset(self):
        for label, image, query, imu in zip(self.labels, self.images, self.queries, self.imus):
            if not image.split(".")[0] == label.split('.')[0] == query.split('.')[0] == imu.split('.')[0]:
                print(f"Warning, file not matching: {label}, {image}, {query}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.data_path, "images", self.images[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = torch.tensor(np.loadtxt(os.path.join(self.data_path, 'labels', self.labels[index])),
                              dtype=torch.float32)
        # only take the first three datapoints in the label file (id, dist, angle)
        queries = torch.tensor(np.loadtxt(os.path.join(self.data_path, 'queries', self.queries[index])),
                               dtype=torch.float32)[..., 0:3]
        imu_data = torch.tensor(np.loadtxt(os.path.join(self.data_path, 'imu', self.imus[index])),
                                dtype=torch.float32)
        # ensure 2D shape:
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)

        labels_extended = torch.zeros(queries.size(dim=0), 5, dtype=torch.float32)
        if labels.numel() > 0:
            labels_extended[labels[:, 0].long(), :] = labels[:, :]

        labels_mask = torch.full((labels_extended.size(dim=0),), fill_value=False)
        if labels.numel() > 0:
            labels_mask[labels[:, 0].long()] = True

        queries_mask = torch.full((queries.size(dim=0),), fill_value=True)

        img = torch.tensor(img).permute(2, 0, 1)  # / 255

        name = os.path.join(self.data_path, "images", self.images[index])

        sample = (img, queries, labels_extended, queries_mask, labels_mask, name, imu_data)
        return sample


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)

    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def prepare_ap_data(outputs, labels, labels_mask):
    src_boxes = outputs['pred_boxes']
    src_logits = outputs['pred_logits']
    labels = labels[..., 1:]
    preds = []
    target = []
    batch_size = src_boxes.size(0)
    for i in range(0, batch_size):
        dct = {"boxes": src_boxes[i],
               "scores": src_logits[i],
               "labels": torch.zeros_like(src_logits[i], device=src_logits.device, dtype=int)}
        preds.append(dct)

        dct2 = {"boxes": labels[i][labels_mask[i]],
                "labels": torch.zeros((labels_mask[i].sum()), device=labels_mask.device, dtype=int)}
        target.append(dct2)

    return preds, target


def computeIOU(bb_pred, bb_label):
    bb_pred = box_cxcywh_to_xyxy(bb_pred)
    bb_label = box_cxcywh_to_xyxy(bb_label)
    iou = box_iou(bb_pred, bb_label)[0]
    return torch.diag(iou)


def compute_metrics(outputs, labels, queries_mask, labels_mask, results_dict, iou_thresh=0.5, conf_thresh=0.90):
    src_logits = outputs['pred_logits'].cpu().detach()  # only get logits, that have corresponding label
    conf_mask = torch.full(src_logits.shape, fill_value=False)
    conf_mask[src_logits >= conf_thresh] = True  # mask for conf logits that are greater than thresh
    bb_filtered = outputs['pred_boxes'][conf_mask & labels_mask].cpu().detach()
    labels_filtered = labels[conf_mask & labels_mask][..., 1:]
    fp_conf = src_logits[conf_mask & ~labels_mask & queries_mask].numel()
    fn_conf = src_logits[~conf_mask & labels_mask].numel()  # fp through conf: has corresp label but is below conf level
    res = computeIOU(bb_filtered, labels_filtered)
    fp_iou = res[res < iou_thresh].numel()
    fn_iou = fp_iou
    tp = res.numel() - fp_iou

    results_dict['tp'] += tp
    results_dict['fp'] += fp_iou + fp_conf
    results_dict['fn'] += fn_iou + fn_conf
    results_dict['IoU'] += res.sum().item()
    results_dict['tp_match'] += bb_filtered.size(0)


def print_metrics(metrics):
    metrics["precision"] = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if (metrics["tp"] + metrics["fp"]) > 0 else 0
    metrics["recall"] = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0
    metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]) if (
                                                                                                                             metrics[
                                                                                                                                 "precision"] +
                                                                                                                             metrics[
                                                                                                                                 "recall"]) > 0 else 0
    metrics["mean_IoU"] = metrics["IoU"] / metrics["tp_match"] if metrics["tp_match"] > 0 else 0
    for k, v in metrics.items():
        print(f"{k}:".ljust(6), v)
    with open("results.json", "w") as f:
        json.dump(metrics, f)


def print_latency(latency):
    latency = latency["time"] / latency["count"]
    print("Latency: ", round(latency), "ms")
    print("FPS: ", round(1 / (latency / 1000), 2))


def compute_F1_over_dist(outputs, queries, labels, queries_mask, labels_mask, results_dict_distance, iou_thresh=0.5,
                         conf_thresh=0.90):
    src_logits = outputs['pred_logits'].cpu().detach()  # only get logits, that have corresponding label
    bb_preds = outputs['pred_boxes'].cpu().detach()
    for b in range(0, src_logits.size(0)):  # batch size
        for q in range(0, src_logits.size(1)):  # query index
            if queries_mask[b, q] == True:
                dist = int(queries[b, q, 0] * 1000 // 50)
                if dist not in results_dict_distance:
                    results_dict_distance[dist] = {"tp": 0, "fp": 0, "fn": 0, "IoU": 0, "tp_match": 0}

                if src_logits[b, q] >= conf_thresh:
                    if labels_mask[b, q] == True:
                        res = computeIOU(bb_preds[b, q].unsqueeze(0), labels[b, q][..., 1:].unsqueeze(0))
                        results_dict_distance[dist]["IoU"] += res
                        results_dict_distance[dist]["tp_match"] += 1
                        if res > iou_thresh:
                            results_dict_distance[dist]["tp"] += 1
                        else:
                            results_dict_distance[dist]["fp"] += 1
                            results_dict_distance[dist]["fn"] += 1
                    else:
                        results_dict_distance[dist]["fp"] += 1
                else:
                    if labels_mask[b, q] == True:
                        results_dict_distance[dist]["fn"] += 1


def save_F1_over_dist(metrics, test_dir):
    distances = sorted([k for k in metrics])
    f1_scores = []
    res = np.zeros((len(distances), 3))
    for i, k in enumerate(distances):
        p = metrics[k]["tp"] / (metrics[k]["tp"] + metrics[k]["fp"]) if metrics[k]["tp"] + metrics[k]["fp"] > 0 else 0
        r = metrics[k]["tp"] / (metrics[k]["tp"] + metrics[k]["fn"]) if metrics[k]["tp"] + metrics[k]["fn"] > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        iou = metrics[k]["IoU"] / metrics[k]["tp_match"] if metrics[k]["tp_match"] > 0 else 0
        res[i, 0] = k
        res[i, 1] = f1
        res[i, 2] = iou
        f1_scores.append(f1)
    print()
    print("F1 and IoU over distances:")
    print(res)
    np.save(os.path.join(test_dir, 'np_arr.npy'), res)


@torch.no_grad()
def test(model, data_loader, device, output_dir="test_results"):
    model.eval()
    ap_metric = MeanAveragePrecision(box_format="cxcywh", iou_type='bbox')

    metrics_dict = defaultdict(float)
    metrics_dict_distance = {}
    latency_dict = defaultdict(float)
    with tqdm(data_loader, desc=str(f"Test").ljust(8), ncols=150) as pbar:
        for model_inputs, pad_l, pad_mask_l, pad_q, pad_mask_q, name in pbar:
            start_time = time.perf_counter()
            queries = pad_q.to(device)[..., 1:]  # remove index from queries (only for debugging reasons)
            queries_mask = pad_mask_q.to(device)
            labels = pad_l.to(device)
            labels_mask = pad_mask_l.to(device)
            outputs = model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in model_inputs.items()})

            latency_dict["time"] += (time.perf_counter() - start_time) * 1000
            latency_dict["count"] += pad_l.size(0)

            preds, target = prepare_ap_data(outputs, labels, labels_mask)
            ap_metric.update(preds, target)
            compute_metrics(outputs, labels.cpu().detach(), queries_mask.cpu().detach(), labels_mask.cpu().detach(),
                            metrics_dict)
            compute_F1_over_dist(outputs, queries.cpu().detach(), labels.cpu().detach(), queries_mask.cpu().detach(),
                                 labels_mask.cpu().detach(), metrics_dict_distance)

    print("Results:")
    print_metrics(metrics_dict)
    print()
    print_latency(latency_dict)
    print()
    save_F1_over_dist(metrics_dict_distance, output_dir)


def prepare_collate_fn(input_collate_fn):
    def collate_fn(batch):
        img, queries, labels, queries_mask, labels_mask, name, imu_data = zip(*batch)
        model_inputs = input_collate_fn(img, queries, queries_mask, imu_data)
        pad_q = pad_sequence(queries, batch_first=True, padding_value=0.0)
        pad_l = pad_sequence(labels, batch_first=True, padding_value=0.0)
        pad_mask_q = pad_sequence(queries_mask, batch_first=True, padding_value=False)
        pad_mask_l = pad_sequence(labels_mask, batch_first=True, padding_value=False)
        return model_inputs, pad_l, pad_mask_l, pad_q, pad_mask_q, name

    return collate_fn


if __name__ == '__main__':
    dataset_test = BuoyDataset(yaml_file="dataset.yaml", mode='val')
    data_loader = DataLoader(dataset_test,
                             batch_size=8,
                             sampler=torch.utils.data.SequentialSampler(dataset_test),
                             drop_last=False,
                             collate_fn=prepare_collate_fn(input_collate_fn),
                             num_workers=0,
                             )
    test(model=get_model().to("cuda:0"), data_loader=data_loader, device="cuda:0")

