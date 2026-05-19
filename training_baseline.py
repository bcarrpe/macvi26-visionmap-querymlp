import os
import time
import math
import sys
import random

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.buoy_dataset_baseline import BuoyDataset, collate_fn
from models.detr import DETR, SetCriterion
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from util.misc import save_on_master, BasicLogger

# ── reproducibility ────────────────────────────────────────────────────────────
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ── helpers ────────────────────────────────────────────────────────────────────

def init_position_encoding(hidden_dim):
    return PositionEmbeddingSine(hidden_dim // 2, normalize=True)


def init_backbone(lr_backbone, hidden_dim, backbone_name='resnet50', dilation=False):
    position_embedding = init_position_encoding(hidden_dim)
    backbone = Backbone(backbone_name, lr_backbone > 0, False, dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def init_transformer(hidden_dim, dropout, nheads, dim_feedforward,
                     enc_layers, dec_layers, pre_norm):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )


# ── training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, data_loader, optimizer, device,
                    epoch, max_norm=0.1, logger=None):
    model.train()
    criterion.train()

    loss_total, loss_obj, loss_boxL1, loss_giou = [], [], [], []

    with tqdm(data_loader, desc=f"Train - Epoch {epoch}".ljust(16), ncols=150) as pbar:
        for images, queries, labels, queries_mask, labels_mask, name, imu in pbar:
            images       = images.to(device)
            queries      = queries.to(device)[..., 1:]   # strip id column
            labels       = labels.to(device)
            queries_mask = queries_mask.to(device)
            labels_mask  = labels_mask.to(device)

            outputs   = model(images, queries, queries_mask)
            loss_dict = criterion(outputs, labels, queries_mask, labels_mask)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict)
            loss_total.append(losses.item())
            loss_obj.append(sum(loss_dict[k] * weight_dict[k]
                               for k in loss_dict if 'loss_bce'  in k).item())
            loss_boxL1.append(sum(loss_dict[k] * weight_dict[k]
                                  for k in loss_dict if 'loss_bbox' in k).item())
            loss_giou.append(sum(loss_dict[k] * weight_dict[k]
                                 for k in loss_dict if 'loss_giou' in k).item())

            pbar.set_postfix({
                "Loss":  f"{sum(loss_total)/len(loss_total):.3f}",
                "Obj":   f"{sum(loss_obj)/len(loss_obj):.3f}",
                "BoxL1": f"{sum(loss_boxL1)/len(loss_boxL1):.3f}",
                "GIoU":  f"{sum(loss_giou)/len(loss_giou):.3f}",
            })

            if not math.isfinite(losses.item()):
                print(f"Loss is {losses.item()}, stopping training")
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm,
                                               error_if_nonfinite=True)
            optimizer.step()

    if logger is not None:
        result = {
            "loss_total": sum(loss_total) / len(loss_total),
            "loss_obj":   sum(loss_obj)   / len(loss_obj),
            "loss_boxL1": sum(loss_boxL1) / len(loss_boxL1),
            "loss_giou":  sum(loss_giou)  / len(loss_giou),
        }
        logger.updateLosses(result, epoch, 'train')
        return result
    return None


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch, logger=None):
    model.eval()
    criterion.eval()

    loss_total, loss_obj, loss_boxL1, loss_giou = [], [], [], []

    with tqdm(data_loader, desc=f"Val   - Epoch {epoch}".ljust(16), ncols=150) as pbar:
        for images, queries, labels, queries_mask, labels_mask, name, imu in pbar:
            images       = images.to(device)
            queries      = queries.to(device)[..., 1:]   # strip id column
            labels       = labels.to(device)
            queries_mask = queries_mask.to(device)
            labels_mask  = labels_mask.to(device)

            outputs   = model(images, queries, queries_mask)
            loss_dict = criterion(outputs, labels, queries_mask, labels_mask)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict)
            loss_total.append(losses.item())
            loss_obj.append(sum(loss_dict[k] * weight_dict[k]
                               for k in loss_dict if 'loss_bce'  in k).item())
            loss_boxL1.append(sum(loss_dict[k] * weight_dict[k]
                                  for k in loss_dict if 'loss_bbox' in k).item())
            loss_giou.append(sum(loss_dict[k] * weight_dict[k]
                                 for k in loss_dict if 'loss_giou' in k).item())

            pbar.set_postfix({
                "Loss":  f"{sum(loss_total)/len(loss_total):.3f}",
                "Obj":   f"{sum(loss_obj)/len(loss_obj):.3f}",
                "BoxL1": f"{sum(loss_boxL1)/len(loss_boxL1):.3f}",
                "GIoU":  f"{sum(loss_giou)/len(loss_giou):.3f}",
            })

            if logger is not None:
                logger.computeStats(outputs,
                                    labels.cpu().detach(),
                                    queries_mask.cpu().detach(),
                                    labels_mask.cpu().detach(),
                                    mode='val')

    if logger is not None:
        result = {
            "loss_total": sum(loss_total) / len(loss_total),
            "loss_obj":   sum(loss_obj)   / len(loss_obj),
            "loss_boxL1": sum(loss_boxL1) / len(loss_boxL1),
            "loss_giou":  sum(loss_giou)  / len(loss_giou),
        }
        logger.updateLosses(result, epoch, 'val')
        logger.printCF(thresh=0.5, mode='val')
        ap50 = logger.print_mAP50(mode='val')
        logger.print_mAP50_95(mode='val')
        result['AP50'] = ap50
        return result
    return None


# ── settings ───────────────────────────────────────────────────────────────────

# General
transfer_learning = True
path_to_weights   = "detr-r50-e632da11.pth"
output_dir        = "checkpoints_baseline"
start_epoch       = 0

# Backbone
lr_backbone = 1e-5

# Transformer
hidden_dim      = 256
enc_layers      = 6
dec_layers      = 6
dim_feedforward = 2048
dropout         = 0.1
nheads          = 8
pre_norm        = True
input_dim_gt    = 2    # [dist_norm, bearing_norm] — baseline, no QueryMLP
use_embeddings  = False

# Loss
aux_loss       = True
bce_loss_coef  = 1
bbox_loss_coef = 3
giou_loss_coef = 7

# Optimizer / scheduler
lr             = 1e-4
weight_decay   = 1e-3
epochs         = 183
lr_drop        = 135
clip_max_norm  = 0.1

# DataLoader
batch_size  = 16
num_workers = 0

# Device
device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_to_dataset = "dataset.yaml"

# ── build model ────────────────────────────────────────────────────────────────

backbone    = init_backbone(lr_backbone, hidden_dim)
transformer = init_transformer(hidden_dim, dropout, nheads, dim_feedforward,
                               enc_layers, dec_layers, pre_norm)
model = DETR(
    backbone,
    transformer,
    input_dim_gt=input_dim_gt,
    aux_loss=aux_loss,
    use_embeddings=use_embeddings,
)
model.to(device)

model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {n_parameters:,}")

# ── build loss ─────────────────────────────────────────────────────────────────

weight_dict = {'loss_bce': bce_loss_coef, 'loss_bbox': bbox_loss_coef,
               'loss_giou': giou_loss_coef}
if aux_loss:
    aux_weight_dict = {}
    for i in range(dec_layers - 1):
        aux_weight_dict.update({f"{k}_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
criterion = SetCriterion(weight_dict, losses=['labels', 'boxes'])

# ── build optimizer ────────────────────────────────────────────────────────────

param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad]},
    {"params": [p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad],
     "lr": lr_backbone},
]
optimizer    = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

# ── build datasets ─────────────────────────────────────────────────────────────

dataset_train = BuoyDataset(yaml_file=path_to_dataset, mode='train', augment=True)
dataset_val   = BuoyDataset(yaml_file=path_to_dataset, mode='val')

data_loader_train = DataLoader(dataset_train, batch_size,
                               sampler=torch.utils.data.RandomSampler(dataset_train),
                               collate_fn=collate_fn, num_workers=num_workers)
data_loader_val   = DataLoader(dataset_val, batch_size,
                               sampler=torch.utils.data.SequentialSampler(dataset_val),
                               drop_last=False, collate_fn=collate_fn,
                               num_workers=num_workers)

# ── load COCO-pretrained weights ───────────────────────────────────────────────

if transfer_learning:
    print(f"Loading pretrained weights from {path_to_weights} ...")
    checkpoint = torch.load(path_to_weights, map_location='cpu')
    del checkpoint['model']['class_embed.weight']
    del checkpoint['model']['class_embed.bias']
    missing, unexpected = model_without_ddp.load_state_dict(
        checkpoint['model'], strict=False)
    print(f"  Missing keys   : {missing}")
    print(f"  Unexpected keys: {unexpected}")

# ── training loop ──────────────────────────────────────────────────────────────

logger     = BasicLogger()
best_ap    = -1
best_epoch = -1
print("Start training")
start_time = time.time()

for epoch in range(start_epoch, epochs):
    logger.resetStats()

    train_results = train_one_epoch(model, criterion, data_loader_train,
                                    optimizer, device, epoch, clip_max_norm, logger)
    val_results   = evaluate(model, criterion, data_loader_val,
                             device, epoch, logger)
    lr_scheduler.step()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.saveLossLogs(output_dir)
        logger.saveStatsLogs(output_dir, epoch)
        logger.plotLoss(output_dir)

        if val_results["AP50"] > best_ap:
            print(f"  ✓ New best at epoch {epoch}  (AP50={val_results['AP50']:.4f})")
            logger.plotPRCurve(path=output_dir, mode='val')
            logger.plotConfusionMat(path=output_dir, thresh=0.5, mode='val')
            logger.plotPRCurveDet(path=output_dir, mode='val')
            best_ap    = val_results["AP50"]
            best_epoch = epoch
            save_on_master({
                'model':        model_without_ddp.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch':        epoch,
            }, os.path.join(output_dir, "best.pth"))

total_time = time.time() - start_time
h = int(total_time // 3600)
m = int((total_time % 3600) // 60)
s = int(total_time % 60)
print(f"Training time: {h:02d}:{m:02d}:{s:02d}")
logger.writeEpochStatsLog(path=output_dir, best_epoch=best_epoch)
print(f"Best validation result at epoch {best_epoch}")
