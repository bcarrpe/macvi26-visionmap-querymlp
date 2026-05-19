"""Custom Dataset for Buoy Data — MLP query version

Queries are extended with MLP-predicted pixel coordinates [cx, cy].
Final query shape per buoy: [id, dist_norm, bearing_norm, cx, cy] → 5 cols
After id-strip in collate: [dist_norm, bearing_norm, cx, cy] → input_dim_gt=4
"""
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import torch
import yaml
import os
import numpy as np
import cv2
import random
import warnings

from util.box_ops import box_cxcywh_to_xyxy, box_iou
from query_mlp import QueryMLP

warnings.filterwarnings("ignore", category=UserWarning)


def collate_fn(batch):
    img, queries, labels, queries_mask, labels_mask, name, imu_data = zip(*batch)
    img = torch.stack(img, dim=0)
    pad_q      = pad_sequence(queries,      batch_first=True, padding_value=0.0)
    pad_l      = pad_sequence(labels,       batch_first=True, padding_value=0.0)
    pad_mask_q = pad_sequence(queries_mask, batch_first=True, padding_value=False)
    pad_mask_l = pad_sequence(labels_mask,  batch_first=True, padding_value=False)
    imu_data   = torch.stack(imu_data, dim=0)
    return img, pad_q, pad_l, pad_mask_q, pad_mask_l, name, imu_data


class BuoyDataset(Dataset):
    def __init__(self, yaml_file, mode='train', transform=False, augment=False,
                 mlp_weights='query_mlp.pth') -> None:
        super().__init__()

        self.yaml_file = yaml_file
        if mode in ["train", "test", "val"]:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode ({mode}) for DataSet")
        self.data_path = None

        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform = None
        if transform:
            self.transform = tf
        self.augment = augment
        self.augment_thresh = 0.5
        random.seed(0)
        torch.manual_seed(0)
        self.processYAML()

        self.labels  = sorted(os.listdir(os.path.join(self.data_path, "labels")))
        self.images  = sorted(os.listdir(os.path.join(self.data_path, "images")))
        self.queries = sorted(os.listdir(os.path.join(self.data_path, "queries")))
        self.imus    = sorted(os.listdir(os.path.join(self.data_path, "imu")))

        self.checkdataset()

        # ── load frozen MLP ────────────────────────────────────────────────
        self.mlp = QueryMLP(input_dim=6, hidden_dim=128)
        self.mlp.load_state_dict(torch.load(mlp_weights, map_location='cpu'))
        self.mlp.eval()
        for p in self.mlp.parameters():
            p.requires_grad = False
        print(f"Loaded frozen QueryMLP from {mlp_weights}")

        # normalization constants (must match train_query_mlp.py)
        self.PITCH_SCALE   = 10.0
        self.ROLL_SCALE    = 10.0
        self.HEADING_SCALE = 180.0

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

    def flip_img(self, img, labels, queries):
        """Horizontal flip. Adjusts cx and bearing."""
        img = cv2.flip(img, 1)
        if labels.numel() > 0:
            labels[:, 1] = 1 - labels[:, 1]
        queries[:, -1] *= -1
        return img, labels, queries

    def random_scale_crop(self, img, labels, scale_range=(0.5, 1.0)):
        H, W = img.shape[:2]
        scale  = random.uniform(scale_range[0], scale_range[1])
        crop_w = int(W * scale)
        crop_h = int(H * scale)
        x0 = random.randint(0, W - crop_w)
        y0 = random.randint(0, H - crop_h)

        img_cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]
        img_resized = cv2.resize(img_cropped, (W, H), interpolation=cv2.INTER_LINEAR)

        if labels.numel() > 0:
            cx_abs = labels[:, 1] * W
            cy_abs = labels[:, 2] * H
            bw_abs = labels[:, 3] * W
            bh_abs = labels[:, 4] * H

            inside = (cx_abs >= x0) & (cx_abs < x0 + crop_w) & \
                     (cy_abs >= y0) & (cy_abs < y0 + crop_h)

            if inside.any():
                labels  = labels[inside]
                cx_abs  = cx_abs[inside]
                cy_abs  = cy_abs[inside]
                bw_abs  = bw_abs[inside]
                bh_abs  = bh_abs[inside]

                labels        = labels.clone()
                labels[:, 1]  = ((cx_abs - x0) / crop_w).clamp(0, 1)
                labels[:, 2]  = ((cy_abs - y0) / crop_h).clamp(0, 1)
                labels[:, 3]  = (bw_abs / crop_w).clamp(0, 1)
                labels[:, 4]  = (bh_abs / crop_h).clamp(0, 1)
            else:
                return img, labels

        return img_resized, labels

    def color_jitter(self, img, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-brightness, brightness)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-contrast, contrast)
            mean = img.mean()
            img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            img_hsv[:, :, 1] *= 1.0 + random.uniform(-saturation, saturation)
            img_hsv[:, :, 0] += random.uniform(-hue * 180, hue * 180)
            img_hsv[:, :, 0]  = img_hsv[:, :, 0] % 180
            img_hsv[:, :, 1]  = np.clip(img_hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img

    def queries_add_noise(self, queries, dist_coeff=15, bearing_coeff=30):
        noise = lambda: 2 * (torch.rand(queries.size(dim=0), dtype=torch.float32) - 0.5)
        delta_dist    = noise() * dist_coeff
        delta_bearing = torch.atan2(noise() * bearing_coeff, queries[:, 1]) / torch.pi * 180
        queries[:, 1] += delta_dist
        queries[:, 2] += delta_bearing
        return queries

    def _mlp_features(self, queries_raw, imu):
        """
        queries_raw: [N, 3] — [id, dist_m, bearing_deg] (before normalization)
        imu: 1D tensor [pitch, roll, heading, ...]
        returns: [N, 2] tensor of [cx, cy] from frozen MLP
        """
        dist_m      = queries_raw[:, 1].numpy()
        bearing_deg = queries_raw[:, 2].numpy()
        pitch_deg   = float(imu[0])
        roll_deg    = float(imu[1])
        heading_deg = float(imu[2])

        dist_norm = dist_m / 1000.0
        inv_dist  = np.clip(1.0 / np.maximum(dist_norm, 0.001), 0.0, 10.0)

        mlp_input = np.stack([
            dist_norm,
            inv_dist,
            bearing_deg / 180.0,
            np.full_like(dist_norm, pitch_deg   / self.PITCH_SCALE),
            np.full_like(dist_norm, roll_deg    / self.ROLL_SCALE),
            np.full_like(dist_norm, heading_deg / self.HEADING_SCALE),
        ], axis=1)  # [N, 6]

        with torch.no_grad():
            mlp_out = self.mlp(torch.tensor(mlp_input, dtype=torch.float32))  # [N, 2]
        return mlp_out

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.data_path, "images", self.images[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = torch.tensor(
            np.loadtxt(os.path.join(self.data_path, 'labels', self.labels[index])),
            dtype=torch.float32)

        # load raw queries [id, dist_m, bearing_deg] — keep only first 3 cols
        queries_raw = torch.tensor(
            np.loadtxt(os.path.join(self.data_path, 'queries', self.queries[index])),
            dtype=torch.float32)[..., 0:3]

        imu_data = torch.tensor(
            np.loadtxt(os.path.join(self.data_path, 'imu', self.imus[index])),
            dtype=torch.float32)

        if queries_raw.ndim == 1:
            queries_raw = queries_raw.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)

        if self.augment:
            if random.random() > self.augment_thresh:
                img, labels, queries_raw = self.flip_img(img, labels, queries_raw)
                queries_raw = self.queries_add_noise(queries_raw)
            if random.random() > self.augment_thresh:
                img, labels = self.random_scale_crop(img, labels, scale_range=(0.5, 1.0))
            if random.random() > self.augment_thresh:
                img = self.color_jitter(img)

        # ── get MLP pixel predictions (uses raw dist_m, bearing_deg) ──────
        mlp_out = self._mlp_features(queries_raw, imu_data.flatten())  # [N, 2]

        # ── normalize dist and bearing ─────────────────────────────────────
        queries_raw = queries_raw.clone()
        queries_raw[..., 1] = queries_raw[..., 1] / 1000
        queries_raw[..., 2] = queries_raw[..., 2] / 180

        # ── build final query: [id, dist_norm, bearing_norm, cx, cy] ──────
        queries = torch.cat([queries_raw, mlp_out], dim=1)  # [N, 5]

        # ── align labels to query order ────────────────────────────────────
        labels_extended = torch.zeros(queries.size(0), 5, dtype=torch.float32)
        if labels.numel() > 0:
            labels_extended[labels[:, 0].long(), :] = labels[:, :]

        labels_mask = torch.full((labels_extended.size(0),), fill_value=False)
        if labels.numel() > 0:
            labels_mask[labels[:, 0].long()] = True

        queries_mask = torch.full((queries.size(0),), fill_value=True)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2, 0, 1) / 255

        name = os.path.join(self.data_path, "images", self.images[index])
        return (img, queries, labels_extended, queries_mask, labels_mask, name, imu_data)