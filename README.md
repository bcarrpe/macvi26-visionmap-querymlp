V2W-Transformer
========
This repo contains the modified architecture of DETR based Object Detection Models. It Replaces the fixed amount of object queries with a varying amount of sampled nearby chart markers. The spatial position of the chart markers w.r.t. the camera is encoded with an MLP and passed as an embedding to the decoder. After the final decoder layer, each chart marker embedding predicts its visibility for the given frame as well as bounding box coordinates.

![transformer_architecture](https://github.com/user-attachments/assets/8ffa77b7-5a87-48d3-849e-db56a00b888e)

## Challenges participation
Be sure to read the challenges details at https://macvi.org/workshop/cvpr/challenges/vision_map

To get started
1. Clone this repository
   ```bash
   git clone https://github.com/mkaraaslan-dev/CVPR2026-Transformer
   cd CVPR2026-Transformer
   ```
2. Configure a working conda enviroment
   ```bash
   conda create -y -n cvpr2026_macvi_visionmap python=3.11
   conda activate cvpr2026_macvi_visionmap
   pip install -r requirements.txt
   ```
3. Download the dataset from https://drive.google.com/drive/folders/1OXSok1Aux0rfygNQHFIe8goB695DeN3f
4. Change the paths in `dataset.yaml` to point to the downloaded dataset
5. Either download example weights (`best.pth`) from https://drive.google.com/drive/folders/1QSybp1gAVP2HNXc9ye-5T_X-7j1huSZP or train a model yourself (see Training)
6. Change `path_to_weights` in `evaluate_example.py` to your model weights. 
7. Either create `test_results/` directory or change the path in the script.
8. Run evaluation:
   ```bash
   python evaluate_example.py
   ```
9. Upload `result.json` to the leaderboard at https://macvi.org/leaderboard/surface/vision-to-chart/vision-to-chart
   

## Dataset
The dataset has to be provided in the following format:
```
|
├── train
│   ├── images
│   |   ├── 00001.png
│   |   └── 00002.png
│   ├── labels
│   |   ├── 00001.txt
│   |   └── 00002.txt
│   └── queries
│   |   ├── 00001.txt
│   |   └── 00002.txt
│   └── imu
│       ├── 00001.txt
│       └── 00002.txt
├── test
├── val
└── dataset.yaml
```
The dataset.yaml contains the paths to the train test and val folders. Each folder consists of an image, label and query subfolder, where the ids of the individual samples can be used as cross reference to the other folders. 

A labels.txt file contains the bounding box annotations in YOLO format alongside the corresponding query id (first position). A query.txt file contains all sampled chart markers (queries) for the given frame, containing dist, bearing, lat, lng as well as its id.


## Training
To train the model, run:
```bash
python training.py
```
Hyperparameters as well as paths to weights and Datasets can be set under the SETTINGS section in the script.
The script also support multi GPU training (however not distributed on different compute nodes). This can be enabled by setting distributed to True.


## Testing
To test the model on labeled data run:
```bash
python test.py
```

## Inference on Video
To run inference on a video to compute the associations between chart markers and detected objects, run:
```bash
python buoyAssociation.py
```
Specify the Path to the Video and IMU File in the Function call (at the bottom of the script)
