"""Script to create Dataset in Yolo Format for Transformer (including images, buoyGT data, BB Labels, and raw IMU data)
Modified to automatically split data into train/val/test sets from all available data"""

import os
import json
from os.path import isdir
import numpy as np
import shutil
import cv2
from util.association_utility import GetGeoData, getIMUData, filterBuoys, createQueryData, haversineDist

def labelsJSON2Yolo(labels, queries, ship_pose):
    # function converts json data to yolo format with corresponding query ID
    result = []
    for BB in labels[1]["objects"]:
        if "distanceGT" in BB["attributes"]:
            # get query ID
            lat_BB = BB["attributes"]["buoyLat"]
            lng_BB = BB["attributes"]["buoyLng"]
            distances = list(map(lambda x: haversineDist(lat_BB, lng_BB, x[2], x[3]), queries))
            queryID = np.argmin(distances)
            if distances[queryID] > 30:
                if verbose:
                    print("\t \t Skipping: Distance between query Buoy and Label buoy exceeds thresh")
                    print("\t \t Distances to queries: ", [round(x) for x in distances])
                    print("\t \t Distance to camera: ", round(haversineDist(lat_BB, lng_BB, *ship_pose[:2])))
                return None
           
            # get BB info in yolo format
            x1 = BB["x1"]
            y1 = BB["y1"]
            x2 = BB["x2"]
            y2 = BB["y2"]
            bbCenterX = ((x1 + x2) / 2) / 1920 
            bbCenterY = ((y1 + y2) / 2) / 1080 
            bbWidth = (x2-x1) / 1920 
            bbHeight = (y2-y1) / 1080 
            bbInfo = str(queryID) + " " + str(bbCenterX) + " " + str(bbCenterY) + " " + str(bbWidth) + " " + str(bbHeight) + "\n"

            result.append(bbInfo)

    return result 

def process_sample(src_path_img, labels_path, imu_data, frame_id, target_dir, sample_name, buoyGTData):
    """Process a single sample and save to target directory (including raw IMU data)"""
    # create query file
    imu_curr = imu_data[frame_id] 
    ship_pose = [imu_curr[3],imu_curr[4],imu_curr[2]]
    buoys_on_tile = buoyGTData.getBuoyLocations(ship_pose[0], ship_pose[1]) 
    filteredGT = filterBuoys(ship_pose, buoys_on_tile)
    queries = createQueryData(ship_pose, filteredGT)
    if len(queries) == 0:
        if verbose:
            print(f"\t \t Skipping: No nearby buoys found for file {src_path_img}")
        return False
    queryFile = os.path.join(target_dir, "queries", sample_name + '.txt')

    # create labels file
    label_data = json.load(open(labels_path, 'r'))
    txtlabels = labelsJSON2Yolo(label_data, queries, ship_pose)
    if txtlabels is None: # if dist between label buoy and query buoy too large -> skip 
        return False
    if len(txtlabels) == 0: # if labels file empty
        if verbose:
            print(f"\t \t Warning: Empty labels file: {labels_path}")
    labelfile = os.path.join(target_dir, "labels", sample_name + '.txt')

    # save image
    dest_path = os.path.join(target_dir, "images", sample_name + '.png')
    if resize_imgs == False:
        shutil.copy(src_path_img, dest_path)
    else:
        img = cv2.imread(src_path_img)
        img_resized = cv2.resize(img, (0,0), fx=resize_coeffs[0], fy=resize_coeffs[1])
        cv2.imwrite(dest_path, img_resized)
    
    # save query file
    with open(queryFile, 'w') as f:
        data = [str(i) + " " + str(data[0]) + " " + str(data[1]) + " " + str(data[2]) + " " + str(data[3]) + "\n" for i,data in enumerate(queries)]
        f.writelines(data)

    # save label file
    with open(labelfile, 'w') as f:
        f.writelines(txtlabels)
    
    # save raw IMU data for this frame
    imuFile = os.path.join(target_dir, "imu", sample_name + '.txt')
    with open(imuFile, 'w') as f:
        # Write IMU data as space-separated values
        # Format: timestamp latitude longitude heading ...other IMU fields
        imu_line = " ".join([str(val) for val in imu_curr]) + "\n"
        f.write(imu_line)
    
    return True

# Settings:
verbose = True
resize_imgs = True
resize_coeffs = [0.5, 0.5]

# Split ratios (train, val, test) - must sum to 1.0
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Paths
target_dir = "D:/1. My folder/LESSON/LOOKOUT/aton-dataset"
data_path = "D:/1. My folder/LESSON/LOOKOUT/labeled/"

# Check if target directory exists
if os.path.exists(target_dir):
    raise ValueError("Aborting... Specified target dir already exists:", target_dir)

# Create directory structure for train/val/test
os.makedirs(target_dir, exist_ok=True)
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(target_dir, split)
    os.makedirs(split_dir, exist_ok=False)
    os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "queries"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "imu"), exist_ok=True)  # Add IMU directory

buoyGTData = GetGeoData()

# Collect all valid datafolders (excluding test folders and specific folders)
datafolders = []
for folder in os.listdir(data_path):
    if os.path.isdir(os.path.join(data_path, folder)):
        if folder != 'Testdata' and folder != 'True_Negatives' and folder != "Boston_oak":
            datafolders.append(folder)

print("Folders Found: ", sorted(datafolders))

# Collect all samples from all folders
all_samples = []
for folder in datafolders:
    parent_folder = os.path.join(data_path, folder)
    for subfolder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, subfolder)
        print("Collecting from: ", folder_path)
        images = os.path.join(folder_path, "images")
        imu = os.listdir(os.path.join(folder_path, "imu"))[0]
        labels = os.path.join(folder_path, "labels")
        imu_data = getIMUData(os.path.join(folder_path, "imu", imu))
        
        for sample in os.listdir(images):
            src_path_img = os.path.join(images, sample)
            frame_id = int(sample.split(".")[0]) - 1
            src_path_label = os.path.join(labels, sample + ".json")
            
            all_samples.append({
                'img_path': src_path_img,
                'label_path': src_path_label,
                'imu_data': imu_data,
                'frame_id': frame_id
            })

print(f"Total samples collected: {len(all_samples)}")

# Shuffle samples for random split
np.random.seed(42)  # for reproducibility
indices = np.random.permutation(len(all_samples))

# Calculate split indices
train_end = int(len(all_samples) * train_ratio)
val_end = train_end + int(len(all_samples) * val_ratio)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

# Process samples for each split
sample_counters = {'train': 0, 'val': 0, 'test': 0}

for split_name, split_indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
    print(f"\nProcessing {split_name} split...")
    split_dir = os.path.join(target_dir, split_name)
    
    for idx in split_indices:
        sample = all_samples[idx]
        sample_name = "0" * (5 - len(str(sample_counters[split_name]))) + str(sample_counters[split_name])
        
        success = process_sample(
            sample['img_path'],
            sample['label_path'],
            sample['imu_data'],
            sample['frame_id'],
            split_dir,
            sample_name,
            buoyGTData
        )
        
        if success:
            sample_counters[split_name] += 1

print("\nDONE!")
print(f"Final counts - Train: {sample_counters['train']}, Val: {sample_counters['val']}, Test: {sample_counters['test']}")
print("Total Processed Folders: ", datafolders)