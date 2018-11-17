import csv
import numpy as np
import os

# ========== CONSTANTS ========== START
DATA_DIR = 'data'
IMAGE_DIM = 1024
CSV_FILE = 'stage_1_train_labels.csv'

# Dict of patientId : locations
#         <patiend_Id> : [[x1, y1, w1, h1], ... , [xn, yn, wn, hn]]
def create_true_box_dict():
    pneumonia_dict = {}
    # load table
    with open(os.path.join(DATA_DIR, CSV_FILE), mode='r') as infile:
        
        #read in the csv as a dictionary
        dict_reader = csv.DictReader(infile)

        # loop through rows
        for row in dict_reader:
            p_id   = row["patientId"]
            # if row contains pneumonia add label to dictionary
            if row["Target"] == '1':
                x      = int(float(row["x"])) 
                y      = int(float(row["y"])) 
                width  = int(float(row["width"])) 
                height = int(float(row["height"])) 
                b_box = [x, y, width, height]
                
                if p_id in pneumonia_dict: 
                    pneumonia_dict[p_id].append(b_box)
                else:
                    pneumonia_dict[p_id] = [b_box]
            else:
                pneumonia_dict[p_id] = [[0,0,0,0]]
    return pneumonia_dict

def create_bbox_mask(box):
    mask = np.zeros((IMAGE_DIM, IMAGE_DIM))
    for i in range (len(box)):
        x, y, w, h = box[i]
        mask[y:y+h, x:x+w] = 1
    return mask

def calculate_iou(box_true, box_pred):
    true_mask = create_bbox_mask(box_true)
    pred_mask = create_bbox_mask(box_pred)
    intersection_mask = 1*np.logical_and(true_mask, pred_mask)
    union_mask = 1*np.logical_or(true_mask, pred_mask)
    intersection = np.sum(intersection_mask)
    union = np.sum(union_mask)
    if intersection == 0 :
        return 0
    return float(intersection) / float(union)

def avg_iou(true_dict, pred_dict):
    cnt = 0
    f1 = 0
    for pId, box_pred in pred_dict.items():
        box_true = true_dict[pId]
        cnt += 1
        f1 = calculate_iou(box_true, box_pred)
    return (f1/cnt)




