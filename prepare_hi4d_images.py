import sys
import os
import argparse
import pickle as pkl
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device) 
    return image.permute(2, 0, 1).contiguous()

parser = argparse.ArgumentParser()
parser.add_argument("--images-subdir", default="images")
parser.add_argument("--hi4d-dir", default="datasets/original/Hi4D/")
parser.add_argument("--output-dir", default="datasets/processed/Hi4D/cropped_images")
args = parser.parse_args()

for split in ['val', 'test']:
    if not os.path.exists(args.output_dir):
        os.system('mkdir \"'+args.output_dir+'\"')

    with open(os.path.join(args.hi4d_dir.replace('original', 'processed'), 'processed_single_camera.pkl'), 'rb') as f:
        data = pkl.load(f)

    split_info = np.load(os.path.join(args.hi4d_dir.replace('original', 'processed'), 'train_val_test_split.npz'))

    os.system('mkdir \"'+args.output_dir+'\"')
    camera = '4'
    # fnames = list(os.listdir(os.path.join(args.hi4d_dir, args.images_subdir)))
    # start = 0
    # end = len(fnames)
    # fnames = fnames[start:end]
    # for fname in tqdm(fnames):
    for pair in split_info[split]:
        # if 'pair19_dance19_cam4_000061' not in fname:
        #     continue
        for action in data['pair'+pair]:
            img_dir = os.path.join(args.hi4d_dir, 'pair'+pair, action, args.images_subdir, camera)
            for fname in os.listdir(img_dir):
                path = os.path.join(img_dir, fname)
                img_np = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                boxes = []
                datum = None
                # print(type(data[fname.split('_')[0]]))
                # print(type(data[fname.split('_')[0]][fname.split('_')[1]]))
                for d in data['pair'+pair][action]['image_data']:
                    if fname.split('.')[0] in d[camera]:
                        datum = d[camera][fname.split('.')[0]]
                        break
                keypoints_key = 'vitpose'
                for i in range(2):
                    min_x = None
                    max_x = None
                    min_y = None
                    max_y = None
                    # print(i, datum[i][keypoints_key][:25])
                    for j in range(25):
                        if datum[i][keypoints_key][j,-1] < 1e-8:
                            continue
                        if min_x is None or min_x > datum[i][keypoints_key][j,0]:
                            min_x = datum[i][keypoints_key][j,0]
                        if min_y is None or min_y > datum[i][keypoints_key][j,1]:
                            min_y = datum[i][keypoints_key][j,1]
                        if max_x is None or max_x < datum[i][keypoints_key][j,0]:
                            max_x = datum[i][keypoints_key][j,0]
                        if max_y is None or max_y < datum[i][keypoints_key][j,1]:
                            max_y = datum[i][keypoints_key][j,1]
                    if min_x is None or max_x is None or min_y is None or max_y is None:
                        continue
                    boxes.append([max(0, min_x-70), max(0, min_y-70), min(img_np.shape[1], max_x+70), min(img_np.shape[0], max_y+70)])
                out_fname = pair+'_'+action+'_'+camera+'_'+fname.split('.')[0]+'.jpg'
                if len(boxes) < 2:
                    if len(boxes) < 1:
                        print('Less than 1 valid box', out_fname)
                        os.system('cp \"'+path+'\" \"'+args.output_dir+'\"')
                        continue
                    print('Only one box', out_fname)
                    boxes = [[boxes[0][i] - 70 for i in range(2)]+[boxes[0][i] + 70 for i in range(2, 4)]]
                    # continue
                print('Two boxes')
                annotated_img = cv2.imread(path)
                covering_box = [min([box[0] for box in boxes]), min([box[1] for box in boxes]), max([box[2] for box in boxes]), max([box[3] for box in boxes])]
                annotated_img = annotated_img[int(covering_box[1]):int(covering_box[3]), int(covering_box[0]):int(covering_box[2])]
                try:
                    cv2.imwrite(os.path.join(args.output_dir, out_fname), annotated_img)
                except Exception as e:
                    print(e)
                    annotated_img = cv2.imread(path)
                    covering_box = [min([box[0] for box in boxes]), min([box[1] for box in boxes]), max([box[2] for box in boxes]), max([box[3] for box in boxes])]
                    print(annotated_img.size, covering_box)
                    annotated_img = annotated_img[int(covering_box[1]):int(covering_box[3]), int(covering_box[0]):int(covering_box[2])]
                    cv2.imwrite(os.path.join(args.output_dir, out_fname), annotated_img)
