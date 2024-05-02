import sys
import os
import pickle as pkl
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm
import argparse

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device) 
    return image.permute(2, 0, 1).contiguous()

parser = argparse.ArgumentParser()
parser.add_argument("--images-subdir", default="train/s02/images") # change s02 to s03 for "val" split
parser.add_argument("--chi3d-dir", default="datasets/original/CHI3D/")
parser.add_argument("--output-dir", default="datasets/processed/CHI3D/cropped_images") # change s02 to s03 for "val" split
args = parser.parse_args()

for split in ['trainval', 'val']:
    processed_path = os.path.join(args.chi3d_dir.replace('original', 'processed'), split+'_optimization.pkl')
    sections_dict = np.load(os.path.join(args.chi3d_dir.replace('original', 'processed'), 'train', f'train_{split}_split.npz'))
    sections = sections_dict[split]
    assert len(sections) == 1
    section = sections[0]

    if not os.path.exists(args.output_dir):
        os.system("mkdir \""+args.output_dir)+"\"")

    with open(processed_path, 'rb') as f:
        data = pkl.load(f)


    for datum in tqdm(data):
        if not datum['is_contact_frame']:
            continue
        path = os.path.join(args.chi3d_dir, args.images_subdir, datum['imgname'].replace('.png', '.jpg'))
        if not os.path.exists(path):
            print(path)
            continue
        if os.path.exists(os.path.join(args.output_dir, section+'_'+datum['imgname'].replace('.png', '.jpg'))):
            continue
        print(datum['imgname'])
        img_np = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        boxes = []
        keypoints_key = 'vitpose'
        for i in range(datum[keypoints_key].shape[0]):
            min_x = None
            max_x = None
            min_y = None
            max_y = None
            for j in range(25):
                if datum[keypoints_key][i,j,-1] < 1e-8:
                    continue
                if min_x is None or min_x > datum[keypoints_key][i,j,0]:
                    min_x = datum[keypoints_key][i,j,0]
                if min_y is None or min_y > datum[keypoints_key][i,j,1]:
                    min_y = datum[keypoints_key][i,j,1]
                if max_x is None or max_x < datum[keypoints_key][i,j,0]:
                    max_x = datum[keypoints_key][i,j,0]
                if max_y is None or max_y < datum[keypoints_key][i,j,1]:
                    max_y = datum[keypoints_key][i,j,1]
            if min_x is None or max_x is None or min_y is None or max_y is None:
                break
            boxes.append([max(0, min_x-70), max(0, min_y-70), min(img_np.shape[1], max_x+70), min(img_np.shape[0], max_y+70)])
        if len(boxes) < 2:
            print('Less than 2 valid boxes', datum['imgname'])
            os.system('cp \"'+path+'\" \"'+os.path.join(output_dir, section+'_'+datum['imgname'].replace('.png', '.jpg'))+'\"')
            continue

        annotated_img = cv2.imread(path)
        covering_box = [min([box[0] for box in boxes]), min([box[1] for box in boxes]), max([box[2] for box in boxes]), max([box[3] for box in boxes])]
        cropped_boxes = [[box[t] - covering_box[t % 2] for t in range(4)] for box in boxes]
        annotated_img = annotated_img[int(covering_box[1]):int(covering_box[3]), int(covering_box[0]):int(covering_box[2])]
        try:
            cv2.imwrite(os.path.join(args.output_dir, section+'_'+datum['imgname'].replace('.png', '.jpg')), annotated_img)
        except Exception as e:
            print(e)
            annotated_img = cv2.imread(path)
            covering_box = [min([box[0] for box in boxes]), min([box[1] for box in boxes]), max([box[2] for box in boxes]), max([box[3] for box in boxes])]
            print(annotated_img.size, covering_box)
            annotated_img = annotated_img[int(covering_box[1]):int(covering_box[3]), int(covering_box[0]):int(covering_box[2])]
            cv2.imwrite(os.path.join(args.output_dir, section+'_'+datum['imgname'].replace('.png', '.jpg')), annotated_img)
