from som_app.app import SAM
import sys
import os
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm
import supervision as sv
import argparse

from segment_anything.utils.transforms import ResizeLongestSide

resize_transform = ResizeLongestSide(SAM.image_encoder.img_size)

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device) 
    return image.permute(2, 0, 1).contiguous()

parser = argparse.ArgumentParser()
parser.add_argument("--images-subdir", default="images")
parser.add_argument("--flickrci3d-dir", default="datasets/original/FlickrCI3D_Signatures/")
parser.add_argument("--output-subdir", default="cropped_segmented_images")
args = parser.parse_args()

for split in ['val', 'test']:
    test = (split == "test")
    with open(os.path.join(args.flickrci3d_dir, 'data.json')) as f:
        ci3d = json.load(f)
    ci3d_train_dict = {datum['image_url'].split('/')[-1]: datum for datum in ci3d['test' if test else 'train']}
    with open(args.flickrci3d_dir+'/'+('test' if test else 'train')+'/interaction_contact_signature.json') as f:
        contacts = json.load(f)
    if test:
        keys_per_split = {'test': [key+'_'+str(i) for key in contacts for i in range(len(contacts[key]['ci_sign']))]}
    else:
        keys_per_split = np.load(os.path.join(args.flickrci3d_dir.replace('original', 'processed'), 'train', 'train_val_split.npz'))
        keys_per_split = {spl: keys_per_split[spl].tolist() for spl in keys_per_split.files}
        keys_per_split = {spl: [key+'_'+str(i) for key in keys_per_split[spl] for i in range(len(contacts[key]['ci_sign']))] for spl in keys_per_split}

    os.system('mkdir \"'+os.path.join(args.flickrci3d_dir, 'test' if test else 'train', args.output_subdir)+'\"')
    for spl in keys_per_split:
        if (not test) and spl != 'val':
            continue
        if len(keys_per_split[spl]) == 0:
            continue
        start = 0
        end = len(keys_per_split[spl])
        if len(sys.argv) > 1:
            start = int(sys.argv[1])
            end = int(sys.argv[2])
        for key in tqdm(keys_per_split[spl][start:end]):
            # if 'Rugby_7239_0' != key:
            #     continue
            print(key)
            img_key = '_'.join(key.split('_')[:2])
            path = args.flickrci3d_dir+f'/{"test" if test else "train"}/{args.images_subdir}/{img_key}.png'
            img_np = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            contacts[img_key]['bbxes'] = [
                [max(0, num) for num in box]
                for box in contacts[img_key]['bbxes']
            ]
            all_boxes = torch.tensor(contacts[img_key]['bbxes']).to(SAM.device)
            print(all_boxes)
            batched_input = [
                {
                    'image': prepare_image(img_np, resize_transform, SAM.device),
                    'boxes': resize_transform.apply_boxes_torch(all_boxes, img_np.shape[:2]),
                    'original_size': img_np.shape[:2]
                }
            ]
            batched_output = SAM(batched_input, multimask_output=False)
            pair_index = int(key.split('_')[-1])
            annotated_img = cv2.imread(path)
            print('num masks', len(batched_output[0]['masks']))
            for j in range(len(contacts[img_key]['bbxes'])):
                if j not in contacts[img_key]['ci_sign'][pair_index]['person_ids']:
                    annotated_img = np.where(
                        np.repeat(batched_output[0]['masks'].cpu().numpy()[j].reshape(*annotated_img.shape[:2], 1), axis=2, repeats=3) > 0,
                        255*np.ones_like(annotated_img),
                        annotated_img
                    )
            boxes = [contacts[img_key]['bbxes'][j] for j in contacts[img_key]['ci_sign'][pair_index]['person_ids']]
            covering_box = [min([box[0] for box in boxes]), min([box[1] for box in boxes]), max([box[2] for box in boxes]), max([box[3] for box in boxes])]
            cropped_boxes = [[box[t] - covering_box[t % 2] for t in range(4)] for box in boxes]
            cropped_masks = [batched_output[0]['masks'].cpu()[j].view(*annotated_img.shape[:2])[int(covering_box[1]):int(covering_box[3]),int(covering_box[0]):int(covering_box[2])] for j in contacts[img_key]['ci_sign'][pair_index]['person_ids']]
            annotated_img = annotated_img[int(covering_box[1]):int(covering_box[3]), int(covering_box[0]):int(covering_box[2])]
            detections = sv.Detections(
                xyxy=np.array(cropped_boxes),
                mask=torch.stack(cropped_masks).numpy(),
            )
            try:
                cv2.imwrite(os.path.join(args.flickrci3d_dir, "test" if test else "train", args.output_subdir, key+'.png'), annotated_img)
            except Exception as e:
                print(e)
                annotated_img = cv2.imread(path)
                covering_box = [min([box[0] for box in boxes]), min([box[1] for box in boxes]), max([box[2] for box in boxes]), max([box[3] for box in boxes])]
                print(annotated_img.size, covering_box)
                annotated_img = annotated_img[int(covering_box[1]):int(covering_box[3]), int(covering_box[0]):int(covering_box[2])]
                cv2.imwrite(os.path.join(args.flickrci3d_dir, "test" if test else "train", args.output_subdir, key+'.png'), annotated_img)
