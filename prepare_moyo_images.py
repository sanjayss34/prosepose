import argparse
import json
import os
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moyo-dir", default="datasets/original/moyo_cam05_centerframe/")
    parser.add_argument("--images-subdir", default="images")
    parser.add_argument("--output-subdir", default="cropped_images")
    args = parser.parse_args()


    os.system('mkdir \"'+os.path.join(args.moyo_dir, 'test', args.images_subdir)+'\"')
    os.system('mkdir \"'+os.path.join(args.moyo_dir, 'test', args.output_subdir)+'\"')
    image_root = os.path.join(args.moyo_dir, 'images', 'train')
    val_image_root = os.path.join(args.moyo_dir, 'images', 'val')
    for split in ['trainval', 'test']:
        os.system('mkdir \"'+os.path.join(args.moyo_dir, split)+'\"')
        os.system('mkdir \"'+os.path.join(args.moyo_dir, split, args.images_subdir)+'\"')
        os.system('mkdir \"'+os.path.join(args.moyo_dir, split, args.output_subdir)+'\"')
        with open(os.path.join(args.moyo_dir, split, split+'.json')) as f:
            keys = json.load(f)
        for key in keys:
            path = glob.glob(os.path.join(args.moyo_dir, 'images', '*', '*'+'_'.join(key['key'].split('_')[:-4]), 'YOGI_Cam_05', '*'+key['key']+'.jpg'))[0]
            os.system('ln -s \"'+path+'\" \"'+os.path.join(args.moyo_dir, split, args.images_subdir, key['key']+'.jpg'))

            img = Image.open(os.path.join(args.moyo_dir, split, args.images_subdir, key['key']+'.jpg'))
            path = os.path.join(args.moyo_dir, args.images_subdir, '../vitpose', key+'_keypoints.json')
            if not os.path.exists(path):
                path = path.replace('/vitpose/', '/openpose/')
                print('using openpose')
            else:
                print('using vitpose')
            # path = '/home/sanjayss/buddi/moyo_cam05_centerframe_trainval/vitpose/'+key+'_keypoints.json'
            if os.path.exists(path):
                # with open('/home/sanjayss/buddi/moyo_cam01_trainval/openpose/'+key+'.json') as f:
                with open(path) as f:
                    keypoints_dict = json.load(f)
                bbox = None
                for x in keypoints_dict['people']:
                    keypoints = x['pose_keypoints_2d']
                    kpts = np.array(keypoints).reshape(-1, 3)
                    conf = kpts[:,-1]
                    x0, y0, _ = kpts[conf > 0].min(0)
                    x1, y1, _ = kpts[conf > 0].max(0)
                    if bbox is None or (y1-y0)*(x1-x0) > (bbox[3]-bbox[1])*(bbox[2]-bbox[0]):
                        bbox = [x0, y0, x1, y1]
                bbox = [bbox[0]-80, bbox[1]-80, bbox[2]+80, bbox[3]+80]
                img = img.crop(bbox)
            else:
                print('no keypoints')
            img.save(os.path.join(args.moyo_dir, split, args.output_subdir, key['key']+'.jpg'))
