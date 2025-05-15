import os
import pickle as pkl
import json
import glob
import argparse
import torch
from tqdm import tqdm
import numpy as np
import smplx
from llib.utils.threed.distance import ContactMap
from llib.utils.selfcontact import SelfContact
from llib.methods.hhcs_optimization.evaluation.utils import moyo_get_smplx_gt, moyo_read_cam_params, chi3d_verts_world2cam, verts2joints
from llib.methods.hhcs_optimization.evaluation.moyo_eval import get_contact_map

IMAGE_PROMPT = "Describe the contact points between the pair of people."
FINE_GRAINED_NAMES = {
    0: "left upper belly",
    1: "left foot",
    2: "back left shoulder",
    3: "left fingers",
    4: "front right upper belly",
    5: "front left forearm",
    6: "back right upper arm",
    7: "right upper chest",
    8: "back right forearm",
    9: "right butt",
    10: "right back",
    11: "back left elbow",
    12: "back right knee",
    13: "right middle back",
    14: "back left neck",
    15: "back left upper thigh",
    16: "front right knee",
    17: "left hand",
    18: "back right lower leg",
    19: "front right upper thigh",
    20: "front left lower thigh",
    21: "back left lower leg",
    22: "front right shin",
    23: "right palm",
    24: "back right elbow",
    25: "front left shin",
    26: "right fingers",
    27: "right hand",
    28: "back right lower thigh",
    29: "back right shoulder",
    30: "back left knee",
    31: "front right shoulder",
    32: "front right forearm",
    33: "left fingers",
    34: "back left elbow",
    35: "left middle back",
    36: "back right shin",
    37: "front right upper arm",
    38: "right upper back",
    39: "back left skull",
    40: "right fingers",
    41: "left upper back",
    42: "right foot",
    43: "back right neck",
    44: "front right elbow",
    45: "front left upper thigh",
    46: "front right upper thigh",
    47: "front right neck",
    48: "left lower back",
    49: "front left upper arm",
    50: "front pelvis",
    51: "front right skull",
    52: "back right skull",
    53: "back left lower thigh",
    54: "front left neck",
    55: "back right upper thigh",
    56: "back left shin",
    57: "back left upper arm",
    58: "right upper chest",
    59: "left upper chest",
    60: "front left lower leg",
    61: "left lower chest",
    62: "front right lower leg",
    63: "right upper back",
    64: "front left knee",
    65: "right foot",
    66: "front left skull",
    67: "left foot",
    68: "right lower belly",
    69: "back left forearm",
    70: "left upper back",
    71: "left butt",
    72: "left palm",
    73: "front left shoulder",
    74: "left lower belly",
}
INV_COARSE_REGION_NAMES = {
    'left hand': [3, 33, 72],
    'right hand': [23, 26, 27, 40],
    'left arm': [5, 11, 17, 34, 49, 57, 69],
    'right arm': [6, 8, 24, 32, 37, 44],
    'left foot': [1, 67],
    'right foot': [42, 65],
    'left leg': [15, 20, 21, 25, 30, 45, 53, 56, 60, 64],
    'right leg': [12, 16, 18, 19, 22, 28, 36, 46, 55, 62],
    'back': [13, 35, 38, 41, 63, 70, 2, 29, 10, 48, 14, 43],
    'head': [39, 51, 52, 66],
    'neck': [14, 43, 47, 54],
    'butt': [9, 71],
    'waist': [10, 48, 50, 68, 74],
    'waist (back)': [10, 48],
    'waist (front)': [50, 68, 74],
    'left shoulder (front)': [73],
    'left shoulder (back)': [2],
    'right shoulder (front)': [31],
    'right shoulder (back)': [29],
    'left shoulder': [2, 73],
    'right shoulder': [29, 31],
    'chest': [7, 58, 59, 61],
    'stomach': [0, 4, 68, 74]
}
COARSE_REGION_NAMES = {
  i: [key for key in INV_COARSE_REGION_NAMES if i in INV_COARSE_REGION_NAMES[key]][0]
  for i in FINE_GRAINED_NAMES
}

def hi4d_get_smplx_gt(frame_ids, gt_params, body_model, num_betas=10):

    num_frames = len(frame_ids)

    params = {
        'global_orient': torch.from_numpy(gt_params['smplx_global_orient_unit']),
        'transl': torch.from_numpy(gt_params['smplx_transl_unit']),
        'body_pose': torch.from_numpy(gt_params['smplx_body_pose']).view(num_frames, 2, -1),
        'betas': torch.from_numpy(gt_params['smplx_betas'])[:,:,:num_betas],
    }
    for key in params:
        if key != 'betas':
            assert params[key].shape[0] == num_frames, key

    vertices = np.zeros((2, num_frames, 10475, 3)) #body.vertices.detach().to('cpu').numpy()[0]
    joints = np.zeros((2, num_frames, 127, 3)) #body.joints.detach().to('cpu').numpy()[0]

    for array_idx in range(num_frames):
        params_for_smpl = {} 
        for key, val in params.items():
            params_for_smpl[key] = val[array_idx, :, :].to('cuda')
        body = body_model(**params_for_smpl)
        vertices[:, array_idx] = body.vertices.detach().to('cpu').numpy()
        joints[:, array_idx] = body.joints.detach().to('cpu').numpy()
    
    return params, vertices, joints

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-name")
    parser.add_argument("--contact-threshold", type=float, default=0.01)
    parser.add_argument("--selfcontact-threshold", type=float, default=0.02)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    split_name = args.split_name
    with open('datasets/original/FlickrCI3D_Signatures/train/interaction_contact_signature.json') as f:
      flickr_trainval_contacts = json.load(f)
    with open('datasets/original/FlickrCI3D_Signatures/test/interaction_contact_signature.json') as f:
      flickr_test_contacts = json.load(f)
    flickr_split = np.load('datasets/processed/FlickrCI3D_Signatures/train/train_val_split.npz')
    flickr_split = {key: flickr_split[key] for key in flickr_split.files}
    flickr_split['test'] = [name for name in flickr_test_contacts]
    flickr_root = 'datasets/original/FlickrCI3D_Signatures/'
    data = {split_name: [] for split_name in ['train']+[s+'_'+d for s in ['val', 'test'] for d in ['flickr', 'chi3d', 'hi4d']]}
    flickr_contacts = {'train': flickr_trainval_contacts, 'val': flickr_trainval_contacts, 'test': flickr_test_contacts}
    chi3d_split = {'train': 's04', 'val': 's02', 'test': 's03'}
    chi3d_processed_path = f'datasets/processed/CHI3D/{"trainval" if args.split_name == "val" else "testval_full"}_optimization.pkl'
    with open(chi3d_processed_path, 'rb') as f:
      chi3d_processed = pkl.load(f)
    chi3d_annotation_fn =  os.path.join(
        'datasets/original/CHI3D/', 'train', 's02' if args.split_name == 'val' else 's03', 'interaction_contact_signature.json'
    )
    chi3d_annotations = json.load(open(chi3d_annotation_fn, 'r'))
    cmapper = ContactMap(region_to_vertex='essentials/contact/flickrci3ds_r75_rid_to_smplx_vid.pkl')
    hi4d_split = np.load('datasets/processed/Hi4D/train_val_test_split.npz')
    num_betas = 16
    bm_smplx_hi4d = smplx.create(
      model_path=os.path.join('essentials', 'body_models'),
      model_type='smplx',
      batch_size=2,
      num_betas=num_betas,
    ).to('cuda:0')
    with open('datasets/processed/Hi4D/processed_single_camera.pkl', 'rb') as f:
      hi4d_processed = pkl.load(f)
    with open('datasets/processed/Hi4D/val_optimization.pkl', 'rb') as f:
      hi4d_val_processed = pkl.load(f)
      hi4d_val_ids = set([datum['imgname'] for datum in hi4d_val_processed])
    with open('datasets/processed/Hi4D/test_optimization.pkl', 'rb') as f:
      hi4d_test_processed = pkl.load(f)
      hi4d_test_ids = set([datum['imgname'] for datum in hi4d_test_processed])

    os.makedirs(args.output_dir, exist_ok=True)
    data = {split_name+('' if split_name == 'train' else f'_{ds}'): [] for ds in ['flickr', 'chi3d', 'hi4d']}
    # FlickrCI3D
    for name in tqdm(flickr_split[split_name]):
      response = ""
      for i, pair in enumerate(flickr_contacts[split_name][name]['ci_sign']):
        ids = pair['person_ids']
        pair_response = ""
        coarse_pairs = set()
        for region_pair in pair['smplx']['region_id']:
          coarse_pairs.add((COARSE_REGION_NAMES[region_pair[0]], COARSE_REGION_NAMES[region_pair[1]]))
        for coarse_pair in coarse_pairs:
          pair_response += f' Person 1\'s {coarse_pair[0]} is touching Person 2\'s {coarse_pair[1]}.'
        response += pair_response
        pair_response = pair_response.strip()
        new_datum = {
          'question_id': name+'_'+str(i),
          'question': IMAGE_PROMPT,
          'answer': response,
          'image': os.path.join('/home/sanjays1/', 'buckets', 'data', 'pose_cropped_better2_top_labeled_images', 'flickr', name+f'_{i}.jpg')
        }
        data[split_name+('' if split_name == 'train' else '_flickr')].append(new_datum)
    # CHI3D
    contacts_processed = [datum for datum in chi3d_processed if datum['is_contact_frame']]
    for datum in contacts_processed:
      print(datum['imgname'], datum['img_out_fn'])
      contact_map = np.zeros((75, 75)).astype(bool)
      annotation = chi3d_annotations[datum['imgname'].split('_')[0]]
      region_id = annotation[f'smplx_signature']['region_id']        
      for rid in region_id:
          contact_map[rid[0], rid[1]] = True
      contact_map = torch.from_numpy(contact_map)
      region_pairs = torch.nonzero(contact_map, as_tuple=False).tolist()
      coarse_pairs = set()
      frame_response = ""
      if len(region_pairs) > 0:
        for r1, r2 in region_pairs:
          coarse_pairs.add((COARSE_REGION_NAMES[r1], COARSE_REGION_NAMES[r2]))
        for coarse_pair in coarse_pairs:
          frame_response += f"Person 1's {coarse_pair[0]} is touching Person 2's {coarse_pair[1]}. "
      else:
        frame_response += "The people are not in contact. "
      frame_response = frame_response.strip()
      new_datum = {
        'question_id': datum['img_out_fn'][4:-2],
        'question': IMAGE_PROMPT,
        'answer': frame_response,
        'image': datum['imgname'],
      }
      data[split_name+('' if split_name == 'train' else '_chi3d')].append(new_datum)
    
    # Hi4D
    for pair in tqdm(hi4d_processed):
      if pair[4:] not in hi4d_split[split_name]:
        continue
      for activity in hi4d_processed[pair]:
        cameras = [cam for cam in hi4d_processed[pair][activity]['image_data'][0]]
        params = None
        verts = None
        joints = None
        contact_frame_ids = [int(list(img[cameras[0]].keys())[0]) for img in hi4d_processed[pair][activity]['image_data']]
        assert contact_frame_ids == sorted(contact_frame_ids)
        params, verts, joints = hi4d_get_smplx_gt(contact_frame_ids, hi4d_processed[pair][activity], bm_smplx_hi4d, num_betas=num_betas)
        for camera in cameras:
          if int(camera) != 4:
            continue
          frame_fns = sorted([frame_key for f2 in range(len(hi4d_processed[pair][activity]['image_data'])) for camera in cameras for frame_key in hi4d_processed[pair][activity]['image_data'][f2][camera]])
          start_frame = int(frame_fns[0].split('.')[0])
          end_frame = int(frame_fns[-1].split('.')[0])+1
          added_frames = set()
          response = ""
          for i in range(0, end_frame-start_frame):
            if split_name == 'val' and pair.split('pair')[1]+'_'+activity+'_4_{:06d}'.format(start_frame+i) not in hi4d_val_ids:
              continue
            if split_name == 'test' and pair.split('pair')[1]+'_'+activity+'_4_{:06d}'.format(start_frame+i) not in hi4d_test_ids:
              continue
            response += f"Frame {i}\n"
            frame_response = ""
            coarse_pairs = set()
            if start_frame+i in contact_frame_ids:
              contact_index = contact_frame_ids.index(start_frame+i)
              distances = cmapper.get_full_heatmap(torch.from_numpy(verts[:1,contact_index]), torch.from_numpy(verts[1:,contact_index]))
              contact_map = distances[0] < args.contact_threshold
              region_pairs = torch.nonzero(contact_map, as_tuple=False).tolist()
              if len(region_pairs) > 0:
                for r1, r2 in region_pairs:
                  coarse_pairs.add((COARSE_REGION_NAMES[r1], COARSE_REGION_NAMES[r2]))
                for coarse_pair in coarse_pairs:
                  frame_response += f"Person 1's {coarse_pair[0]} is touching Person 2's {coarse_pair[1]}. "
              else:
                frame_response += "The people are not in contact. "
            else:
              frame_response += "The people are not in contact. "
            frame_response = frame_response.strip()
            response += frame_response
            if i+start_frame not in added_frames:
              new_datum = {
                'question_id': pair+'_'+activity+f'_{camera}_'+str(i+start_frame),
                'question': IMAGE_PROMPT,
                'answer': frame_response,
              }
              data[split_name+('' if split_name == 'train' else '_hi4d')].append(new_datum)
              added_frames.add(i+start_frame)
    columns = [f"Person {p} Body Part" for p in range(2)]
    column_width = len(columns[0])+2
    for split_key in data:
      with open(os.path.join(args.output_dir, split_key+'.json'), 'w') as fout:
        for datum in data[split_key]:
          fout.write(json.dumps(datum)+'\n')
      outputs = {}
      for res in data[split_key]:
        table = "| "+columns[0]+" | "+columns[1]+" |\n|"+'-'*column_width+'|'+'-'*column_width+'|\n'
        sentences = [s.lower().strip() for s in res['answer'].split('.') if len(s.strip()) > 0]
        for sent in sentences:
          try:
            part1 = sent.split('person 1\'s')[1].split('is touching')[0].strip()
            part2 = sent.split('person 2\'s')[1].strip()
            table += "| "+part1+" "*(column_width-len(part1)-1)+"| "+part2+" "*(column_width-len(part2)-1)+"|\n"
          except Exception as e:
            print('Error:', sent)
        key = res['question_id']
        if 'pair' in res['question_id']:
          key = key.split('_')[0].split('pair')[1]+'_'+key.split('_')[1]+'_'+key.split('_')[2]+'_{:06d}'.format(int(key.split('_')[-1]))
        outputs[key] = [{'table_response': table}]
      with open(os.path.join(args.output_dir, 'tables_'+split_key+'.json'), 'w') as fout:
          json.dump(outputs, fout)
    
    # MOYO
    sc_module = SelfContact( 
        essentials_folder='./selfcontact-essentials/',
        geothres=0.3, 
        euclthres=args.selfcontact_threshold, 
        model_type='smplx',
        test_segments=True,
        compute_hd=False
    )
    bm_smplx = smplx.create(
        model_path='essentials/body_models',
        model_type='smplx',
        num_betas=10,
        batch_size=1,
        gender='neutral',
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        use_pca=False,
        flat_hand_mean=True,
    ).to('cuda:0')
    J14_REGRESSOR_PATH = 'essentials/body_model_utils/joint_regressors/SMPLX_to_J14.pkl'
    J14_REGRESSOR = torch.from_numpy(
        pkl.load(open(J14_REGRESSOR_PATH, 'rb'), encoding='latin1')).to('cuda').float()
    SMPLX_GT_DIR = f'/datasets/moyo/current/MOYO/20220923_20220926_with_hands/AMASS/YOGI_2_latest_smplx_neutral/train/'
    with open(os.path.join('datasets', 'processed', 'moyo_cam05_centerframe', ('trainval' if args.split_name == 'val' else 'test')+'_keys.json')) as f:
        fnames = json.load(f)
    moyo_outputs = []
    for fname in fnames:
        base_name = fname.split('_YOGI_Cam')[0]
        smplx_fname = list(glob.glob(os.path.join(SMPLX_GT_DIR, '*'+base_name+'*')))
        if len(smplx_fname) == 0:
            smplx_fname = list(glob.glob(os.path.join(SMPLX_GT_DIR.replace('/train/', '/val/'), '*'+base_name+'*')))
            if len(smplx_fname) == 0:
                print('MISSING', fname)
                continue
        print(fname)
        smplx_fname = smplx_fname[0]
        frame_id = int(fname.split('_')[-1])
        params_gt, verts_gt, joints_gt = moyo_get_smplx_gt(smplx_fname, [frame_id*2], bm_smplx)
        verts_gt = torch.from_numpy(verts_gt).to('cuda:0').float()
        day = smplx_fname.split('/')[-1].split('_')[0]
        cam_path = f'/datasets/moyo/current/MOYO/20220923_20220926_with_hands/cameras/20{day}/{day}_Afternoon_PROCESSED_CAMERA_PARAMS/cameras_param.json'
        if not os.path.exists(cam_path):
            cam_path = cam_path.replace('Afternoon', 'Morning')
        cam_params = moyo_read_cam_params(cam_path, int(fname.split('_')[-2]))
        verts_gt_camera = chi3d_verts_world2cam(verts_gt, cam_params)

        gt_smplx_vertices = [verts_gt_camera[0]]
        gt_smplx_joints = [verts2joints(verts_gt_camera[0][None], J14_REGRESSOR)]

        gt_in_contact = get_contact_map(gt_smplx_vertices[0], sc_module)
        assert len(gt_in_contact.shape) == 2
        region_pairs = torch.nonzero(gt_in_contact, as_tuple=False).tolist()
        coarse_pairs = set()
        response = ""
        if len(region_pairs) > 0:
          for r1, r2 in region_pairs:
            coarse_pairs.add((COARSE_REGION_NAMES[r1], COARSE_REGION_NAMES[r2]))
          for coarse_pair in coarse_pairs:
            response += f"Person 1's {coarse_pair[0]} is touching Person 2's {coarse_pair[1]}. "
        else:
          response += "The people are not in contact. "
        moyo_outputs.append({'question_id': fname, 'answer': response.strip()})
    with open(os.path.join(args.output_dir, args.split_name+'_moyo.jsonl'), 'w') as fout:
      for datum in moyo_outputs:
        fout.write(json.dumps(datum)+'\n')
