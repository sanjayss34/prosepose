import argparse
import ipdb
import glob
import torch 
import numpy as np
import cv2
import os
from tqdm import tqdm
import os.path as osp
import pickle
import json
from llib.visualization.utils import *
from llib.utils.metrics.build import build_metric
from llib.bodymodels.build import build_bodymodel
from llib.utils.threed.distance import ContactMap
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms import euler_angles_to_matrix
import math
import smplx 
import shutil
import matplotlib.pyplot as plt
from llib.methods.hhcs_optimization.evaluation.utils import *

from llib.utils.threed.distance import pcl_pcl_pairwise_distance
from llib.defaults.main import (
    config as default_config,
    merge as merge_configs
)
from llib.utils.selfcontact import SelfContact

ESSENTIALS_HOME = os.environ['ESSENTIALS_HOME']
PROJECT_HOME = '/is/cluster/lmueller2/projects/HumanHumanContact/humanhumancontact' #os.environ['HUMANHUMANCONTACT_HOME']
REGION_TO_VERTEX_PATH = osp.join(ESSENTIALS_HOME, 'contact/flickrci3ds_r75_rid_to_smplx_vid.pkl')
with open(REGION_TO_VERTEX_PATH, 'rb') as f:
    rid2vid = pickle.load(f)


J14_REGRESSOR_PATH = f'{ESSENTIALS_HOME}/body_model_utils/joint_regressors/SMPLX_to_J14.pkl'
J14_REGRESSOR = torch.from_numpy(
    pickle.load(open(J14_REGRESSOR_PATH, 'rb'), encoding='latin1')).to('cuda').float()

# Indices to get the 14 LSP joints from the ground truth SMPL joints
jreg_path = f'{ESSENTIALS_HOME}/body_model_utils/joint_regressors/J_regressor_h36m.npy'
SMPL_TO_H36M = torch.from_numpy(np.load(jreg_path)).to('cuda').float()
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

# We need this to only evaluate on images that do not miss keypoint/BEV detections
ORIG_DATA_FOLDER = 'datasets/original/CHI3D'
PROCESSED_DATA_FOLDER = 'datasets/processed/CHI3D'

PROCESSED_DATA = pickle.load(open(f'{PROCESSED_DATA_FOLDER}/train/images_contact_processed.pkl', 'rb'))
TRAIN_VAL_SPLIT = np.load(f'{PROCESSED_DATA_FOLDER}/train/train_val_split.npz')

def get_contact_map(vertices, sc_module, euclthres=None):
    vertices = (vertices - vertices.mean(0)).unsqueeze(0)

    # Segment mesh into inside and outside vertices
    verts_distances \
    = sc_module.segment_vertices(
        vertices,
        compute_hd=False,
        test_segments=False,
        return_pair_distances=True)
    if euclthres is None:
        verts_in_contact = (verts_distances < sc_module.euclthres)
        verts_in_contact = verts_in_contact.squeeze(0)
        assert verts_in_contact.shape[0] == verts_in_contact.shape[1] == vertices.shape[1]
        contact_map = torch.tensor([[verts_in_contact[rid2vid[r1]][:,rid2vid[r2]].any() and r1 != r2 for r2 in rid2vid] for r1 in rid2vid])
        return contact_map
    verts_in_contact = [(verts_distances < thres).squeeze(0) for thres in euclthres]
    assert all([mat.shape[0] == mat.shape[1] == vertices.shape[1] for mat in verts_in_contact])
    contact_maps = [torch.tensor([[mat[rid2vid[r1]][:,rid2vid[r2]].any() and r1 != r2 for r2 in rid2vid] for r1 in rid2vid]) for mat in verts_in_contact]
    return contact_maps

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg', 
        type=str, dest='exp_cfgs', nargs='+', default='llib/methods/hhcs_optimization/evaluation/chi3d_eval.yaml', 
        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
        nargs='*', help='The configuration of the Detector') 
    parser.add_argument('--predictions-folder', type=str, 
        default=f'{PROJECT_HOME}/results/HHC/optimization/fit_mocap_flickrci3ds_test_v02')
    parser.add_argument('--eval-split', default='train', type=str, choices=['train', 'val'])
    parser.add_argument('--print_result', action='store_true', default=False, help='Print the result to the console')
    parser.add_argument('--img-names-to-evaluate-path')
    parser.add_argument('--backoff-predictions-folder')
    parser.add_argument('--bev-backoff', action='store_true')
    parser.add_argument('--bev-pkl-path')
    parser.add_argument('--per-img-pcc', type=float, default=None)
    cmd_args = parser.parse_args()

    cfg = merge_configs(cmd_args, default_config)

    return cfg, cmd_args



def get_smplx_pred(human, body_model_smplx=None):
    """
    Returns the SMPL parameters of a human.
    """
    def to_tensor(x):
        return torch.tensor(x).to('cuda')

    params = dict(
        #betas = torch.cat([to_tensor(human[f'betas']),to_tensor(human[f'scale'])], dim=1),
        global_orient = to_tensor(human[f'global_orient']),
        body_pose = to_tensor(human[f'body_pose']),
        betas = to_tensor(human[f'betas']),
        scale = to_tensor(human[f'scale']),
    )

    verts, joints = None, None
    if body_model_smplx is not None:
        body = body_model_smplx(**params)
        verts = body.vertices.detach() 
        joints = torch.matmul(J14_REGRESSOR, verts)

    return params, verts, joints

def get_smpl_pred(human, body_model):
    def to_tensor(x):
        return torch.tensor(x).to('cuda')
    params = dict(
        global_orient = to_tensor(human['bev_smpl_global_orient']),
        body_pose = to_tensor(human['bev_smpl_body_pose']),

        betas = to_tensor(human['bev_smpl_betas']),
        scale = to_tensor(human['bev_smpl_scale']),
        transl = to_tensor(human['bev_smplx_transl'])
    )
    body = body_model(**params)
    return body.vertices.detach()

def main(cfg, cmd_args):

    PREDICTIONS_FOLDER = cmd_args.predictions_folder

    # cmd args and logging
    # actions, subjects_ll, cc, img_names = [], [], [], []

    # build metrics 
    contact_metric = build_metric(cfg.evaluation.cmap_iou)
    scale_mpjpe_metric = build_metric(cfg.evaluation.scale_mpjpe)
    mpjpe_metric = build_metric(cfg.evaluation.mpjpe)
    pa_mpjpe_metric = build_metric(cfg.evaluation.pa_mpjpe)

    # SMPL model for predictions
    model_folder = osp.join(ESSENTIALS_HOME, 'body_models')
    kid_template = osp.join(model_folder, 'smil/smplx_kid_template.npy')
    bm_smplxa = smplx.create(
        model_path=model_folder, 
        model_type='smplx',
        kid_template_path=kid_template, 
        age='kid',
        batch_size=1
    ).to(cfg.device)

    bm_smplxa = build_bodymodel(
        cfg=cfg.body_model, 
        batch_size=1, 
        device=cfg.device
    )

    # SMPL model for ground truth
    bm_smplx = smplx.create(
        model_path=model_folder,
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
    ).to(cfg.device)

    bm_smpl = smplx.create(
        model_path=model_folder,
        model_type='smpl',
        num_betas=10,
        batch_size=1,
        gender='neutral',
    ).to(cfg.device)

    results = ResultLogger(
        method_names=['est'],
        output_fn=f'{PREDICTIONS_FOLDER}/tmp_results2.pkl' if not cmd_args.bev_backoff else f'{PREDICTIONS_FOLDER}/tmp_results_bev_backoff.pkl'
    )
    results.info = {
        'actions': [],
        'subjects': [],
        'contact_counts': [],
        'img_names': [],
        'est_img_pcc': [],
    }

    cmapper = ContactMap(
        region_to_vertex=REGION_TO_VERTEX_PATH,
    )

    SMPLX_GT_DIR = f'/scratch/partial_datasets/moyo/MOYO/20220923_20220926_with_hands/AMASS/YOGI_2_latest_smplx_neutral/train/'

    sc_module = SelfContact( 
        essentials_folder='./selfcontact-essentials/',
        geothres=0.3, 
        euclthres=0.02, 
        model_type='smplx',
        test_segments=True,
        compute_hd=False
    )

    fnames_to_evaluate = None
    if cmd_args.img_names_to_evaluate_path is not None:
        with open(cmd_args.img_names_to_evaluate_path) as f:
            fnames_to_evaluate = json.load(f)

    if cmd_args.backoff_predictions_folder is None:
        cmd_args.backoff_predictions_folder = PREDICTIONS_FOLDER

    bev_pkl = None
    if cmd_args.bev_pkl_path is not None:
        with open(cmd_args.bev_pkl_path, 'rb') as f:
            bev_pkl = pickle.load(f)
        bev_pkl = {datum['imgpath'].split('/')[-1].split('.jpg')[0]+'_'+str(datum['contact_index']): datum for datum in bev_pkl}
    with open(os.path.join(os.environ['ESSENTIALS_HOME'], 'smpl_to_smplx', 'smpl_to_smplx.pkl'), 'rb') as f:
        SMPL_TO_SMPLX = pickle.load(f)['matrix']
    SMPL_TO_SMPLX = torch.from_numpy(SMPL_TO_SMPLX).unsqueeze(0).to(cfg.device).float() # [2, SMPLX vertices, SMPL vertices]
    for fname in os.listdir(osp.join(cmd_args.backoff_predictions_folder, 'results')):
        if fname[-4:] == '.pkl':
            if fnames_to_evaluate is not None and fname.replace('.pkl', '.png') not in fnames_to_evaluate:
                continue
            # if 'Camel_Pose_or_Ustrasana_-b_YOGI_Cam_05_0468' not in fname:
            #     continue
            # if fname.split('.')[0] in ['Plow_Pose_or_Halasana_-b_YOGI_Cam_05_0606_0', 'Child_Pose_or_Balasana_-a_YOGI_Cam_05_0362_0', 'Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a_YOGI_Cam_05_0262_0']:
            #     continue
            print(fname)
            # predicted data
            backoff = False
            if osp.exists(osp.join(PREDICTIONS_FOLDER, 'results', fname)):
                with open(osp.join(PREDICTIONS_FOLDER, 'results', fname), 'rb') as f:
                    pred_item = pickle.load(f)
                    # print('here1')
            else:
                backoff = True
                with open(osp.join(cmd_args.backoff_predictions_folder, 'results', fname), 'rb') as f:
                    pred_item = pickle.load(f)
                    # print('here2')
            params_pred, verts_pred, joints_pred = get_smplx_pred(pred_item['humans'], bm_smplxa)
            base_name = fname.split('_YOGI_Cam')[0]
            # print(fname)
            # print(os.path.join(SMPLX_GT_DIR, '*'+base_name+'*'))
            smplx_fname = list(glob.glob(os.path.join(SMPLX_GT_DIR, '*'+base_name+'*')))
            if len(smplx_fname) == 0:
                smplx_fname = list(glob.glob(os.path.join(SMPLX_GT_DIR.replace('/train/', '/val/'), '*'+base_name+'*')))
                if len(smplx_fname) == 0:
                    print('MISSING', fname)
                    continue
            smplx_fname = smplx_fname[0]
            frame_id = int(fname.split('_')[-2])
            params_gt, verts_gt, joints_gt = moyo_get_smplx_gt(smplx_fname, [frame_id*2], bm_smplx)
            verts_gt = torch.from_numpy(verts_gt).to(cfg.device).float()
            day = smplx_fname.split('/')[-1].split('_')[0]
            cam_path = f'/scratch/partial_datasets/moyo/MOYO/20220923_20220926_with_hands/cameras/20{day}/{day}_Afternoon_PROCESSED_CAMERA_PARAMS/cameras_param.json'
            if not os.path.exists(cam_path):
                cam_path = cam_path.replace('Afternoon', 'Morning')
            cam_params = moyo_read_cam_params(cam_path, int(fname.split('_')[-3]))
            verts_gt_camera = chi3d_verts_world2cam(verts_gt, cam_params)
            # ipdb.set_trace()

            if bev_pkl is not None and (backoff or not cmd_args.bev_backoff):
                # verts_pred = torch.from_numpy(bev_pkl[fname.split('.')[0]]['bev_smplx_vertices']).to(verts_gt_camera.device)
                verts_pred = get_smpl_pred(bev_pkl[fname.split('.')[0]], bm_smpl).to(verts_gt_camera.device)
                bev_smplx_vertices = torch.bmm(SMPL_TO_SMPLX, verts_pred)
                # bev_smplx_vertices = verts_pred
                # est_smplx_vertices = [verts_pred[0]]
                est_smplx_joints = [
                    verts2joints(verts_pred[0][None], SMPL_TO_H36M)[:,H36M_TO_J14, :], 
                ]
                # est_smplx_joints = [verts2joints(verts_pred[0][None], J14_REGRESSOR)]
            else:
                print('not bev')
                est_smplx_vertices = [verts_pred[0]]
                est_smplx_joints = [verts2joints(verts_pred[0][None], J14_REGRESSOR)]
            print(verts_pred.shape)

            gt_smplx_vertices = [verts_gt_camera[0]]
            gt_smplx_joints = [verts2joints(verts_gt_camera[0][None], J14_REGRESSOR)]

            if True:
                gt_in_contact = get_contact_map(gt_smplx_vertices[0], sc_module)

                # ipdb.set_trace()
                gt_num_contact_points = gt_in_contact.long().sum().item() / 2
                print('num gt contact points', gt_num_contact_points)
                results.info['contact_counts'].append(gt_num_contact_points)
                if gt_num_contact_points > 0:
                    if bev_pkl is None:
                        pred_in_contacts = get_contact_map(est_smplx_vertices[0], sc_module, euclthres=results.pcc_x.reshape(-1).tolist())
                    else:
                        pred_in_contacts = get_contact_map(bev_smplx_vertices[0], sc_module, euclthres=results.pcc_x.reshape(-1).tolist())
                    pcc = [(in_contact.long()*gt_in_contact.long()).sum().item() / (2*gt_num_contact_points) for in_contact in pred_in_contacts]
                    print(pcc)
                else:
                    pcc = [-1 for _ in results.pcc_x.reshape(-1).tolist()]
                results.output['est_pcc'].append(np.array(pcc))
                if cmd_args.per_img_pcc is not None:
                    img_pcc = [1 if val >= cmd_args.per_img_pcc else 0 for val in pcc]
                    results.info['est_img_pcc'].append(np.array(img_pcc))

            results.output[f'est_mpjpe_h0'].append(
                mpjpe_metric(est_smplx_joints[0], gt_smplx_joints[0]).mean())
            results.output[f'est_scale_mpjpe_h0'].append(
                scale_mpjpe_metric(est_smplx_joints[0].cpu().numpy(), gt_smplx_joints[0].cpu().numpy()).mean())
            results.output[f'est_pa_mpjpe_h0'].append(
                pa_mpjpe_metric(est_smplx_joints[0].cpu().numpy(), gt_smplx_joints[0].cpu().numpy()).mean())
            results.info['img_names'].append(fname)

    results.topkl(print_result=cmd_args.print_result)
    print('--------')
    print('Contact subset results')
    contact_indices = [i for i in range(len(results.info['img_names'])) if results.info['contact_counts'][i] > 0]
    print('est_pcc', np.stack([results.output['est_pcc'][i] for i in contact_indices]).mean())
    for key in ['est_mpjpe_h0', 'est_scale_mpjpe_h0', 'est_pa_mpjpe_h0']:
        print(key, np.mean([results.output[key][i] for i in contact_indices]))

if __name__ == "__main__":

    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
