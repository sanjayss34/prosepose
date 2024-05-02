import argparse
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
import math
import smplx 
import shutil
import trimesh
import pickle as pkl
from llib.methods.hhcs_optimization.evaluation.utils import *
from llib.utils.threed.distance import pcl_pcl_pairwise_distance
from llib.defaults.main import (
    config as default_config,
    merge as merge_configs
)

ESSENTIALS_HOME = os.environ['ESSENTIALS_HOME']
PROJECT_HOME = '/is/cluster/lmueller2/projects/HumanHumanContact/humanhumancontact' #os.environ['HUMANHUMANCONTACT_HOME']
REGION_TO_VERTEX_PATH = osp.join(ESSENTIALS_HOME, 'contact/flickrci3ds_r75_rid_to_smplx_vid.pkl')
J14_REGRESSOR_PATH = f'{ESSENTIALS_HOME}/body_model_utils/joint_regressors/SMPLX_to_J14.pkl'
J14_REGRESSOR = torch.from_numpy(
    pickle.load(open(J14_REGRESSOR_PATH, 'rb'), encoding='latin1')).to('cuda').float()

# SMPL JOINT REGRESSOR
# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints

# Indices to get the 14 LSP joints from the ground truth SMPL joints
jreg_path = f'{ESSENTIALS_HOME}/body_model_utils/joint_regressors/J_regressor_h36m.npy'
SMPL_TO_H36M = torch.from_numpy(np.load(jreg_path)).to('cuda').float()
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

#GT_DATA = pickle.load(open(f'{PROJECT_HOME}/datasets/processed/CHI3D/val.pkl', 'rb'))
#INTERACTION_SIGNATURE = json.load(
#    open(f'{PROJECT_HOME}/datasets/processed/CHI3D/train/s02/interaction_contact_signature.json', 'rb'))
#IMAGE_FOLDER = f'{PROJECT_HOME}/datasets/processed/CHI3D/train/s02/images_contact/'

# We need this to only evaluate on images that do not miss keypoint/BEV detections
BEV_RESULT_FOLDER = '/is/cluster/work/lmueller2/results/HHC/optimization/cvpr2023/bev_estimates'
ORIG_DATA_FOLDER = 'datasets/original/CHI3D'
PROCESSED_DATA_FOLDER = 'datasets/processed/CHI3D'

PROCESSED_DATA = pickle.load(open(f'{PROCESSED_DATA_FOLDER}/train/images_contact_processed.pkl', 'rb'))
the_split = 'val'
TRAIN_VAL_SPLIT = np.load(f'{PROCESSED_DATA_FOLDER}/train/train_{the_split}_split.npz')
POSTPROCESSED_DATA = {datum['img_out_fn']: datum for datum in pickle.load(open(f'{PROCESSED_DATA_FOLDER}/{the_split}_optimization.pkl', 'rb'))}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg', 
        type=str, dest='exp_cfgs', nargs='+', default='llib/methods/hhcs_optimization/evaluation/chi3d_eval.yaml', 
        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
        nargs='*', help='The configuration of the Detector') 
    parser.add_argument('--eval-split', default='train', type=str, choices=['train', 'val'])
    parser.add_argument('--print_result', action='store_true', default=False, help='Print the result to the console')
    parser.add_argument('--other-results-pkl')
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
        transl = to_tensor(human[f'transl']),
    )
    for key in params:
        print(key, params[key].shape)

    verts, joints = None, None
    if body_model_smplx is not None:
        body = body_model_smplx(**params)
        verts = body.vertices.detach() 
        joints = torch.matmul(J14_REGRESSOR, verts)

    return params, verts, joints



def main(cfg, cmd_args):

    # cmd args and logging
    subjects = TRAIN_VAL_SPLIT[cmd_args.eval_split]
    split_folder = 'train' if cmd_args.eval_split in ['train', 'val'] else 'test'
    actions, subjects_ll, img_names = [], [], []

    # build metrics 
    contact_metric = build_metric(cfg.evaluation.cmap_iou)
    scale_mpjpe_metric = build_metric(cfg.evaluation.scale_mpjpe)
    mpjpe_metric = build_metric(cfg.evaluation.mpjpe)
    pa_mpjpe_metric = build_metric(cfg.evaluation.pa_mpjpe)
    pairwise_pa_mpjpe_metric = build_metric(cfg.evaluation.pairwise_pa_mpjpe)

    # SMPL model for ground truth
    bm_smplx = smplx.create(
        model_path=osp.join(ESSENTIALS_HOME, 'body_models'), 
        model_type='smplx',
        batch_size=2
    ).to(cfg.device)

    bm_smpl = smplx.create(
        model_path=osp.join(ESSENTIALS_HOME, 'body_models'), 
        model_type='smpl',
        batch_size=2
    ).to(cfg.device)

    cmapper = ContactMap(
        region_to_vertex=REGION_TO_VERTEX_PATH,
    )

    results = ResultLogger(
        method_names=['est'],
        output_fn=f'{BEV_RESULT_FOLDER}/results.pkl'
    )

    img_names_to_include = None
    if cmd_args.other_results_pkl is not None:
        with open(cmd_args.other_results_pkl, 'rb') as f:
            other_res = pkl.load(f)
        img_names_to_include = set(other_res['info']['img_names'])

    results.info = {
        'actions': [],
        'subjects': [],
        'contact_counts': [],
        'img_names': [],
    }
    ITEMS_TO_EVAL = chi3d_items_for_eval(
        subjects, split_folder, ORIG_DATA_FOLDER, PROCESSED_DATA
    )

    with open(os.path.join(os.environ['ESSENTIALS_HOME'], 'smpl_to_smplx', 'smpl_to_smplx.pkl'), 'rb') as f:
        SMPL_TO_SMPLX = pickle.load(f)['matrix']
    SMPL_TO_SMPLX = torch.from_numpy(SMPL_TO_SMPLX).unsqueeze(0).to(cfg.device).repeat(2, 1, 1).float() # [2, SMPLX vertices, SMPL vertices]

    for subject, actions_dict in tqdm(ITEMS_TO_EVAL.items()):
        annotation_fn =  osp.join(
            ORIG_DATA_FOLDER, split_folder, subject, 'interaction_contact_signature.json'
        )
        annotations = json.load(open(annotation_fn, 'r'))

        #for action, annotation in tqdm(annotations.items()):
        for action, cameras in tqdm(actions_dict.items()):

            orig_subject_folder = osp.join(ORIG_DATA_FOLDER, split_folder, subject)
            processed_subject_folder = osp.join(PROCESSED_DATA_FOLDER, split_folder, subject)

            # gt annotation data
            annotation = annotations[action]
            frame_id = annotation['fr_id']

            # load SMPL params
            smpl_path = f'{orig_subject_folder}/smplx/{action}.json'
            params_gt, verts_gt, joints_gt = chi3d_get_smplx_gt(smpl_path, [frame_id], bm_smplx)
            verts_gt = torch.from_numpy(verts_gt).to(cfg.device).float()

            # gt contact map
            region_id = annotation[f'smplx_signature']['region_id']        
            gt_contact_map = np.zeros((75, 75)).astype(bool)
            for rid in region_id:
                gt_contact_map[rid[0], rid[1]] = True

            for cam in cameras:
                img_name = f'{subject}_{action}_{frame_id:06d}_{cam}_0'
                #img_path = osp.join(processed_subject_folder, 'images_contact', img_name+'.png')
                img_names.append(img_name)
                actions.append(img_name.split('_')[1].split(' ')[0])
                subjects_ll.append(subject)

                if img_names_to_include is not None and img_name not in img_names_to_include:
                    continue
                # get camera params and convert vertices to camera coordinate system
                cam_path = f'{orig_subject_folder}/camera_parameters/{cam}/{action}.json'
                cam_params = chi3d_read_cam_params(cam_path)
                verts_gt_camera =  chi3d_verts_world2cam(verts_gt, cam_params)
                gt_smplx_joints = [
                    verts2joints(verts_gt_camera[0][None], J14_REGRESSOR), 
                    verts2joints(verts_gt_camera[1][None], J14_REGRESSOR)
                ]

                gt_smplx_vertices = [verts_gt_camera[0], verts_gt_camera[1]]
                
                # get bev predictions
                bev_vertices = chi3d_bev_verts_from_processed_data(PROCESSED_DATA, subject, action, cam, frame_id)
                """bev_vertices = get_smplx_pred(
                    {
                        key: POSTPROCESSED_DATA[img_name][key].squeeze(1)
                        for key in ['transl', 'body_pose', 'scale', 'betas', 'global_orient']
                    }, bm_smplx
                )[1]
                est_smplx_joints = [verts2joints(bev_vertices[0][None], J14_REGRESSOR), verts2joints(bev_vertices[1][None], J14_REGRESSOR)]"""
                est_smplx_joints = [
                    verts2joints(bev_vertices[0][None], SMPL_TO_H36M)[:,H36M_TO_J14, :], 
                    verts2joints(bev_vertices[1][None], SMPL_TO_H36M)[:,H36M_TO_J14, :]
                ]

                dists = pcl_pcl_pairwise_distance(
                    gt_smplx_vertices[0][None].cpu(), gt_smplx_vertices[1][None].cpu(), squared=False)
                
                # add some metadata 
                thres = 0.1
                results.info['contact_counts'].append((dists < thres).sum().item())
                results.info['img_names'].append(img_name)
                results.info['actions'].append(img_name.split('_')[1].split(' ')[0])
                results.info['subjects'].append(subject)
                # PCC (not implemented for BEV because bev estimate is SMPL)
                #pgt_cmap_heat = cmapper.get_full_heatmap(verts_gt_camera[0], verts_gt_camera[1])
                #max_points = gt_contact_map.sum()
                #distances = pgt_cmap_heat[gt_contact_map]
                #batch_dist = distances[None].repeat(len(results.pcc_x), 1) \
                #    < results.pcc_x.unsqueeze(1).repeat(1, max_points)
                #pcc = batch_dist.sum(1) / max_points
                #results.output['est_pcc'].append(pcc)
                # PCC (not implemented for BEV because bev estimate is SMPL)
                # pgt_cmap_heat = cmapper.get_full_heatmap(gt_vertices[0][None], gt_smplx_vertices[1][None]).view(*gt_contact_map.shape)
                bev_smplx_vertices = torch.bmm(SMPL_TO_SMPLX, bev_vertices)
                pred_cmap_heat = cmapper.get_full_heatmap(bev_smplx_vertices[0][None], bev_smplx_vertices[1][None]).view(*gt_contact_map.shape)
                max_points = gt_contact_map.sum()
                # max_points = pgt_cmap_heat.sum()
                distances = pred_cmap_heat[gt_contact_map]
                # distances = pred_cmap_heat[pgt_cmap_heat]
                batch_dist = distances[None].repeat(len(results.pcc_x), 1) \
                    < results.pcc_x.unsqueeze(1).repeat(1, max_points)
                pcc = batch_dist.sum(1) / max_points
                results.output['est_pcc'].append(pcc)

                onetwo = pairwise_pa_mpjpe_metric(torch.cat(est_smplx_joints, dim=1).cpu().numpy(),torch.cat(gt_smplx_joints, dim=1).cpu().numpy()).mean()
                twoone = pairwise_pa_mpjpe_metric(torch.cat(est_smplx_joints, dim=1).cpu().numpy(),torch.cat([gt_smplx_joints[1], gt_smplx_joints[0]], dim=1).cpu().numpy()).mean()
                #print(cam, onetwo, twoone)
                if twoone < onetwo:
                    gt_smplx_joints = [gt_smplx_joints[1], gt_smplx_joints[0]]
                    gt_smplx_vertices = [gt_smplx_vertices[1], gt_smplx_vertices[0]]

                # compute errors
                results.output[f'est_mpjpe_h0'].append(
                    mpjpe_metric(est_smplx_joints[0], gt_smplx_joints[0]).mean())
                results.output[f'est_mpjpe_h1'].append(
                    mpjpe_metric(est_smplx_joints[1], gt_smplx_joints[1]).mean())
                results.output[f'est_scale_mpjpe_h0'].append(
                    scale_mpjpe_metric(est_smplx_joints[0].cpu().numpy(), gt_smplx_joints[0].cpu().numpy()).mean())
                results.output[f'est_scale_mpjpe_h1'].append(
                    scale_mpjpe_metric(est_smplx_joints[1].cpu().numpy(), gt_smplx_joints[1].cpu().numpy()).mean())
                print(est_smplx_joints[0].shape, est_smplx_joints[1].shape)
                results.output[f'est_pa_mpjpe_h0'].append(
                    pa_mpjpe_metric(est_smplx_joints[0].cpu().numpy(), gt_smplx_joints[0].cpu().numpy()).mean())
                results.output[f'est_pa_mpjpe_h1'].append(
                    pa_mpjpe_metric(est_smplx_joints[1].cpu().numpy(), gt_smplx_joints[1].cpu().numpy()).mean())
                results.output[f'est_pa_mpjpe_h0h1'].append(pairwise_pa_mpjpe_metric(
                    torch.cat(est_smplx_joints, dim=1).cpu().numpy(), 
                    torch.cat(gt_smplx_joints, dim=1).cpu().numpy()).mean())
                
                # save meshed dor
                # for iii in [0,1]:
                    # save_mesh(bev_vertices[iii].cpu().numpy(), bm_smpl.faces, f'outdebug/bev_test/{subject}_{action}_{cam}_bev_mesh_{iii}.ply')
                    # save_mesh(verts_gt_camera[iii].cpu().numpy(), bm_smplx.faces, f'outdebug/bev_test/{subject}_{action}_{cam}_test_mesh_gt_cam_{iii}.ply')

    for metric in ['est_mpjpe_h0', 'est_mpjpe_h1', 'est_pa_mpjpe_h0', 'est_pa_mpjpe_h1', 'est_pa_mpjpe_h0h1']:
        for action in sorted(set(actions)):
            results.get_action_mean(metric, actions, action)

    for metric in ['est_mpjpe_h0', 'est_mpjpe_h1', 'est_pa_mpjpe_h0', 'est_pa_mpjpe_h1', 'est_pa_mpjpe_h0h1']: 
        for subject in sorted(set(subjects_ll)):
            results.get_subject_mean(metric, subjects_ll, subject)

    results.topkl(print_result=cmd_args.print_result)


if __name__ == "__main__":

    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
