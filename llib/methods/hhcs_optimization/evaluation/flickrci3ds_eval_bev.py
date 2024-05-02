import argparse
import numpy as np
import cv2
import torch 
import os
import json
from tqdm import tqdm
import os.path as osp
from loguru import logger as guru
import pickle
import datetime
import matplotlib.pyplot as plt
from llib.visualization.utils import *
from llib.utils.metrics.build import build_metric
from llib.bodymodels.build import build_bodymodel
from llib.utils.threed.distance import ContactMap
from llib.methods.hhcs_optimization.evaluation.utils import ResultLogger
from llib.defaults.main import (
    config as default_config,
    merge as merge_configs
)

PROJECT_HOME = '/home/sanjayss/buddi_new/' #os.environ['HUMANHUMANCONTACT_HOME']
ESSENTIALS_HOME = os.environ['ESSENTIALS_HOME']

# SMPL JOINT REGRESSOR
REGION_TO_VERTEX_PATH = osp.join(ESSENTIALS_HOME, 'contact/flickrci3ds_r75_rid_to_smplx_vid.pkl')
J14_REGRESSOR_PATH = f'{ESSENTIALS_HOME}/body_model_utils/joint_regressors/SMPLX_to_J14.pkl'
J14_REGRESSOR = torch.from_numpy(pickle.load(open(J14_REGRESSOR_PATH, 'rb'), encoding='latin1')).to('cuda').float()
# Indices to get the 14 LSP joints from the ground truth SMPL joints
jreg_path = f'{ESSENTIALS_HOME}/body_model_utils/joint_regressors/J_regressor_h36m.npy'
SMPL_TO_H36M = torch.from_numpy(np.load(jreg_path)).to('cuda').float()
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

PSGT_FOLDER = '/is/cluster/work/lmueller2/results/HHC/optimization/fit_pseudogt_flickrci3ds_val'
BASELINE_FOLDER = '/is/cluster/work/lmueller2/results/HHC/optimization/fit_baseline_flickrci3ds_val'
DATASET_FOLDER_ORIG = f'{PROJECT_HOME}/datasets/original/FlickrCI3D_Signatures'
DATASET_FOLDER_PROCESSED = f'{PROJECT_HOME}/datasets/processed/FlickrCI3D_Signatures'

#DATA_PROCESSED_TEST = '/is/cluster/lmueller2/projects/HumanHumanContact/humanhumancontact/datasets/processed/FlickrCI3D_Signatures_Transformer/test.pkl'
#DATA_PROCESSED_VAL = '/is/cluster/lmueller2/projects/HumanHumanContact/humanhumancontact/datasets/processed/FlickrCI3D_Signatures_Transformer/val.pkl'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg',  type=str, 
        dest='exp_cfgs', nargs='+', default='llib/methods/hhcs_optimization/evaluation/eval_flickrci3ds.yaml', 
        help='The configuration of the experiment')
        
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
        nargs='*', help='The configuration of the Detector') 

    parser.add_argument('--flickrci3ds-folder-orig', type=str, default=DATASET_FOLDER_ORIG,
        help='Folder where the flickrci3ds dataset is stored.')

    parser.add_argument('--flickrci3ds-folder-processed', type=str, default=DATASET_FOLDER_PROCESSED,
        help='Folder where the flickrci3ds dataset is stored.')

    parser.add_argument('--flickrci3ds-split', type=str, default='val',
        help='The dataset split to evaluate on')

    parser.add_argument('-gt', '--pseudo-ground-truth', type=str, default=PSGT_FOLDER,
        help='Folder where the pseudo ground-truth meshes are stored.')

    parser.add_argument('-p','--predicted', type=str, default=BASELINE_FOLDER,
        help='Folder where the predicted meshes / method to evaluate against are \
            stored. To evaluate BEV, pass the processed pkl file.')

    parser.add_argument('--predicted-is-bev', default=False, action='store_true',
        help='Set flag if predicted meshes are bev estimates (because they are SMPL).')

    parser.add_argument('--predicted-output-folder', type=str, default='temp',
        help='Folder where the optimized meshes are stored.')
    
    parser.add_argument('--use-joint-subset', type=str, default=False,
        help='Use a subset of the Flickr data for which all methods have estimates.')

    cmd_args = parser.parse_args()

    cfg = merge_configs(cmd_args, default_config)

    return cfg, cmd_args



def get_smplx_params(human, body_model_smplx=None):
    """
    Returns the SMPL parameters of a human.
    """
    def to_tensor(x):
        return torch.tensor(x).to('cuda')

    params = dict(
        betas = to_tensor(human[f'betas']),
        global_orient = to_tensor(human[f'global_orient']),
        body_pose = to_tensor(human[f'body_pose']),
        scale = to_tensor(human[f'scale']),
        transl = to_tensor(human['transl'])
    )

    verts, joints = None, None
    if body_model_smplx is not None:
        body = body_model_smplx(**params)
        verts = body.vertices.detach() 
        joints = torch.matmul(J14_REGRESSOR, verts)

    return params, verts, joints


def get_bev_smpl_params(item):
    """Read SMPL parameters from a pkl file."""

    bev_smpl_vertices = torch.from_numpy(item[f'bev_smpl_vertices']).unsqueeze(0).to('cuda')
    bev_smpl_transl = torch.from_numpy(item['bev_cam_trans']).unsqueeze(0).to('cuda')
    bev_smpl_vertices = bev_smpl_vertices + bev_smpl_transl[:,:,np.newaxis,:]
    bev_joints = torch.matmul(SMPL_TO_H36M, bev_smpl_vertices[0])[:,H36M_TO_J14,:]

    return bev_smpl_vertices, bev_joints


def main(cfg, cmd_args):
    """
    Evaluates the performance of a method on the FlickrCI3D dataset.
    Compares the method to pseudo-ground-truth.
    """

    # build metrics 
    contact_metric = build_metric(cfg.evaluation.cmap_iou)
    scale_mpjpe_metric = build_metric(cfg.evaluation.scale_mpjpe)
    mpjpe_metric = build_metric(cfg.evaluation.mpjpe)
    pa_mpjpe_metric = build_metric(cfg.evaluation.pa_mpjpe)
    pairwise_pa_mpjpe_metric = build_metric(cfg.evaluation.pairwise_pa_mpjpe)

    # load ground-truth data from FlickrCI3D folder
    if cmd_args.flickrci3ds_split == 'val':
        dataset_folder_processed = f'{cmd_args.flickrci3ds_folder_processed}/train'
        dataset_folder_orig = f'{cmd_args.flickrci3ds_folder_orig}/train'
        train_val_split = np.load(f'{dataset_folder_processed}/train_val_split.npz')
        img_names = train_val_split['val']
    elif cmd_args.flickrci3ds_split == 'test':
        dataset_folder_processed = f'{cmd_args.flickrci3ds_folder_processed}/test'
        dataset_folder_orig = f'{cmd_args.flickrci3ds_folder_orig}/test'
        img_names = os.listdir(f'{dataset_folder_orig}/images')
        #train_val_split = np.load(f'{dataset_folder_processed}/train_val_split.npz')
        img_names = [x.split('.')[0] for x in img_names]

    est_data = pickle.load(open(cmd_args.predicted, 'rb'))
    bev_data = {}
    for item_idx, item in enumerate(est_data):
        imgname = item['imgname'].split('.')[0]
        if imgname in bev_data.keys():
            bev_data[imgname][item['contact_index']] = item_idx
        else:
            bev_data[imgname] = {item['contact_index']: item_idx}

    # contact map annotations
    annotations_path = f'{dataset_folder_orig}/interaction_contact_signature.json'
    annotations = json.load(open(annotations_path, 'r'))

    # load bad fits 
    bad_fits = os.listdir(cmd_args.pseudo_ground_truth + '/bad_fits/')

    # load body model to get joints from SMPL parameters
    body_model_smplx = build_bodymodel(
        cfg=cfg.body_model, 
        batch_size=2, #cfg.batch_size, 
        device=cfg.device
    )

    with open(os.path.join(os.environ['ESSENTIALS_HOME'], 'smpl_to_smplx', 'smpl_to_smplx.pkl'), 'rb') as f:
        SMPL_TO_SMPLX = pickle.load(f)['matrix']
    SMPL_TO_SMPLX = torch.from_numpy(SMPL_TO_SMPLX).unsqueeze(0).to(cfg.device).repeat(2, 1, 1).float() # [2, SMPLX vertices, SMPL vertices]

    cmapper = ContactMap(
        region_to_vertex=REGION_TO_VERTEX_PATH,
    ).to(cfg.device)

    results = ResultLogger(
        method_names=['est']
    )

    contact_zeros = torch.zeros((75, 75)) \
        .to(torch.bool).unsqueeze(0).to('cuda')

    #for item_idx, item in tqdm(enumerate(pgt_data)):
    total_size, pgt_not_found, est_not_found = 0, 0, 0
    for item_idx, item in tqdm(enumerate(img_names)):
        annos = annotations[item]

        for anno_idx, anno in enumerate(annos['ci_sign']):
            total_size += 1
            p1_idx, p2_idx = anno['person_ids']
            region_id = anno['smplx']['region_id']

            gt_cmap = contact_zeros.clone()
            for rid in region_id:
                gt_cmap[0, rid[0], rid[1]] = True

            # load pseudo ground-truth
            pgt_filename = osp.join(cmd_args.pseudo_ground_truth, 'pkl', f'{item}_{anno_idx}.pkl')
            if not osp.exists(pgt_filename):
                pgt_filename = osp.join(cmd_args.pseudo_ground_truth, 'results', f'{item}_{anno_idx}.pkl')
            if not osp.exists(pgt_filename):
                pgt_not_found += 1
                # print(f'Pseudo ground-truth not found for {item}_{anno_idx}')
                continue
            if f'{item}_{anno_idx}.png' in bad_fits:
                # print(f'Bad fit for {item}_{anno_idx}. Skipping')
                continue

            pgt_item = pickle.load(open(pgt_filename, 'rb'))
            _, verts, joints = get_smplx_params(pgt_item[f'humans'], body_model_smplx) 
            pgt_smplx_vertices = [verts[[0]], verts[[1]]]
            pgt_smplx_joints = [joints[[0]], joints[[1]]]

            # load estimated
            est_item_idx = bev_data[item][anno_idx]
            est_item = est_data[est_item_idx]  
            bev_verts, bev_joints = get_bev_smpl_params(est_item)
            bev_smplx_vertices = torch.bmm(SMPL_TO_SMPLX, bev_verts.squeeze(0))

            # PGT to GT Contact Map error (as in 3D human interactions)
            pgt_cmap_heat = cmapper.get_full_heatmap(bev_smplx_vertices[[0]], bev_smplx_vertices[[len(bev_verts)-1]])
            print(pgt_cmap_heat.shape)
            pgt_cmap_binary = pgt_cmap_heat < 0.013
            iou, precision, recall, fsore = contact_metric(pgt_cmap_binary, gt_cmap)
            results.output['est_iou'].append(iou)
            results.output['est_precision'].append(precision)
            results.output['est_recall'].append(recall)
            results.output['est_fscore'].append(fsore)

            # PCC
            max_points = gt_cmap.sum()
            distances = pgt_cmap_heat[gt_cmap]
            batch_dist = distances[None].repeat(len(results.pcc_x), 1) \
                < results.pcc_x.unsqueeze(1).repeat(1, max_points)
            pcc = batch_dist.sum(1) / max_points
            results.output['est_pcc'].append(pcc)

            def to_j14(x, J14_REGRESSOR=J14_REGRESSOR):
                return torch.matmul(J14_REGRESSOR, x)
            
            # compute metrics
            est_smplx_joints = [bev_joints[[0]], bev_joints[[1]]]                
            pgt_smplx_joints = [to_j14(pgt_smplx_vertices[0]), to_j14(pgt_smplx_vertices[1])]

            results.output[f'est_mpjpe_h0'].append(
                mpjpe_metric(est_smplx_joints[0], pgt_smplx_joints[0]).mean())
            results.output[f'est_mpjpe_h1'].append(
                mpjpe_metric(est_smplx_joints[1], pgt_smplx_joints[1]).mean())
            results.output[f'est_scale_mpjpe_h0'].append(
                scale_mpjpe_metric(est_smplx_joints[0].cpu().numpy(), pgt_smplx_joints[0].cpu().numpy()).mean())
            results.output[f'est_scale_mpjpe_h1'].append(
                scale_mpjpe_metric(est_smplx_joints[1].cpu().numpy(), pgt_smplx_joints[1].cpu().numpy()).mean())
            results.output[f'est_pa_mpjpe_h0'].append(
                pa_mpjpe_metric(est_smplx_joints[0].cpu().numpy(), pgt_smplx_joints[0].cpu().numpy()).mean())
            results.output[f'est_pa_mpjpe_h1'].append(
                pa_mpjpe_metric(est_smplx_joints[1].cpu().numpy(), pgt_smplx_joints[1].cpu().numpy()).mean())
            results.output[f'est_pa_mpjpe_h0h1'].append(pairwise_pa_mpjpe_metric(
                torch.cat(est_smplx_joints, dim=1).cpu().numpy(), torch.cat(pgt_smplx_joints, dim=1).cpu().numpy()).mean())
    
    results.topkl()

    output_folder = cmd_args.predicted_output_folder
    evaluation_folder = osp.join(output_folder, 'evaluation')
    os.makedirs(evaluation_folder, exist_ok=True)
    result_fn = os.path.join(evaluation_folder, f'metrics.txt')
    output_file = open(result_fn, 'a')
    # write date and time in readable format to output file
    _ = output_file.write(str(datetime.datetime.now()))

    for k, v in results.output.items():
        if len(v) == 0:
            continue

        if isinstance(v[0], torch.Tensor):
            v = torch.stack(v).cpu().numpy()
        else:
            v = np.array(v)

        if 'mpjpe' in k: # convert to mm
            v = v * 1000
        
        if 'pcc' in k:
            pcc = v.mean(0)
            # plot with x axis from result.pcc_X as pcc as value
            plt.plot(results.pcc_x.cpu().numpy(), pcc)
            plt.xlabel('Distance threshold (m)')
            plt.ylabel('PCC')
            plt.savefig(os.path.join(evaluation_folder, f'pcc.png'))
            plt.close()
            pcc_str = ';'.join([str(x) for x in pcc])
            print(pcc_str)
            x_str = ';'.join([str(x) for x in results.pcc_x.cpu().numpy()])
            _ = output_file.write(f'{k}_x: {x_str}')
            _ = output_file.write(f'{k}: {pcc_str}')


        error = v.mean()
        print(f'{k}: {error}')
        # write to file in predicted folder
        _ = output_file.write(f'{k}: {error} \n')
    
    _ = output_file.write(f'Pseudo ground-truth not found: {pgt_not_found}/{total_size}')
    _ = output_file.write(f'Predicted not exsits: {est_not_found}/{total_size}')
    output_file.close()



if __name__ == "__main__":

    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
