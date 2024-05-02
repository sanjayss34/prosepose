import os.path as osp 
import ipdb
import json 
import torch
import numpy as np
import os
import cv2
import smplx
import math
import pickle
import trimesh
import sys
from tqdm import tqdm
from llib.utils.image.bbox import iou_matrix
from llib.utils.keypoints.matching import keypoint_cost_matrix
from llib.defaults.body_model.main import conf as body_model_conf
from llib.cameras.perspective import PerspectiveCamera
from llib.bodymodels.utils import smpl_to_openpose
from loguru import logger as guru
from llib.data.preprocess.utils.shape_converter import ShapeConverter
from llib.utils.image.bbox import iou
        

import torch
import torch.nn as nn

KEYPOINT_COST_TRHESHOLD = 0.008

def check_bev_estimate(human_id, human0_id, bev_human_idx):
    """ 
    Check if the best match in BEV is two different people
    for human 0 and human 1.
    """
    ignore = False
    if human_id == 0:
        # first detected person, save detected bev index
        human0_id = bev_human_idx
    else:
        # second detected person, check if detected bev index is the same
        # as for first detected person. If so, ignore image.
        if bev_human_idx == human0_id:
            ignore = True
    return human0_id, ignore

def compare_and_select_openpose_vitpose(
    vitpose_data, op_human_kpts, opvitpose_kpcost_matrix,
    op_human_idx, vitpose_human_idx, KEYPOINT_COST_TRHESHOLD
):
    if vitpose_human_idx == -1:
        vitpose_human_kpts = op_human_kpts
    else:                    
        detection_cost = opvitpose_kpcost_matrix[op_human_idx][vitpose_human_idx]
        if detection_cost <= KEYPOINT_COST_TRHESHOLD:
            vitpose_human_kpts = vitpose_data[vitpose_human_idx]
        else:
            vitpose_human_kpts = op_human_kpts
    return vitpose_human_kpts

class Demo():
    
    BEV_FOV = 60

    def __init__(
        self,
        original_data_folder,
        image_folder='images',
        bev_folder='bev',
        openpose_folder='openpose',
        vitpose_folder='vitpose',
        number_of_regions=75, 
        imar_vision_datasets_tools_folder=None,
        has_gt_contact_annotation=False,
        image_format='png',
        image_name_select='',
        humans_per_example=2,
        largest_bbox_only=False,
        center_bbox_only=False,
        best_match_with_bev_box=False,
        unique_keypoint_match=True,
        contact_map_path=None,
        custom_losses_path=None,
        hmr2=False,
        write_processed_path=None,
        **kwargs,
    ):  

        self.original_data_folder = original_data_folder
        self.data_folder = original_data_folder
        self.image_format = image_format
        self.image_name_select = image_name_select
        self.image_folder = osp.join(self.data_folder, image_folder)
        self.openpose_folder = osp.join(self.data_folder, openpose_folder)
        self.bev_folder = osp.join(self.data_folder, bev_folder)
        self.vitpose_folder = osp.join(self.data_folder, vitpose_folder)


        # convert smpl betas to smpl-x betas 
        self.shape_converter_smpla = ShapeConverter(inbm_type='smpla', outbm_type='smplxa')
        self.shape_converter_smil = ShapeConverter(inbm_type='smil', outbm_type='smplxa')

        self.humans_per_example = humans_per_example
        self.largest_bbox_only = largest_bbox_only
        self.center_bbox_only = center_bbox_only
        self.best_match_with_bev_box = best_match_with_bev_box
        self.write_processed_path = write_processed_path

        # create body model to get bev root translation from pose params
        self.body_model = self.shape_converter_smpla.outbm
        self.body_model_type = 'smplx' # read smplx and convert to smplxa

        # load contact annotations is available
        self.has_gt_contact_annotation = has_gt_contact_annotation
        # if self.has_gt_contact_annotation:
        #     self.imar_vision_datasets_tools_folder =  imar_vision_datasets_tools_folder
        #     annotation_fn = osp.join(
        #         self.data_folder, 'interaction_contact_signature.json'
        #     )
        #     if os.path.exists(annotation_fn):
        #         self.annotation = json.load(open(annotation_fn, 'r'))


        #     contact_regions_fn = osp.join(
        #         self.imar_vision_datasets_tools_folder, 'info/contact_regions.json'
        #     )
        #     contact_regions = json.load(open(contact_regions_fn, 'r'))
        #     self.rid_to_smplx_fids = contact_regions['rid_to_smplx_fids']

        self.number_of_regions = number_of_regions
        self.contact_zeros = torch.zeros(
            (self.number_of_regions, self.number_of_regions)
        ).to(torch.bool)

        # Get SMPL-X pose, if available
        self.global_orient = torch.zeros(3, dtype=torch.float32)
        self.body_pose = torch.zeros(63, dtype=torch.float32)
        self.betas = torch.zeros(10, dtype=torch.float32)
        self.transl = torch.zeros(3, dtype=torch.float32)

        # keypoints 
        self.keypoints = torch.zeros((24, 3), dtype=torch.float32)
        self.unique_keypoint_match = unique_keypoint_match

        # smpl joints to openpose joints for BEV / Openpose matching
        self.smpl_to_op_map = smpl_to_openpose(
            model_type='smpl', use_hands=False, use_face=False,
            use_face_contour=False, openpose_format='coco25')
        
        self.hmr2 = hmr2
        if self.hmr2:
            self.smpl_to_op_map = list(range(25))

        self.number_of_regions = 75
        self.contact_zeros = torch.zeros(
            (self.number_of_regions, self.number_of_regions)
        ).to(torch.bool)
        self.contact_map_dict = {}
        if contact_map_path is not None:
            if len(contact_map_path) > 0:
                if os.path.exists(contact_map_path):
                    with open(contact_map_path) as f:
                        self.contact_map_dict = json.load(f)
        self.custom_loss_dict = {}
        self.custom_loss_keys = []
        self.custom_loss_img_keys = []
        if custom_losses_path is not None:
            if len(custom_losses_path) > 0:
                if os.path.exists(custom_losses_path):
                    with open(custom_losses_path) as f:
                        self.custom_loss_dict = json.load(f)
                    for key in self.custom_loss_dict:
                        nonzero_return_val = False
                        for datum in self.custom_loss_dict[key]:
                            return_val = None
                            try:
                                locals_dict = locals()
                                exec(datum['code'], locals_dict)
                                num_arguments = int(len(datum['code'].split('def loss(')[1].split(')')[0].split(',')))
                                loss_args = [None for _ in range(num_arguments)]
                                return_val = locals_dict['loss'](*loss_args)
                            except:
                                print('error')
                                nonzero_return_val = True
                        if nonzero_return_val:
                            self.custom_loss_keys.append(key)
                    if all([len(key.split('/')) == 2 for key in self.custom_loss_keys]):
                        # MTP
                        self.custom_loss_dict = {key.split('/')[-1]: self.custom_loss_dict[key] for key in self.custom_loss_dict}
                        self.custom_loss_img_keys = [key.split('/')[-1] for key in self.custom_loss_keys]
                    if all(['YOGI_Cam_' in key for key in self.custom_loss_keys]):
                        # MOYO
                        self.custom_loss_dict = {key: self.custom_loss_dict[key] for key in self.custom_loss_dict}
                        self.custom_loss_img_keys = [key for key in self.custom_loss_keys]
                    else:
                        # Flickr
                        self.custom_loss_img_keys = ['_'.join(key.split('_')[:-1]) for key in self.custom_loss_keys]

    def bbox_from_openpose(self, op_data, kp_key='pose_keypoints_2d'):
        bbox = []
        for x in op_data:
            keypoints = x[kp_key]
            kpts = np.array(keypoints).reshape(-1,3)
            conf = kpts[:,-1]
            x0, y0, _ = kpts[conf > 0].min(0) # lower left corner
            x1, y1, _ = kpts[conf > 0].max(0) # upper right corner
            bbox.append([x0,y0,x1,y1])
        bbox = np.array(bbox)
        return bbox

    def bbox_from_bev(self, keypoints):
        llc = keypoints.min(1) # lower left corner
        urc = keypoints.max(1) # upper right corner
        bbox = np.hstack((llc, urc))
        return bbox

    def process_bev(self, bev_human_idx, bev_data, image_size):

        smpl_betas_scale = bev_data['smpl_betas'][bev_human_idx]
        smpl_betas = smpl_betas_scale[:10]
        smpl_scale = smpl_betas_scale[-1]
        smpl_body_pose = bev_data['smpl_thetas'][bev_human_idx][3:]
        smpl_global_orient = bev_data['smpl_thetas'][bev_human_idx][:3]

        if smpl_scale > 0.8:
            smplx_betas_scale = self.shape_converter_smil.forward(torch.from_numpy(smpl_betas).unsqueeze(0))
            smplx_betas = smplx_betas_scale[0,:10].numpy()
            smplx_scale = smplx_betas_scale[0,10].numpy()
            #smplx_scale = smpl_scale # there is no smilxa model, so we keep the scale form bev
        else:
            smplx_betas_scale = self.shape_converter_smpla.forward(torch.from_numpy(smpl_betas_scale).unsqueeze(0))
            smplx_betas = smplx_betas_scale[0,:10].numpy()
            smplx_scale = smplx_betas_scale[0,10].numpy()

        cam_trans = bev_data['cam_trans'][bev_human_idx]
        smpl_joints = bev_data['joints'][bev_human_idx]
        smpl_vertices = bev_data['verts'][bev_human_idx]
        smpl_joints_2d = bev_data['pj2d_org'][bev_human_idx]

        data = {
            'bev_smpl_global_orient': smpl_global_orient,
            'bev_smpl_body_pose': smpl_body_pose,
            'bev_smpl_betas': smpl_betas,
            'bev_smpl_scale': smpl_scale,
            'bev_smplx_betas': smplx_betas,
            'bev_smplx_scale': smplx_scale,
            'bev_cam_trans': cam_trans,
            'bev_smpl_joints': smpl_joints,
            'bev_smpl_vertices': smpl_vertices,
            'bev_smpl_joints_2d': smpl_joints_2d,
        }
        
        height, width = image_size

        # hacky - use smpl pose parameters with smplx body model
        # not perfect, but close enough. SMPL betas are not used with smpl-x.
        if self.body_model_type == 'smplx':
            body_pose = data['bev_smpl_body_pose'][:63]
            global_orient = data['bev_smpl_global_orient']
            betas = data['bev_smplx_betas']
            scale = data['bev_smplx_scale']
        else:
            raise('not implemented: Data loader for SMPL loader in Flickr Signatures')

        # ignore infants, because SMPL-X doesn't support them (template is noisy)
        has_infant = False
        if np.any(data['bev_smpl_scale'] > 0.8):
            has_infant = True
        
        bev_cam_trans = torch.from_numpy(data['bev_cam_trans'])
        bev_camera = PerspectiveCamera(
            rotation=torch.tensor([[0., 0., 180.]]),
            translation=torch.tensor([[0., 0., 0.]]),
            afov_horizontal=torch.tensor([self.BEV_FOV]),
            image_size=torch.tensor([[width, height]]),
            batch_size=1,
            device='cpu'
        )

        bev_vertices = data['bev_smpl_vertices']
        if not self.hmr2:
            bev_root_trans = data['bev_smpl_joints'][[45,46],:].mean(0)
            bev_vertices_root_trans = bev_vertices - bev_root_trans[np.newaxis,:] \
                + bev_cam_trans.numpy()[np.newaxis,:]
            data['bev_smpl_vertices_root_trans'] = bev_vertices_root_trans
        
        smplx_update = {
            'bev_smplx_global_orient': [],
            'bev_smplx_body_pose': [],
            'bev_smplx_transl': [],
            'bev_smplx_keypoints': [],
            'bev_smplx_vertices': [],
        }

        idx = 0
        h_global_orient = torch.from_numpy(global_orient).float().unsqueeze(0)
        smplx_update['bev_smplx_global_orient'].append(h_global_orient)
        
        h_body_pose = torch.from_numpy(body_pose).float().unsqueeze(0)
        smplx_update['bev_smplx_body_pose'].append(h_body_pose)

        h_betas_scale = torch.from_numpy(
            np.concatenate((betas, scale[None]), axis=0)
        ).float().unsqueeze(0)

        print(h_body_pose.shape, h_global_orient.shape, h_betas_scale.shape)
        body = self.body_model(
            global_orient=h_global_orient,
            body_pose=h_body_pose,
            betas=h_betas_scale,
        )

        root_trans = body.joints.detach()[:,0,:]
        transl = -root_trans.to('cpu') + bev_cam_trans.to('cpu')
        smplx_update['bev_smplx_transl'].append(transl)

        body = self.body_model(
            global_orient=h_global_orient,
            body_pose=h_body_pose,
            betas=h_betas_scale,
            transl=transl,
        )

        keypoints = bev_camera.project(body.joints.detach())
        smplx_update['bev_smplx_keypoints'].append(keypoints.detach())

        vertices = body.vertices.detach().to('cpu')
        smplx_update['bev_smplx_vertices'].append(vertices)

        for k, v in smplx_update.items():
            smplx_update[k] = torch.cat(v, dim=0)

        data.update(smplx_update)

        return data, has_infant

    def read_data(self, imgname):

        # annotation / image paths
        img_path = osp.join(self.image_folder, f'{imgname}.{self.image_format}')
        bev_path = osp.join(self.bev_folder, f'{imgname}_0.08.npz')
        vitpose_path = osp.join(self.vitpose_folder, f'{imgname}_keypoints.json')
        openpose_path = osp.join(self.openpose_folder, f'{imgname}.json')

        guru.info(f'Loading {imgname} from {img_path}')
        guru.info(f'Loading BEV from {bev_path}')
        guru.info(f'Loading ViTPose from {vitpose_path}')
        guru.info(f'Loading OpenPose from {openpose_path}')

        # load each annotation file
        IMG = cv2.imread(img_path)
        bev_data = np.load(bev_path, allow_pickle=True)['results'][()]
        if not os.path.exists(vitpose_path):
            with open(openpose_path) as f:
                vitpose_data = json.load(f)['people']
        else:
            with open(vitpose_path) as f:
                vitpose_data = json.load(f)['people']
        if len(vitpose_data) == 0:
            with open(openpose_path) as f:
                vitpose_data = json.load(f)['people']
        if not os.path.exists(openpose_path):
            guru.warning(f'Openpose file does not exist; using ViTPose keypoints only.')
            op_data = vitpose_data 
        else:
            op_data = json.load(open(openpose_path, 'r'))['people']

        return img_path, IMG, bev_data, op_data, vitpose_data   

    def _get_opbev_cost(self, op_data, bev_data, IMG, unique_best_matches=True):
        print('opbev cost')
        print('op', [np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in op_data])
        print('bev', [np.concatenate((x.reshape(-1,2)[self.smpl_to_op_map,:], np.ones((25, 1))), axis=1) for x in bev_data['pj2d_org']])
        matrix, best_match = keypoint_cost_matrix(
            kpts1=[np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in op_data],
            kpts2=[np.concatenate((x.reshape(-1,2)[self.smpl_to_op_map,:], np.ones((25, 1))), axis=1) for x in bev_data['pj2d_org']],
            norm=max(IMG.shape[0], IMG.shape[1]),
            unique_best_matches=unique_best_matches
        )
        print(matrix)
        print('best_match', best_match)
        return matrix, best_match

    def _get_opvitpose_cost(self, op_data, vitpose_data, IMG, unique_best_matches=True):
        print(len(op_data), len(vitpose_data))
        matrix, best_match = keypoint_cost_matrix(
            kpts1=[np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in op_data],
            kpts2=[np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in vitpose_data],
            norm=max(IMG.shape[0], IMG.shape[1]),
            unique_best_matches=unique_best_matches
        )
        return matrix, best_match

    def _get_output_template(self, IMG, imgname, img_path):
        height, width, _ = IMG.shape
        afov_radians = (self.BEV_FOV / 2) * math.pi / 180
        focal_length_px = (max(width, height)/2) / math.tan(afov_radians)
        template = {
            'imgname': f'{imgname}.{self.image_format}',
            'imgpath': img_path,
            'img_height': height,
            'img_width': width,
            'cam_transl': [0., 0., 0.] ,
            'cam_rot': [0., 0., 180.],
            'fl': focal_length_px,
            'afov_horizontal': self.BEV_FOV,
        }
        return template

    def _load_single_human(
        self,
        op_data,
        vitpose_data,
        bev_data,
        op_human_idx,
        opbev_kpcost_best_match,
        opvitpose_kpcost_best_match,
        opvitpose_kpcost_matrix,
        img_height, 
        img_width,
    ):
        bev_human_idx = opbev_kpcost_best_match[op_human_idx]
        human_data, has_infant = self.process_bev(
            bev_human_idx, bev_data, (img_height, img_width))
        human_data['has_infant'] = has_infant
        print('opvitpose kpcost', opvitpose_kpcost_matrix.shape)
        
        # check if infant or no bev match was detected. If so, ignore image.
        if (human_data is None) or (bev_human_idx == -1):
            print(human_data is None, bev_human_idx == -1)
            return None

        # process OpenPose keypoints
        op_kpts = np.zeros((135, 3))
        if op_human_idx != -1:
            kpts = op_data[op_human_idx]
            if 'hand_left_keypoints_2d' not in kpts or len(kpts['hand_left_keypoints_2d']) == 0:
                kpts['hand_left_keypoints_2d'] = [0 for _ in range(20) for _ in range(3)]
            if 'hand_right_keypoints_2d' not in kpts or len(kpts['hand_right_keypoints_2d']) == 0:
                kpts['hand_right_keypoints_2d'] = [0 for _ in range(20) for _ in range(3)]
            if 'face_keypoints_2d' not in kpts or len(kpts['face_keypoints_2d']) == 0:
                kpts['face_keypoints_2d'] = [0 for _ in range(70) for _ in range(3)]

            # print(kpts['pose_keypoints_2d'])
            if isinstance(kpts['pose_keypoints_2d'][0], list):
                kpts['pose_keypoints_2d'] = [num for lst in kpts['pose_keypoints_2d'] for num in lst]
            if isinstance(kpts['hand_left_keypoints_2d'][0], list):
                kpts['hand_left_keypoints_2d'] = [num for lst in kpts['hand_left_keypoints_2d'] for num in lst]
            if isinstance(kpts['hand_right_keypoints_2d'][0], list):
                kpts['hand_right_keypoints_2d'] = [num for lst in kpts['hand_right_keypoints_2d'] for num in lst]
            # print(len(kpts['pose_keypoints_2d']), len(kpts['hand_left_keypoints_2d']), len(kpts['hand_right_keypoints_2d']))
            # body + hands
            body = np.array(kpts['pose_keypoints_2d'] + \
                kpts['hand_left_keypoints_2d'] + kpts['hand_right_keypoints_2d']
            ).reshape(-1,3)
            # face 
            face = np.array(kpts['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
            contour = np.array(kpts['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[:17, :]
            # final openpose
            op_kpts = np.concatenate([body, face, contour], axis=0)

        # OpenPose and vit detection cost (if cost is too high, use Openpose) 
        vitpose_human_idx = opvitpose_kpcost_best_match[op_human_idx]
        
        # OpenPose and vit detection cost (if cost is too high, use Openpose)
        vitpose_kpts = np.zeros_like(op_kpts)
        print(vitpose_human_idx)
        print('all vitpose')
        if vitpose_human_idx != -1:
            kpts = vitpose_data[vitpose_human_idx]
            if 'hand_left_keypoints_2d' not in kpts or len(kpts['hand_left_keypoints_2d']) == 0:
                # print('no left hand')
                kpts['hand_left_keypoints_2d'] = [[0, 0, 0] for _ in range(20)]
            if 'hand_right_keypoints_2d' not in kpts or len(kpts['hand_right_keypoints_2d']) == 0:
                # print('no right hand')
                kpts['hand_right_keypoints_2d'] = [[0, 0, 0] for _ in range(20)]
            if 'face_keypoints_2d' not in kpts or len(kpts['face_keypoints_2d']) == 0:
                # print('no face')
                kpts['face_keypoints_2d'] = [[0, 0, 0] for _ in range(70)]
            # if len(kpts['pose_keypoints_2d']) == 75:
            #     kpts['pose_keypoints_2d'] = np.array(kpts['pose_keypoints_2d']).reshape(-1, 3).tolist()
            for key in ['pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d', 'face_keypoints_2d']:
                if isinstance(kpts[key][0], float) or isinstance(kpts[key][0], int):
                    kpts[key] = np.array(kpts[key]).reshape(-1, 3).tolist()
            print(len(kpts['pose_keypoints_2d']), len(kpts['hand_left_keypoints_2d']), len(kpts['hand_right_keypoints_2d']))
            # body + hands
            body = np.array(kpts['pose_keypoints_2d'] + \
                kpts['hand_left_keypoints_2d'] + kpts['hand_right_keypoints_2d']
            ).reshape(-1,3)
            # face 
            face = np.array(kpts['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
            contour = np.array(kpts['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[:17, :]
            # final openpose
            # print(body.shape, face.shape, contour.shape)
            vitpose_kpts = np.concatenate([body, face, contour], axis=0)
            # print(vitpose_kpts.shape)
            assert vitpose_kpts.shape[-2] in {133, 135}

        # # add keypoints vitposeplus
        # vitposeplus_kpts = np.zeros_like(op_kpts)
        # if vitposeplus_human_idx != -1:
        #     vitposeplus_kpts_orig = vitposeplus_data[vitposeplus_human_idx]['keypoints']
        #     main_body_idxs = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
        #     vitposeplus_kpts[main_body_idxs] = vitposeplus_kpts_orig[:17] # main body keypoints
        #     vitposeplus_kpts[19:25] = vitposeplus_kpts_orig[17:23] # foot keypoints
        #     vitposeplus_kpts[25:46] = vitposeplus_kpts_orig[-42:-21] # left hand keypoints
        #     vitposeplus_kpts[46:67] = vitposeplus_kpts_orig[-21:] # right hand keypoints
        #     #vitposeplus_kpts[67:135] = vitposeplus_kpts_orig[23:-42] # face keypoints
        #     face_countour = vitposeplus_kpts_orig[23:-42] 
        #     face = np.array(face_countour)[17: 17 + 51, :]
        #     contour = np.array(face_countour)[:17, :]
        #     vitposeplus_kpts[67:135] = np.concatenate([face, contour], axis=0) 

        # add idxs, bev data and keypoints to template
        human_data['openpose_human_idx'] = op_human_idx
        human_data['bev_human_idx'] = bev_human_idx
        human_data['vitpose_human_idx'] = vitpose_human_idx
        human_data['vitposeplus_human_idx'] = vitpose_human_idx
        human_data['vitpose'] = vitpose_kpts
        human_data['openpose'] = op_kpts
        human_data['vitposeplus'] = vitpose_kpts
        print('op', op_kpts[:25,:])
        print('vit', vitpose_kpts[:25,:])
    
        for k, v in human_data.items():

            if k in [
                'bev_smplx_global_orient', 'bev_smplx_body_pose', 'bev_smplx_transl', 
                'bev_smplx_keypoints', 'bev_smplx_vertices'
            ]:
                v = v[0]

            human_data[k] = np.array(v).copy()

        return human_data

    def load_from_cmap(self, imgname, annotation):

        bev_path = osp.join(self.bev_folder, f'{imgname}_0.08.npz')
        if not os.path.exists(bev_path):
            return []
        img_path, IMG, bev_data, op_data, vitpose_data = self.read_data(imgname)
        image_data_template = self._get_output_template(IMG, imgname, img_path)

        ################ Match HHC annotation with OpenPose and ViT keypoints #################
        op_bbox = self.bbox_from_openpose(op_data, kp_key='pose_keypoints_2d') 
        ci_bbox = np.array(annotation['bbxes'])
        ciop_iou_matrix, ciop_iou_best_match = iou_matrix(ci_bbox, op_bbox)

        opbev_kpcost_matrix, opbev_kpcost_best_match = \
            self._get_opbev_cost(op_data, bev_data, IMG, False)
        opvitpose_kpcost_matrix, opvitpose_kpcost_best_match = \
            self._get_opvitpose_cost(op_data, vitpose_data, IMG, False)

        image_contact_data = []

        # load contact annotations
        sign_key = 'ci_sign'
        if 'ci_sign' not in annotation:
            sign_key = 'sc_sign'
        for case_ci_idx, case_ci in enumerate(annotation[sign_key]):
            IGNORE_PAIR = False
            image_data = image_data_template.copy()

            image_data['contact_index'] = case_ci_idx
            # contact human id annotation

            if 'person_ids' not in case_ci:
                case_ci['person_ids'] = [case_ci['person_id']]
            person1_id = case_ci['person_ids'][0]
            if len(case_ci['person_ids']) > 1:
                person2_id = case_ci['person_ids'][1]
            image_data['hhc_contacts_human_ids'] = case_ci['person_ids']
            # contact regions annotation
            region_id = case_ci[self.body_model_type]['region_id']
            image_data['hhc_contacts_region_ids'] = region_id

            contact_map = self.contact_zeros.clone()
            if self.has_gt_contact_annotation:
                for rid in region_id:
                    contact_map[rid[0], rid[1]] = True
            image_data['contact_map'] = contact_map

            ################ load the two humans in contact #################
            human0_id = None
            humans = []
            for human_id, bbox_id in enumerate(case_ci['person_ids']):
                bbox = annotation['bbxes'][bbox_id]
                image_data[f'bbox_h{human_id}'] = bbox
                op_human_idx = ciop_iou_best_match[bbox_id]

                h = self._load_single_human(
                    op_data, vitpose_data, bev_data,
                    op_human_idx,
                    opbev_kpcost_best_match,
                    opvitpose_kpcost_best_match,
                    opvitpose_kpcost_matrix,
                    IMG.shape[0], IMG.shape[1]
                )
                humans.append(h)
            concatenated_dict = {}
            for key in humans[0].keys():
                concatenated_dict[key] = np.stack([h[key] for h in humans], axis=0)
            image_data.update(concatenated_dict)

            image_contact_data.append(image_data)

        return image_contact_data

    def load_single_image(self, imgname):

        bev_path = osp.join(self.bev_folder, f'{imgname}_0.08.npz')
        if not os.path.exists(bev_path):
            print(bev_path, 'DOESNT EXIST')
            return []
        op_path = osp.join(self.openpose_folder, f'{imgname}.json')
        if not os.path.exists(op_path):
            print(op_path, 'DOESNT EXIST')
            return []
        img_path, IMG, bev_data, op_data, vitpose_data = self.read_data(imgname)        
        image_data = self._get_output_template(IMG, imgname, img_path)

        ################ Find all overlapping bounding boxes and process these people #################
        op_bbox = self.bbox_from_openpose(op_data, kp_key='pose_keypoints_2d')
        all_person_ids = []
        for bb1_idx in range(op_bbox.shape[0]):
            for bb2_idx in range(bb1_idx + 1, op_bbox.shape[0]):
                bb1, bb2 = op_bbox[bb1_idx], op_bbox[bb2_idx]
                bb12_iou = iou(bb1, bb2)
                if bb12_iou > 0:
                    all_person_ids.append([bb1_idx, bb2_idx])

        if self.humans_per_example == 1:
            all_person_ids = [[bb1_idx] for bb1_idx in range(op_bbox.shape[0])]
        if self.largest_bbox_only:
            areas = [iou(op_bbox[i], np.array([0, 0, IMG.shape[0], IMG.shape[1]])) for i in range(op_bbox.shape[0])]
            indices = sorted(list(range(op_bbox.shape[0])), key=lambda x: -areas[x])
            if len(indices) > 0:
                all_person_ids = [[indices[0]]]
        if self.center_bbox_only:
            centers = [((op_bbox[i,0]+op_bbox[i,2])/2, (op_bbox[i,1]+op_bbox[i,3])/2) for i in range(op_bbox.shape[0])]
            img_center = (IMG.shape[1]/2, IMG.shape[0]/2)
            distances_from_center = [(center[0]-img_center[0])**2+(center[1]-img_center[1])**2 for center in centers]
            if len(distances_from_center) > 0:
                indices = sorted(list(range(len(distances_from_center))), key=lambda x: distances_from_center[x])
                if self.humans_per_example == 1:
                    centerest = np.argmin(distances_from_center)
                    all_person_ids = [[centerest]]
                elif len(indices) >= 2:
                    all_person_ids = [[indices[0], indices[1]]]
                else:
                    all_person_ids = []
        if len(all_person_ids) == 0:
            print(imgname, 'NO OP BBOXES')
            return []
        print('all_person_ids', all_person_ids)

        # cost matric to solve correspondance between openpose and bev and vitpose
        opbev_kpcost_matrix, opbev_kpcost_best_match = \
            self._get_opbev_cost(op_data, bev_data, IMG, self.unique_keypoint_match)
        opvitpose_kpcost_matrix, opvitpose_kpcost_best_match = \
            self._get_opvitpose_cost(op_data, vitpose_data, IMG, self.unique_keypoint_match)
        if self.best_match_with_bev_box:
            print('MATCH WTH BEV BOX')
            ious = [iou(op_bbox[i], bev_data['boxes'][0,:]) for i in range(op_bbox.shape[0])]
            assert self.humans_per_example == 1
            all_person_ids = [[np.argmax(ious)]]
            vitpose_boxes = []
            for i in range(len(vitpose_data)):
                kpts = np.array(vitpose_data[i]['pose_keypoints_2d']).reshape(-1, 3)[:25,:]
                conf = kpts[:,-1]
                x0, y0 = kpts[conf > 0,:2].min(axis=0)
                x1, y1 = kpts[conf > 0,:2].max(axis=0)
                vitpose_boxes.append([x0, y0, x1, y1])
            ious = [iou(box, bev_data['boxes'][0,:]) for box in vitpose_boxes]
            opvitpose_kpcost_matrix = np.array([-value for value in ious])
            opvitpose_kpcost_best_match = np.array([np.argmax(ious) for _ in range(op_bbox.shape[0])])
            bev_boxes = []
            for x in bev_data['pj2d_org']:
                kpts = np.concatenate((x.reshape(-1,2)[self.smpl_to_op_map,:], np.ones((25, 1))), axis=1)
                conf = kpts[:,-1]
                x0, y0 = kpts[conf > 0,:2].min(axis=0)
                x1, y1 = kpts[conf > 0,:2].max(axis=0)
                bev_boxes.append([x0, y0, x1, y1])
            ious_bev = [iou(box, bev_data['boxes'][0,:]) for box in bev_boxes]
            opbev_kpcost_matrix = np.array([-value for value in ious_bev])
            opbev_kpcost_best_match = np.array([np.argmax(ious_bev) for _ in range(op_bbox.shape[0])])
            # print(np.array(op_data[all_person_ids[0][0]]['pose_keypoints_2d']).reshape(-1, 3)[:25,:])

        ################ load the two humans in contact for each pair #################
        all_image_data = []
        all_image_names = []
        for pidx, person_ids in enumerate(all_person_ids):

            image_data['contact_index'] = pidx
            img_out_fn = f'{imgname}_{pidx}'
            image_data['img_out_fn'] = img_out_fn
            bb1 = op_bbox[person_ids[0]]
            image_data['bbox_join'] = bb1
            if len(person_ids) > 1:
                image_data['bbox_join'] = np.array([
                    [min(bb1[0], bb2[0]), min(bb1[1], bb2[1]), max(bb1[2], bb2[2]), max(bb1[3], bb2[3])]
                ])

            # ipdb.set_trace()
            h0 = self._load_single_human(
                    op_data, vitpose_data, bev_data,
                    person_ids[0],
                    opbev_kpcost_best_match,
                    opvitpose_kpcost_best_match,
                    opvitpose_kpcost_matrix,
                    IMG.shape[0], IMG.shape[1],
            )
            if h0 is None:
                guru.warning(f'No BEV match found for {imgname} - ignoring image.')
                assert False
                continue

            if len(person_ids) > 1:
                h1 = self._load_single_human(
                        op_data, vitpose_data, bev_data,
                        person_ids[1],
                        opbev_kpcost_best_match,
                        opvitpose_kpcost_best_match,
                        opvitpose_kpcost_matrix,
                        IMG.shape[0], IMG.shape[1],
                )

                if h1 is None:
                    guru.warning(f'No BEV match found for {imgname} - ignoring image.')
                    continue

            concatenated_dict = {}
            for key in h0.keys():
                if len(person_ids) > 1:
                    concatenated_dict[key] = np.stack((h0[key], h1[key]), axis=0)
                else:
                    concatenated_dict[key] = np.stack([h0[key]], axis=0)
            
            image_data.update(concatenated_dict)

            if len(person_ids) == 1:
                print('ADDING SINGLE PERSON DATUM')
            all_image_data.append(image_data.copy())
            all_image_names.append(imgname+'_'+str(pidx))

        return all_image_data


    def load(self):

        guru.info(f'Processing data from {self.data_folder}')

        data = []
        img_names = []
        print(self.image_folder)
        for imgname in os.listdir(self.image_folder):

            # ignore images that were not selected
            if self.image_name_select != '':
                if self.image_name_select not in imgname:
                    continue   
            imgkey = '.'.join(imgname.split('.')[:-1])
            if len(self.contact_map_dict) > 0:
                if imgkey not in self.contact_map_dict:
                    continue
            if len(self.custom_loss_dict) > 0:
                print(imgkey, list(self.custom_loss_img_keys)[:10])
                if imgkey not in self.custom_loss_img_keys:
                    continue
            # if imgkey != '03234_3dcpscan_1328_0137':
            #     continue

            # get image filetype 
            self.image_format = imgname.split('.')[-1]
           
            print(imgkey)
            if imgkey in self.contact_map_dict:
                data_curr = self.load_from_cmap(imgkey, self.contact_map_dict[imgkey])
            else:
                data_curr = self.load_single_image('.'.join(imgname.split('.')[:-1]))
            if len(data_curr) > 0:
                data += data_curr

        if self.write_processed_path is not None and len(self.write_processed_path) > 0:
            with open(self.write_processed_path, 'wb') as fout:
                pickle.dump(data, fout)
        return data # , img_names
