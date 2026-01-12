"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for 
3D human body mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import sys
import types
import os.path as op
import code
import json
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import numpy as np
import cv2
from src.modeling.bert import BertConfig
from src.modeling.bert.model import DICE_Module
from src.modeling.bert.model import DICE_Network
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
# from src.datasets.build import make_decaf_data_loader, make_decaf_and_inthewild_data_loader
from src.datasets.build import make_folder_data_loader

from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter, EvalMetricsLogger
# from src.utils.renderer import visualize_reconstruction, visualize_reconstruction_test
from src.utils.metric_pampjpe import reconstruction_error
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# Decaf
from src.decaf.FLAME_util import FLAME, get_FLAME_faces, flame_forwarding
from src.decaf.util import denormalize_keys, keypoint_overlay
from src.decaf.tracking_util import mano_forwarding
import mano
import mano.model
import copy
import random
import wandb
from collections import OrderedDict
from pytorch3d.renderer import MeshRasterizer, FoVPerspectiveCameras, RasterizationSettings, PerspectiveCameras, OrthographicCameras, PointLights, SoftPhongShader, TexturesVertex
from pytorch3d.structures import Meshes
from matplotlib import pyplot as plt
from src.utils.visualizer import visualize_keypoints_single
import matplotlib.cm as cm

from torchvision import transforms

from src.utils.one_euro_filter import OneEuroFilter


def save_checkpoint(model, args, epoch, iteration, num_trial=10, face_d_model=None, hand_d_model=None):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            if face_d_model is not None:
                torch.save(face_d_model, op.join(checkpoint_dir, 'face_d_model.bin'))
                torch.save(face_d_model.state_dict(), op.join(checkpoint_dir, 'face_d_state_dict.bin'))
            if hand_d_model is not None:
                torch.save(hand_d_model, op.join(checkpoint_dir, 'hand_d_model.bin'))
                torch.save(hand_d_model.state_dict(), op.join(checkpoint_dir, 'hand_d_state_dict.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def save_scores(args, split, mpjpe, pampjpe, mpve):
    eval_log = []
    res = {}
    res['mPJPE'] = mpjpe
    res['PAmPJPE'] = pampjpe
    res['mPVE'] = mpve
    eval_log.append(res)
    with open(op.join(args.output_dir, split+'_eval_logs.json'), 'w') as f:
        json.dump(eval_log, f)
    logger.info("Save eval scores to {}".format(args.output_dir))
    return

def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs/2.0)  ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def rectify_pose(pose):
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose
    # coord_out = coord_out[has_smpl == 1]
    # coord_gt = coord_gt[has_smpl == 1]
    if len(coord_gt) > 0:
        face = torch.LongTensor(face).cuda()

        d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True) + 1e-6)
        d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)
        d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)

        d1_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,1],:])**2,2,keepdim=True) + 1e-6)
        d2_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)
        d3_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,1],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)

        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)

        loss = torch.cat((diff1, diff2, diff3), 1)
        loss = loss.mean()
        return loss
    # else:
    #     return torch.FloatTensor(1).fill_(0.).to(device)

def save_mesh(path, V, F):
    with open(path, "w") as f:
        for i in range(V.shape[0]):
            f.write("v %f %f %f\n" % (V[i, 0], V[i, 1], V[i, 2]))
        for j in range(F.shape[0]):
            f.write("f %d %d %d\n" % (1+F[j, 0], 1+F[j, 1], 1+F[j, 2]))

def visualize_keypoints(img, keypoints, color=(255,0,0), w=224, h=224):
    keypoints_vis = copy.deepcopy(keypoints)
    keypoints_vis = denormalize_keys(keys=keypoints_vis,w=w,h=h)
    keypoints_vis = keypoints_vis.squeeze().cpu().detach().numpy()
    img = img.squeeze().cpu().permute(1,2,0).detach().numpy()
    img = 255 * (img - img.min()) / (img.max() - img.min())
    img = np.ascontiguousarray(img, dtype=np.uint8)
    img = keypoint_overlay(
        keypoints_vis.astype(int),
        c=color,
        img=img)
    return img

def get_rotation_matrix(s, t):
    R = torch.zeros((s.shape[0], 3, 3)).to(s.device)
    R[:, 0, 0] = s
    R[:, 1, 1] = s
    R[:, 2, 2] = 1
    R[:, 0, 2] = t[:, 0]
    R[:, 1, 2] = t[:, 1]
    return R

def merge_faces(hand_faces, head_faces):
    hand_faces_clone = copy.deepcopy(hand_faces)
    head_faces_clone = copy.deepcopy(head_faces)
    head_faces_clone = head_faces_clone + hand_faces_clone.max() + 1

    return torch.cat([hand_faces_clone, head_faces_clone], dim=0)

def merge_verts(hand_verts, face_verts):
    return torch.cat([hand_verts, face_verts], dim=1)

def depth_to_heatmap(depth_map):
    eps = 1e-6
    # Normalize depth map values between 0 and 1
    normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + eps)
    
    # Apply colormap (jet in this case, you can change it)
    heatmap = cm.jet(normalized_depth_map)
    
    return heatmap[:, :, :3] # excluding alpha channel

def depth_to_heatmap_batch(depth_map):
    batch_size = depth_map.shape[0]
    heatmaps = []
    for i in range(batch_size):
        heatmap = depth_to_heatmap(depth_map[i])
        heatmaps.append(heatmap)
    return heatmaps

def process_zbuf(zbuf):
    # set background to be 2m
    zbuf[zbuf == -1] = 2.0
    return zbuf


def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

from plyfile import PlyData, PlyElement


def save_mesh_to_ply(vertices, faces, texture, filename='mesh.ply'):
    """
    Save a mesh to a PLY file.

    Parameters:
    vertices (torch.Tensor): Tensor of shape (N, 3) containing vertex coordinates.
    faces (torch.Tensor): Tensor of shape (F, 3) containing face indices.
    texture (torch.Tensor): Tensor of shape (N, 3) containing vertex texture colors (normalized in range 0 to 1).
    filename (str): The filename for the PLY file.
    """

    # Convert tensors to numpy arrays
    if isinstance(vertices, torch.Tensor):
        vertices_np = vertices.detach().cpu().numpy()
    else:
        vertices_np = vertices
    if isinstance(faces, torch.Tensor):
        faces_np = faces.detach().cpu().numpy()
    else:
        faces_np = faces
    if isinstance(texture, torch.Tensor):
        texture_np = texture.detach().cpu().numpy()
    else:
        texture_np = texture

    # Scale texture from (0, 1) to (0, 255)
    texture_np = (texture_np * 255).astype(np.uint8)

    # Combine vertices and texture into a single array
    vertices_with_texture = np.hstack([vertices_np, texture_np])

    # Define the vertex and face elements
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    face_dtype = [('vertex_indices', 'i4', (3,))]

    vertex_data = np.array([tuple(v) for v in vertices_with_texture], dtype=vertex_dtype)
    face_data = np.array([([f[0], f[1], f[2]],) for f in faces_np], dtype=face_dtype)

    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    face_element = PlyElement.describe(face_data, 'face')

    # Write to a PLY file
    ply_data = PlyData([vertex_element, face_element])
    ply_data.write(filename)

    def __init__(self):
        super(Face_Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(100+50+3, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, params):
        return self.model(params)


from PIL import Image, ImageDraw, ImageFont
import numpy as np

def overlay_number(image_array, text):
    # Create a blank image
    image = Image.fromarray(image_array)

    # Font settings
    font_size = 20
    font = ImageFont.load_default()
    font_color = (255, 255, 255)  # white color

    # Draw text on the image
    draw = ImageDraw.Draw(image)
    # text_width, text_height = draw.textsize(str(number), font=font)
    x_position = 0
    y_position = 0
    draw.text((x_position, y_position), text, font=font, fill=(255, 255, 255))
    draw.text((x_position, y_position+20), text, font=font, fill=(0, 0, 0))

    # Convert back to numpy array
    overlayed_image_array = np.array(image)

    return overlayed_image_array


def run(args, train_dataloader, val_dataloader, DICE_model, mano_model=None, hand_model=None, head_model=None, hand_sampler=None, head_sampler=None, hand_renderer=None, face_renderer=None, hand_F=None, head_F=None, hand_discriminator=None, face_discriminator=None):
    hand_model.eval()
    head_model.eval()
    mano_model.eval()
    DICE_model.eval()

    hand_model.to(args.device)
    head_model.to(args.device)
    

    max_iter = len(train_dataloader)
    print("max_iter: ", max_iter)
    iters_per_epoch = max_iter // args.num_train_epochs
    


    optimizer = torch.optim.AdamW(params=list(DICE_model.parameters()),
                                           lr=args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=1e-4)

    # define loss function (criterion) and optimizer
    if args.distributed:  
        DICE_model = torch.nn.parallel.DistributedDataParallel(
            DICE_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

        logger.info(
                ' '.join(
                ['Local rank: {o}', 'Max iteration: {a}', 'iters_per_epoch: {b}','num_train_epochs: {c}',]
                ).format(o=args.local_rank, a=max_iter, b=iters_per_epoch, c=args.num_train_epochs)
            )

    start_training_time = time.time()
    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    inference_time = AverageMeter()

    # save mesh
    asset_path = args.data_path+"/assets/"
    flame_model_path = asset_path+"/generic_model.pkl"
    flame_faces = get_FLAME_faces(flame_model_path)

    stiffness = np.load("src/modeling/data/stiffness_final.npy")
    stiffness = torch.from_numpy(stiffness).float().cuda(args.device)

    rh_ref_vs_ = torch.from_numpy(torch.load("src/modeling/data/rh_ref_vs.pt")).to(args.device) # 778, 3
    head_ref_vs_ = torch.from_numpy(torch.load("src/modeling/data/head_ref_vs.pt")).to(args.device) # 5023, 3
    neck_idx = np.load(args.data_path+"/assets/neck_idx.npy")

    one_euro_filter = OneEuroFilter(0.0, 0.0, d_cutoff=30.0)

    for iteration, data in tqdm(enumerate(train_dataloader)):

        iteration += 1
        epoch = iteration // iters_per_epoch
        batch_size = data["single_img_seqs"].shape[0]
        adjust_learning_rate(optimizer, epoch, args)
        data_time.update(time.time() - end)

        stiffness_batch = stiffness.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        deform_weight_mask = torch.reciprocal((1 - stiffness_batch + 0.1) ** 1.5)
        deform_weight_mask = deform_weight_mask / deform_weight_mask.min()

        single_image = data["single_img_seqs"].cuda(args.device)
        imgs_3x = data["img_3x"].cuda(args.device)

        rh_ref_vs = rh_ref_vs_.expand(batch_size, -1, -1)
        head_ref_vs = head_ref_vs_.expand(batch_size, -1, -1)

        head_ref_vs_sub2 = head_sampler.downsample(head_ref_vs.double()).float()
        rh_ref_vs_sub = hand_sampler.downsample(rh_ref_vs.double()).float()
        head_ref_kps = head_model.convert_vs2landmarks(head_ref_vs)
        rh_ref_kps = mano_model.get_3d_joints(rh_ref_vs)

        ###################################
        # Masking percantage
        # We observe that 30% works better for human body mesh. Further details are reported in the paper.
        mvm_percent = 0.3
        ###################################
        

        mvm_mask = np.ones((21+68+195+559,1))
        num_vertices = 21+68+195+559
        pb = np.random.random_sample()
        masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
        indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
        mvm_mask[indices,:] = 0.0
        mvm_mask = torch.from_numpy(mvm_mask).float()

        # hand_mvm_mask_ = hand_mvm_mask.expand(batch_size,-1,2051)
        mvm_mask_ = mvm_mask.expand(batch_size,-1,2051)


        start_time = time.time()

            # forward-pass
        pred_camera_temp, pred_3d_hand_kps, pred_3d_head_kps, pred_3d_hand_vs_sub, pred_3d_hand_vs, pred_3d_head_vs_sub2, pred_3d_head_vs_sub1, pred_3d_head_vs, pred_hand_contacts_sub, pred_hand_contacts, pred_head_contacts_sub2, pred_head_contacts_sub1, pred_head_contacts, pred_rh_betas, pred_rh_transl, pred_rh_rot, pred_rh_pose, pred_face_shape, pred_face_exp, pred_face_pose, pred_face_rot, pred_face_transl, pred_deformations_sub2, pred_deformations_sub1, pred_deformations, pred_contact_presence = DICE_model(single_image, rh_ref_kps, head_ref_kps, rh_ref_vs_sub, head_ref_vs_sub2, mvm_mask=mvm_mask_, is_train=True)

        infer_time = time.time() - start_time

        pred_deformations[:, neck_idx, :] = 0.0

        pred_jaw_pose = pred_face_pose[:, 3:]
        # gt_jaw_pose = gt_face_pose[:, 3:]

        pred_camera_s = 10 * torch.abs(pred_camera_temp[:,0])
        pred_camera_t = pred_camera_temp[:,1:]
        pred_camera = torch.zeros_like(pred_camera_temp)
        pred_camera[:,0] = pred_camera_s
        pred_camera[:,1:] = pred_camera_t

        # print(single_image.shape)
        image_size = torch.tensor([single_image.shape[2], single_image.shape[3]]).repeat(batch_size,1).float().to(args.device)

        R = torch.diag(torch.tensor([-1, -1, 1])).unsqueeze(0).repeat(batch_size,1,1).to(args.device)
        T = torch.tensor([0, 0, 1]).unsqueeze(0).repeat(batch_size,1).to(args.device)

        cameras = OrthographicCameras(focal_length = pred_camera_s, principal_point = pred_camera_t, device=args.device, R=R, T=T, in_ndc=False, image_size = image_size)
        cameras_3x = OrthographicCameras(focal_length = pred_camera_s * 3.0, principal_point = pred_camera_t * 3.0, device=args.device, R=R, T=T, in_ndc=False, image_size = image_size * 3)

        pred_3d_head_vs_parametric, pred_3d_head_kps_parametric = flame_forwarding(
            flame_model=head_model,
            head_shape_params=pred_face_shape,
            head_expression_params=pred_face_exp,
            head_pose_params=pred_face_pose,
            head_rotation= pred_face_rot,
            head_transl= pred_face_transl,
            head_scale_params=  torch.ones((batch_size,1)).to(args.device),
            return2d=False,
            return_2d_verts=False
        )

        pred_3d_hand_vs_parametric, pred_3d_hand_kps_parametric = mano_forwarding(
              h_model=hand_model,
              betas=pred_rh_betas,
              transl= pred_rh_transl,
              rot= pred_rh_rot,
              pose=pred_rh_pose,
              return_2d=False,
              return_2d_verts=False
          )


        pred_head_center_parametric = pred_3d_head_vs_parametric.mean(dim=1, keepdim=True)
        pred_3d_head_vs_parametric = pred_3d_head_vs_parametric - pred_head_center_parametric
        pred_3d_hand_vs_parametric = pred_3d_hand_vs_parametric - pred_head_center_parametric
        pred_3d_head_kps_parametric = pred_3d_head_kps_parametric - pred_head_center_parametric
        pred_3d_hand_kps_parametric = pred_3d_hand_kps_parametric - pred_head_center_parametric

        pred_head_center_np = pred_3d_head_vs.mean(dim=1, keepdim=True)
        pred_3d_head_vs = pred_3d_head_vs - pred_head_center_np
        pred_3d_head_vs_sub2 = pred_3d_head_vs_sub2 - pred_head_center_np
        pred_3d_head_vs_sub1 = pred_3d_head_vs_sub1 - pred_head_center_np
        pred_3d_hand_vs = pred_3d_hand_vs - pred_head_center_np
        pred_3d_hand_vs_sub = pred_3d_hand_vs_sub - pred_head_center_np
        pred_3d_head_kps = pred_3d_head_kps - pred_head_center_np
        pred_3d_hand_kps = pred_3d_hand_kps - pred_head_center_np


        # get 3d kps from face and hand model
        pred_3d_head_kps_from_model = head_model.convert_vs2landmarks(pred_3d_head_vs.float())
        pred_3d_hand_kps_from_model = mano_model.get_3d_joints(pred_3d_hand_vs.float())

        # obtain 2d joints, which are projected from 3d joints of smpl mesh
        pred_2d_hand_kps = cameras.transform_points_screen(pred_3d_hand_kps)[:, :, :2]
        pred_2d_head_kps = cameras.transform_points_screen(pred_3d_head_kps)[:, :, :2]
        pred_2d_hand_kps_from_model = cameras.transform_points_screen(pred_3d_hand_kps_from_model)[:, :, :2]
        pred_2d_head_kps_from_model = cameras.transform_points_screen(pred_3d_head_kps_from_model)[:, :, :2]
        pred_2d_hand_kps_parametric = cameras.transform_points_screen(pred_3d_hand_kps_parametric)[:, :, :2]
        pred_2d_head_kps_parametric = cameras.transform_points_screen(pred_3d_head_kps_parametric)[:, :, :2]

        pred_2d_hand_vs = cameras.transform_points_screen(pred_3d_hand_vs)[:, :, :2]
        pred_2d_head_vs = cameras.transform_points_screen(pred_3d_head_vs)[:, :, :2]

        pred_2d_hand_vs_sub = cameras.transform_points_screen(pred_3d_hand_vs_sub)[:, :, :2]
        pred_2d_head_vs_sub2 = cameras.transform_points_screen(pred_3d_head_vs_sub2)[:, :, :2]

        pred_3d_hand_vs_parametric_sub = hand_sampler.downsample(pred_3d_hand_vs_parametric.double())
        pred_3d_head_vs_parametric_sub2 = head_sampler.downsample(pred_3d_head_vs_parametric.double())

        pred_2d_hand_vs_parametric_sub = cameras.transform_points_screen(pred_3d_hand_vs_parametric_sub.float())[:, :, :2]
        pred_2d_head_vs_parametric_sub2 = cameras.transform_points_screen(pred_3d_head_vs_parametric_sub2.float())[:, :, :2]

        inference_time.update(infer_time / batch_size, batch_size)
        print(f"current inference time: {infer_time} for batch size {batch_size}")
        print(f"running average inference time: {inference_time.avg}")

        batch_time.update(time.time() - end)
        end = time.time()



        if True:
            if not os.path.exists(args.output_dir + f"/mesh_{epoch}_{iteration}"):
                try:
                    os.mkdir(args.output_dir + f"/mesh_{epoch}_{iteration}") 
                except FileExistsError:
                    print(args.output_dir + f"/mesh_{epoch}_{iteration} + " + "already exists")
  
            raster_settings = RasterizationSettings(
                image_size=single_image.shape[2], 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )
            if args.local_rank == 0:
                visual_imgs = []
                gt_kp_imgs = []
                pred_kp_imgs = []
                pred_kp_parametric_imgs = []
                gt_vs_imgs = []
                pred_vs_imgs = []
                pred_vs_parametric_imgs = []
                input_imgs = []
                rasterizer=MeshRasterizer(
                        cameras=cameras, 
                        raster_settings=raster_settings
                    ).to(args.device)
                lights = PointLights(device=args.device, location=[[0.0, 0.0, -3.0]])
                shader=SoftPhongShader(
                    device=args.device, 
                    cameras=cameras,
                    lights=lights
                )
                vertices = merge_verts(pred_3d_hand_vs_parametric, pred_3d_head_vs_parametric - pred_deformations).float()
                faces = merge_faces(torch.from_numpy(hand_model.faces.astype(np.int32)), flame_faces).float().to(vertices.device)
                faces = faces.repeat(batch_size, 1, 1)
                texture_map = torch.ones((batch_size, vertices.shape[1], 3), dtype=torch.float32).to(args.device)  # (N, V, C)
                textures = TexturesVertex(verts_features=texture_map)
                mesh = Meshes(verts=vertices, faces=faces, textures=textures)
                fragments = rasterizer(mesh)
                images = shader(meshes=mesh, fragments=fragments)
                zbuf = fragments.zbuf.squeeze(-1) # B, 224, 224
                image_p = images[..., :3].detach().cpu().numpy()
                # print(zbuf.shape, depth_map.shape, image.shape)
                image_p = image_p[:min(batch_size, 10), :, :, :]*255
                image_p = image_p.reshape(-1, single_image.shape[2], 3)
                
                # render non-parametric mesh
                vertices = merge_verts(pred_3d_hand_vs, pred_3d_head_vs - pred_deformations).float()
                textures = TexturesVertex(verts_features=texture_map)
                mesh = Meshes(verts=vertices, faces=faces, textures=textures)
                fragments = rasterizer(mesh)
                images = shader(meshes=mesh, fragments=fragments)
                image_np = images[..., :3].detach().cpu().numpy()
                image_np = image_np[:min(batch_size, 10), :, :, :]*255
                image_np = image_np.reshape(-1, single_image.shape[2], 3)
                print("rendering complete")
                for i in range(min(batch_size, 10)):
                    img = visualize_keypoints_single(single_image[i], pred_2d_hand_kps_from_model[i].detach() / single_image.shape[2], pred_2d_head_kps_from_model[i].detach() / single_image.shape[2], color1=(0,255,0), color2=(0,0,255))
                    img = overlay_number(img, data["img_name"][i])
                    pred_kp_imgs.append(img)
                    img = visualize_keypoints_single(single_image[i], pred_2d_hand_kps_parametric[i].detach() / single_image.shape[2], pred_2d_head_kps_parametric[i].detach() / single_image.shape[2], color1=(0,255,0), color2=(0,0,255))
                    img = overlay_number(img, data["img_name"][i])
                    pred_kp_parametric_imgs.append(img)
                    img = visualize_keypoints_single(single_image[i], pred_2d_hand_vs_sub[i].detach() / single_image.shape[2], pred_2d_head_vs_sub2[i].detach() / single_image.shape[2], color1=(0,255,0), color2=(0,0,255))
                    img = overlay_number(img, data["img_name"][i])
                    pred_vs_imgs.append(img)
                    img = visualize_keypoints_single(single_image[i], pred_2d_hand_vs_parametric_sub[i].detach() / single_image.shape[2], pred_2d_head_vs_parametric_sub2[i].detach() / single_image.shape[2], color1=(0,255,0), color2=(0,0,255))
                    img = overlay_number(img, data["img_name"][i])
                    pred_vs_parametric_imgs.append(img)
                    img = single_image[i].squeeze().cpu().detach()
                    denormalizer = transforms.Compose([
                        transforms.Normalize(mean=[0., 0., 0.],
                                            std=[1/0.229, 1/0.224, 1/0.225]),
                        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                            std=[1., 1., 1.])
                    ])
                    img = denormalizer(img).permute(1,2,0).numpy()
                    img = 255 * img
                    img = np.ascontiguousarray(img, dtype=np.uint8)
                    input_imgs.append(img)

                    texture_map = 0.8 * torch.ones((1, pred_3d_head_vs_parametric.shape[1], 3), dtype=torch.float32).to(args.device)  # (N, V, C)
                    textures = TexturesVertex(verts_features=texture_map)
                    contact_map = pred_head_contacts[i].squeeze().detach()
                    contact_map = (contact_map - contact_map.max()) / (contact_map.min() - contact_map.max())
                    texture_map[:, :, 1] = contact_map
                    texture_map = texture_map.squeeze()
                    save_mesh_to_ply(pred_3d_head_vs_parametric[i] - pred_deformations[i], flame_faces, texture_map, args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_head_mesh_with_contacts.ply")

                    texture_map = 0.8 * torch.ones((1, pred_3d_hand_vs_parametric.shape[1], 3), dtype=torch.float32).to(args.device)  # (N, V, C)
                    textures = TexturesVertex(verts_features=texture_map)
                    contact_map = pred_hand_contacts[i].squeeze().detach()
                    contact_map = (contact_map - contact_map.max()) / (contact_map.min() - contact_map.max())
                    texture_map[:, :, 1] = contact_map
                    texture_map = texture_map.squeeze()
                    save_mesh_to_ply(pred_3d_hand_vs_parametric[i], hand_model.faces, texture_map, args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_hand_mesh_with_contacts.ply")

                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_pred_head_mesh_{i}.obj", pred_3d_head_vs[i].cpu().detach().numpy(), flame_faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_pred_hand_mesh_{i}.obj", pred_3d_hand_vs[i].cpu().detach().numpy(), hand_model.faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_pred_head_mesh_parametric_undeformed_{i}.obj", pred_3d_head_vs_parametric[i].cpu().detach().numpy(), flame_faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_pred_hand_mesh_parametric_{i}.obj", pred_3d_hand_vs_parametric[i].cpu().detach().numpy(), hand_model.faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_pred_head_mesh_parametric_deformed_{i}.obj", (pred_3d_head_vs_parametric[i] - pred_deformations[i]).cpu().detach().numpy(), flame_faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_head_kp_pred_np_{i}.obj", pred_3d_head_kps[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_head_kp_pred_from_model_{i}.obj", pred_3d_head_kps_from_model[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_head_kp_pred_parametric_{i}.obj", pred_3d_head_kps_parametric[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_hand_kp_pred_np_{i}.obj", pred_3d_hand_kps[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_hand_kp_pred_from_model_{i}.obj", pred_3d_hand_kps_from_model[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_hand_kp_pred_parametric_{i}.obj", pred_3d_hand_kps_parametric[i].cpu().detach().numpy(), np.array([]))


                pred_kp_imgs = np.vstack(pred_kp_imgs)
                pred_vs_imgs = np.vstack(pred_vs_imgs)
                pred_kp_parametric_imgs = np.vstack(pred_kp_parametric_imgs)
                pred_vs_parametric_imgs = np.vstack(pred_vs_parametric_imgs)
                input_imgs = np.vstack(input_imgs)
                pred_depth_maps = zbuf[:min(batch_size, 10), :, :]
                pred_depth_maps = process_zbuf(pred_depth_maps).detach().cpu().numpy() # (N, H, W)
                pred_depth_maps_vis = depth_to_heatmap_batch(pred_depth_maps)
                pred_depth_maps_vis = np.vstack(pred_depth_maps_vis)
                pred_depth_maps_vis = (pred_depth_maps_vis * 255).astype(np.uint8)
                visual_imgs = np.hstack([pred_kp_imgs, pred_kp_parametric_imgs, pred_vs_imgs, pred_vs_parametric_imgs, input_imgs, image_p, image_np, pred_depth_maps_vis])

                raster_settings_3x = RasterizationSettings(
                    image_size=single_image.shape[2] * 3, 
                    blur_radius=0.0, 
                    faces_per_pixel=1, 
                )
                rasterizer=MeshRasterizer(
                        cameras=cameras_3x, 
                        raster_settings=raster_settings_3x
                    ).to(args.device)
                lights = PointLights(device=args.device, location=[[0.0, 0.0, -3.0]])
                shader=SoftPhongShader(
                    device=args.device, 
                    cameras=cameras_3x,
                    lights=lights
                )

                print("no shift")
                vertices = merge_verts(pred_3d_hand_vs_parametric, pred_3d_head_vs_parametric - pred_deformations).float()
                if args.smoothing:
                    vertices = one_euro_filter(iteration, vertices.detach())
                    print("smoothing enabled")
                else:
                    print("no smoothing")

                faces = merge_faces(torch.from_numpy(hand_model.faces.astype(np.int32)), flame_faces).float().to(vertices.device)
                faces = faces.repeat(batch_size, 1, 1)
                texture_map = torch.ones((batch_size, vertices.shape[1], 3), dtype=torch.float32).to(args.device)  # (N, V, C)
                textures = TexturesVertex(verts_features=texture_map)
                mesh = Meshes(verts=vertices, faces=faces, textures=textures)
                fragments = rasterizer(mesh)
                images = shader(meshes=mesh, fragments=fragments)
                zbuf = fragments.zbuf.squeeze(-1) # B, 224, 224
                image_p_3x = images[..., :3].detach().cpu().numpy()
                image_p_3x = image_p_3x[:min(batch_size, 10), :, :, :]*255
                image_p_3x = image_p_3x.reshape(-1, single_image.shape[2] * 3, 3)
                input_imgs_3x = [imgs_3x[i].permute(1,2,0).detach().cpu().numpy() for i in range(min(batch_size, 10))]
                input_imgs_3x = np.vstack(input_imgs_3x) * 255.0
                print("input_imgs_3x.shape", input_imgs_3x.shape)
                print("image_p_3x.shape", image_p_3x.shape)
                dice_vid_frames = np.hstack([input_imgs_3x, image_p_3x])

                wandb.log(step=iteration, data=
                    {"visuals": [wandb.Image(cv2.cvtColor(visual_imgs, cv2.COLOR_RGB2BGR), caption="2d keypoints and 3d vertices")]})


            if args.local_rank == 0:
                stamp = str(epoch) + '_' + str(iteration)
                for i in range(batch_size):
                    size = dice_vid_frames.shape[0] // batch_size
                    cur_frame = dice_vid_frames[i*size:(i+1)*size]
                    index = (iteration-1) * batch_size + i
                    cv2.imwrite(args.output_dir + '/vis_frames/' + str(index).zfill(5) + '.jpg', cur_frame)

    print("visualization finished")
        


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml", default='imagenet2012/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--val_yaml", default='imagenet2012/test.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.") 
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=200, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=1000.0, type=float)          
    parser.add_argument("--joints_loss_weight_3d", default=1000.0, type=float)
    parser.add_argument("--joints_loss_weight_2d", default=500.0, type=float)
    parser.add_argument("--contacts_loss_weight", default=300.0, type=float)
    parser.add_argument("--params_loss_weight", default=500.0, type=float)
    parser.add_argument('--collision_loss_weight', type=float, default=500.0)
    parser.add_argument('--touch_loss_weight', type=float, default=100.0)
    parser.add_argument('--normal_vector_loss_weight', type=float, default=0.0)
    parser.add_argument('--edge_length_loss_weight', type=float, default=0.0)
    parser.add_argument('--presence_loss_weight', type=float, default=500.0)
    parser.add_argument("--vloss_head_full", default=0.2, type=float)
    parser.add_argument("--vloss_head_sub", default=0.2, type=float)
    parser.add_argument("--vloss_head_sub2", default=0.2, type=float)
    parser.add_argument("--vloss_hand_full", default=0.2, type=float)
    parser.add_argument("--vloss_hand_sub", default=0.2, type=float)
    parser.add_argument("--closs_hand_full", default=0.2, type=float)
    parser.add_argument("--closs_hand_sub", default=0.2, type=float)
    parser.add_argument("--closs_head_full", default=0.2, type=float)
    parser.add_argument("--closs_head_sub", default=0.2, type=float)
    parser.add_argument("--closs_head_sub2", default=0.2, type=float)
    # parser.add_argument("--vloss_w_full", default=0.33, type=float) 
    # parser.add_argument("--vloss_w_sub", default=0.33, type=float) 
    # parser.add_argument("--vloss_w_sub2", default=0.33, type=float) 
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--output_feat_dim", default='512,128,3', type=str, 
                        help="The Image Feature Dimension.")                       
    parser.add_argument("--legacy_setting", default=True, action='store_true',)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument('--logging_steps', type=int, default=500, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--local-rank", type=int, default=0, 
                        help="For distributed training.")


    ##############################################
    #FAST
    parser.add_argument("--model_dim_1", default=512, type=int)
    parser.add_argument("--model_dim_2", default=128, type=int)
    parser.add_argument("--feedforward_dim_1", default=2048, type=int)
    parser.add_argument("--feedforward_dim_2", default=512, type=int)
    parser.add_argument("--conv_1x1_dim", default=2048, type=int)
    parser.add_argument("--transformer_dropout", default=0.1, type=float)
    parser.add_argument("--transformer_nhead", default=8, type=int)
    parser.add_argument("--pos_type", default='sine', type=str)
    parser.add_argument("--edge_length_loss", default="false", type=str)
    parser.add_argument("--normal_vector_loss", default="false", type=str)
    ############################

    # DeCaf
    parser.add_argument('--win_size', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cam_space_deform', type=int, default=1) 
    parser.add_argument('--back_aug', type=int, default=1)
    parser.add_argument('--train_imgrot_aug', type=int, default=1)
    parser.add_argument('--img_wh', type=tuple, default=(1920,1080))
    parser.add_argument('--max_epoch', type=int, default=1500)
    parser.add_argument('--n_pca', type=int, default=45)
    parser.add_argument('--pre_train', type=int, default=199)
    parser.add_argument('--dist_thresh', type=float, default=0.1)
    parser.add_argument('--hidden', type=int, default=5023*1)
    parser.add_argument('--dyn_iter', type=int, default=200)#args.num_workers
    parser.add_argument('--deform_thresh', type=int, default=0)
    parser.add_argument('--flipping', type=int, default=1)
    parser.add_argument('--debug_val', type=str, default="false")
    parser.add_argument('--data_path', default='../datasets/DecafDataset/', type=str)
    parser.add_argument('--image_data_path', type=str, default="/code/datasets/DecafDataset_images/") 
    parser.add_argument('--single_image_path', type=str, default="/code/datasets/Decaf_imgs_single/")
    parser.add_argument('--model', type=str, default="dice")
    parser.add_argument('--deform_reg_weight', type=float, default=10.0)
    parser.add_argument('--deformation_loss_weight', type=float, default=5000.0)
    parser.add_argument('--inthewild_deformation_loss_weight', type=float, default=1000.0)
    parser.add_argument('--depth_loss_weight', type=float, default=100.0)
    parser.add_argument('--discriminator_loss_weight', type=float, default=100.0)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--decaf_multiscale', action='store_true')
    parser.add_argument('--scale_range', type=str, default="0.75,1.25")
    parser.add_argument('--burn_in_epochs', type=int, default=0)
    parser.add_argument('--itw_resample', type=int, default=50)
    parser.add_argument('--inthewild_root_dir', type=str, default="/code/datasets/itw_new_crop_center_clean")
    parser.add_argument('--upsample_factor', type=int, default=2)
    parser.add_argument('--input_folder', type=str, default="")
    parser.add_argument('--smoothing', default=False, action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    if args.local_rank == 0:
        wandb_run = wandb.init(project="dice", name=args.run_name, config=args)
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        synchronize()

    mkdir(args.output_dir)
    logger = setup_logger("DICE", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))


    asset_path = args.data_path+"/assets/"
    mano_model_path = asset_path+'/mano_v1_2/models/MANO_RIGHT.pkl'
    flame_model_path = asset_path+"/generic_model.pkl"
    flame_landmark_path = asset_path+"/landmark_embedding.npy"

    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.cuda()
    rh_model = mano.model.load(
              model_path=mano_model_path,
              is_right= True, 
              num_pca_comps=args.n_pca, 
              batch_size=1, 
              flat_hand_mean=True).to(args.device)

    flame_model = FLAME(flame_model_path, flame_landmark_path).to(args.device)
    flame_faces = get_FLAME_faces(flame_model_path)

    transforms = torch.load("head_mesh_transforms.pt")
    A, U, D, F = transforms["A"], transforms["U"], transforms["D"], transforms["F"]
    head_F = [f.numpy() for f in F]
    head_sampler = Mesh(A, U, D, num_downsampling=2, nsize=1)

    # torch.save(A[1], "head_559_adjacency_matrix.pt")

    transforms = torch.load("hand_mesh_transforms.pt")
    A, U, D, F = transforms["A"], transforms["U"], transforms["D"], transforms["F"]
    hand_F = [f.numpy() for f in F]
    hand_sampler = Mesh(A, U, D, num_downsampling=1, nsize=1)

    # torch.save(A[0], "hand_195_adjacency_matrix.pt")

    print("adjacency matrix save success")


    # Renderer for visualization
    hand_renderer = None
    face_renderer = None

    # Module rename compatibility
    import src.modeling.bert.model
    import src.modeling.hrnet.hrnet_cls_net_featmaps
    
    if 'metro' not in sys.modules:
        sys.modules['metro'] = types.ModuleType('metro')
    if 'metro.modeling' not in sys.modules:
        sys.modules['metro.modeling'] = types.ModuleType('metro.modeling')
    if 'metro.modeling.bert' not in sys.modules:
        sys.modules['metro.modeling.bert'] = types.ModuleType('metro.modeling.bert')
    if 'metro.modeling.bert.modeling_metro' not in sys.modules:
        sys.modules['metro.modeling.bert.modeling_metro'] = types.ModuleType('metro.modeling.bert.modeling_metro')
    
    # Map modeling_bert module
    if 'metro.modeling.bert.modeling_bert' not in sys.modules:
        import src.modeling.bert.bert
        sys.modules['metro.modeling.bert.modeling_bert'] = src.modeling.bert.bert

    # Map param_regressor module
    if 'metro.modeling.bert.param_regressor' not in sys.modules:
        import src.modeling.bert.param_regressor
        sys.modules['metro.modeling.bert.param_regressor'] = src.modeling.bert.param_regressor

    if 'metro.modeling.hrnet' not in sys.modules:
        sys.modules['metro.modeling.hrnet'] = types.ModuleType('metro.modeling.hrnet')
    
    # Map the class
    sys.modules['metro.modeling.bert.modeling_metro'].METRO_Decaf_Network_DeformBranch_Presence = src.modeling.bert.model.DICE_Network
    sys.modules['metro.modeling.bert.modeling_metro'].METRO_Encoder = src.modeling.bert.model.DICE_Encoder
    sys.modules['metro.modeling.bert.modeling_metro'].METRO = src.modeling.bert.model.DICE_Module
    sys.modules['metro.modeling.bert'].param_regressor = src.modeling.bert.param_regressor

    # Map the module
    sys.modules['metro.modeling.hrnet.hrnet_cls_net_featmaps'] = src.modeling.hrnet.hrnet_cls_net_featmaps

    _dice_network = torch.load(args.resume_checkpoint)
    
    # Fix for apex FusedLayerNorm missing memory_efficient attribute
    try:
        from apex.normalization.fused_layer_norm import FusedLayerNorm
        for module in _dice_network.modules():
            if isinstance(module, FusedLayerNorm):
                if not hasattr(module, 'memory_efficient'):
                    setattr(module, 'memory_efficient', False)
    except ImportError:
        pass

    _dice_network.to(args.device)
    _dice_network.eval()
    hand_discriminator = None
    face_discriminator = None
    

    train_dataloader = make_folder_data_loader(args, args.distributed, is_train=False, drop_last=False)
    val_dataloader = None
    run(args, train_dataloader, val_dataloader, _dice_network, mano_model=mano_model, hand_model=rh_model, head_model=flame_model, hand_sampler=hand_sampler, head_sampler=head_sampler, hand_renderer=hand_renderer, face_renderer=face_renderer, hand_F=hand_F, head_F=head_F, hand_discriminator=hand_discriminator, face_discriminator=face_discriminator)


if __name__ == "__main__":
    args = parse_args()
    main(args)
