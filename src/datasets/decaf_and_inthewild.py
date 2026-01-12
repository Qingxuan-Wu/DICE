from src.datasets.inthewild import InTheWildDataset
from src.datasets.decaf import make_decaf_dataset
from src.datasets.decaf_motion import make_decaf_motion_dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from src.datasets.renderme_motion import RenderMeMotionDataset
from src.datasets.freihand import FreihandMotionDataset
import torch
import random

class SampleArgs:
  def __init__(self):
    self.data_dir = "datasets"
    self.train_yaml = "imagenet2012/train.yaml"
    self.val_yaml = "imagenet2012/test.yaml"
    self.num_workers = 4
    self.img_scale_factor = 1
    self.model_name_or_path = "src/modeling/bert/bert-base-uncased/"
    self.resume_checkpoint = None
    self.output_dir = "output/"
    self.config_name = ""
    self.per_gpu_train_batch_size = 30
    self.per_gpu_eval_batch_size = 30
    self.lr = 0.0001
    self.num_train_epochs = 200
    self.vertices_loss_weight = 1000.0
    self.joints_loss_weight_3d = 1000.0
    self.joints_loss_weight_2d = 500.0
    self.contacts_loss_weight = 300.0
    self.params_loss_weight = 500.0
    self.collision_loss_weight = 500.0
    self.touch_loss_weight = 100.0
    self.normal_vector_loss_weight = 0.0
    self.edge_length_loss_weight = 0.0
    self.vloss_head_full = 0.2
    self.vloss_head_sub = 0.2
    self.vloss_head_sub2 = 0.2
    self.vloss_hand_full = 0.2
    self.vloss_hand_sub = 0.2
    self.closs_hand_full = 0.2
    self.closs_hand_sub = 0.2
    self.closs_head_full = 0.2
    self.closs_head_sub = 0.2
    self.closs_head_sub2 = 0.2
    self.drop_out = 0.1
    self.arch = "hrnet-w64"
    self.num_hidden_layers = 4
    self.hidden_size = -1
    self.num_attention_heads = 4
    self.intermediate_size = -1
    self.input_feat_dim = "2051,512,128"
    self.hidden_feat_dim = "1024,256,128"
    self.output_feat_dim = "512,128,3"
    self.legacy_setting = True
    self.run_eval_only = False
    self.logging_steps = 500
    self.device = "cuda"
    self.seed = 88
    self.local_rank = 0
    self.model_dim_1 = 512
    self.model_dim_2 = 128
    self.feedforward_dim_1 = 2048
    self.feedforward_dim_2 = 512
    self.conv_1x1_dim = 2048
    self.transformer_dropout = 0.1
    self.transformer_nhead = 8
    self.pos_type = "sine"
    self.edge_length_loss = "false"
    self.normal_vector_loss = "false"
    self.win_size = 0
    self.save = 0
    self.cam_space_deform = 1
    self.back_aug = 1
    self.train_imgrot_aug = 1
    self.img_wh = (1920, 1080)
    self.max_epoch = 1500
    self.n_pca = 45
    self.pre_train = 199
    self.dist_thresh = 0.1
    self.hidden = 5023
    self.dyn_iter = 200
    self.deform_thresh = 0
    self.flipping = 1
    self.debug_val = "false"
    self.data_path = "/code/datasets/DecafDataset/"
    self.image_data_path = "/code/datasets/DecafDataset_images/"
    self.single_image_path = "/code/datasets/Decaf_imgs_single/"
    self.model = "decaf"
    self.deform_reg_weight = 10.0
    self.deformation_loss_weight = 5000.0


class DeCafAndInTheWildDataset(Dataset):
    def __init__(self, args, inthewild_root_dir=None, transform=None, decaf_prob=0.6):
        # self.decaf_dataset = Decaf_6()
        self.decaf_prob = decaf_prob
        self.inthewild_dataset = InTheWildDataset(args.inthewild_root_dir, transform, itw_image_num=args.inthewild_image_num)
        self.decaf_full_dataset = make_decaf_dataset(args, is_train=True)
        self.decaf_full_length = len(self.decaf_full_dataset)
        self.decaf_motion_dataset = make_decaf_motion_dataset(args, is_train=True)
        self.decaf_motion_length = len(self.decaf_motion_dataset)
        self.freihand_motion_dataset = FreihandMotionDataset()
        self.freihand_motion_length = len(self.freihand_motion_dataset)
        self.renderme_motion_dataset = RenderMeMotionDataset()
        self.renderme_motion_length = len(self.renderme_motion_dataset)
        self.dataset = args.dataset
        self.itw_resample = args.itw_resample
        print("inthewild + full decaf")

    def __len__(self):
        if self.dataset == "combined":
            return len(self.decaf_full_dataset) + self.itw_resample * len(self.inthewild_dataset)
        elif self.dataset == "inthewild":
            return len(self.inthewild_dataset)
        else:
            raise ValueError("Invalid dataset name")
        # return len(self.decaf_full_dataset)
        # return len(self.inthewild_dataset)

    def get_decaf_data(self, idx):
        decaf_data = self.decaf_full_dataset[idx]
        decaf_data['depth_map'] = torch.zeros(224, 224)
        decaf_data["dataset_name"] = "decaf"
        decaf_data["data_index"] = idx
        data = decaf_data
        return data

    def get_inthewild_data(self, idx):
        inthewild_data = self.inthewild_dataset[idx]
        data = {}
        data["depth_map"] = inthewild_data["depth_map"]
        data["single_img_seqs"] = inthewild_data["image"]
        data["head_vs_in_cam"] = torch.zeros([5023, 3])
        data["rh_vs_in_cam"] = torch.zeros([778, 3])
        data["head_keys_in_cam"] = torch.zeros([68, 3])
        data["rh_keys_in_cam"] = torch.zeros([21, 3])
        data["dataset_name"] = "inthewild"
        data["data_index"] = idx
        data["head_vs_in_cam_deformed"] = torch.zeros([5023, 3])
        data['rh_keys_proj_single'] = inthewild_data['hand_keypoints']
        data['head_keys_proj_single'] = inthewild_data['face_keypoints']
        data['rh_vs_proj_single'] = torch.zeros([778, 2])
        data['head_vs_proj_single'] = torch.zeros([5023, 2])
        data['deformation_cam'] = torch.zeros([5023, 3])
        data['rh_con_labels'] = torch.zeros([778])
        data['head_con_labels'] = torch.zeros([5023])
        data['rh_ref_vs'] = inthewild_data['rh_ref_vs']
        data['head_ref_vs'] = inthewild_data['head_ref_vs']
        data['rh_betas'] = torch.zeros([10])
        data['rh_transl'] = torch.zeros([3])
        data['rh_rot'] = torch.zeros([3])
        data['rh_pose'] = torch.zeros([45])
        data['face_shape'] = torch.zeros([100])
        data['face_exp'] = torch.zeros([50])
        data['face_pose'] = torch.zeros([6])
        data['face_rot'] = torch.zeros([3])
        data['face_transl'] = torch.zeros([3])
        data['mode'] = 'n/a'
        data['sub_id'] = 'n/a'
        data['cam_id'] = 'n/a'
        data['frame_id'] = str(idx)
        # data["img_name"] = "_".join(inthewild_data["img_name"].split("/")[-2:])
        return data
        
    def __getitem__(self, idx):
        # print("in-the-wild and decaf ")
        if self.dataset == "combined":
            if idx < len(self.decaf_full_dataset):
                data = self.get_decaf_data(idx)
            else:
                idx = idx - self.decaf_full_length
                data = self.get_inthewild_data(idx % len(self.inthewild_dataset))
        elif self.dataset == "inthewild":
            data = self.get_inthewild_data(idx)

        if random.uniform(0, 1) <= self.decaf_prob:
            sample_idx = np.random.randint(self.decaf_motion_length)
            motion_sample = self.decaf_motion_dataset[sample_idx]
            data['sampled_face_shape'] = motion_sample['face_shape']
            data['sampled_face_exp'] = motion_sample['face_exp']
            data['sampled_face_pose'] = motion_sample['face_pose']
            # print("decaf face")
        else:
            sample_idx = np.random.randint(self.renderme_motion_length)
            motion_sample = self.renderme_motion_dataset[sample_idx]
            data['sampled_face_shape'] = motion_sample['face_shape']
            data['sampled_face_exp'] = motion_sample['face_exp']
            data['sampled_face_pose'] = motion_sample['face_pose']
            # print("renderme face")

        if random.uniform(0, 1) <= self.decaf_prob:
            sample_idx = np.random.randint(self.decaf_motion_length)
            motion_sample = self.decaf_motion_dataset[sample_idx]
            data['sampled_rh_betas'] = motion_sample['rh_betas']
            data['sampled_rh_pose'] = motion_sample['rh_pose']
            # print("decaf hand")
        else:
            sample_idx = np.random.randint(self.freihand_motion_length)
            motion_sample = self.freihand_motion_dataset[sample_idx]
            data['sampled_rh_betas'] = motion_sample['rh_betas']
            data['sampled_rh_pose'] = motion_sample['rh_pose']
            # print("freihand hand")

        if data["dataset_name"] == "decaf":
            data["has_2d_kp"] = 1
            data["has_3d_kp"] = 1
            data["has_3d_mesh"] = 1
            data["has_depth"] = 0
            data["has_contact"] = 1
            data["has_params"] = 1
            data["has_deform"] = 1
            # data["img_name"] = f'{data["mode"]} {data["sub_id"]} {data["cam_id"]} {data["frame_id"]}'
        else:
            data["has_2d_kp"] = 1
            data["has_3d_kp"] = 0
            data["has_3d_mesh"] = 0
            data["has_depth"] = 1
            data["has_contact"] = 0
            data["has_params"] = 0
            data["has_deform"] = 0

        return data