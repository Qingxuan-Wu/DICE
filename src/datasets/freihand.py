import torch
import argparse
from src.modeling._mano import MANO
import random
import os.path as op
from src.datasets.hand_mesh_tsv import (HandMeshTSVDataset, HandMeshTSVYamlDataset)


def build_hand_dataset(yaml_file, args, is_train=True, scale_factor=1):
    print(yaml_file)
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        # code.interact(local=locals())
        assert op.isfile(yaml_file)
    return HandMeshTSVYamlDataset(args, yaml_file, is_train, False, scale_factor)

def build_freihand_dataset(freihand_path=None):
    if not freihand_path:
        freihand_path = "/code/datasets/metro/freihand/train.yaml"
    return build_hand_dataset(freihand_path, None, is_train=True, scale_factor=1)


class FreihandDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, n_samples):
        dataset = build_freihand_dataset()
        self.dataset = dataset
        self.mano = MANO()
        self.mano.eval()
        self.n_samples = n_samples
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        idx = random.randrange(0, len(self.dataset))
        image, transformed_images, annotations = self.dataset[idx]
        data = {}        
        transformed_images = transformed_images[[2,1,0],:,:]
        gt_pose, gt_betas = annotations['pose'].unsqueeze(0), annotations['betas'].unsqueeze(0)
        data["dataset_name"] = "freihand"
        data['single_img_seqs'] = transformed_images
        data['rh_vs_in_cam'], data['rh_keys_in_cam'] = self.mano.layer(gt_pose, gt_betas)
        data['rh_vs_in_cam'] = data['rh_vs_in_cam'].squeeze(0)
        data['rh_keys_in_cam'] = data['rh_keys_in_cam'].squeeze(0)
        data['rh_keys_proj_single'] = annotations['joints_2d'][:,:-1]

        data["rh_vs_proj_single"] = torch.zeros(778, 2)
        data["head_keys_proj_single"] = torch.zeros(68, 2)
        data["head_vs_proj_single"] = torch.zeros(5023, 2)
        data["head_vs_in_cam_deformed"] = torch.zeros(5023, 3)
        data["deformation_cam"] = torch.zeros(5023, 3)
        data["head_vs_in_cam"] = torch.zeros(5023, 3)
        data["head_keys_in_cam"] = torch.zeros(68, 3)
        data["rh_con_labels"] = torch.zeros(778)
        data["head_con_labels"] = torch.zeros(5023)
        
        data["rh_betas"] = gt_betas.squeeze()
        data["rh_pose"] = gt_pose.squeeze()[3:]
        data["rh_rot"] = torch.zeros(3)
        data["rh_transl"] = torch.zeros(3)

        data["face_shape"] = torch.zeros(100)
        data["face_exp"] = torch.zeros(50)
        data["face_rot"] = torch.zeros(3)
        data["face_transl"] = torch.zeros(3)
        data["face_pose"] = torch.zeros(6)

        # print(data["rh_vs_in_cam"][0].shape)
        # print(data["rh_vs_in_cam"][1].shape)
        # input()

        # for key, value in data.items():
        #     if hasattr(value, 'shape'):
        #         print(key, value.shape)
        #     else:
        #         print(key, value)

        # input("freihand shape")
        

        return data



class FreihandMotionDataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset = build_freihand_dataset()
        self.dataset = dataset
        self.mano = MANO()
        self.mano.eval()
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        idx = random.randrange(0, len(self.dataset))
        image, transformed_images, annotations = self.dataset[idx]
        data = {}        
        transformed_images = transformed_images[[2,1,0],:,:]
        gt_pose, gt_betas = annotations['pose'].unsqueeze(0), annotations['betas'].unsqueeze(0)
        # data["dataset_name"] = "freihand"
        # data['single_img_seqs'] = transformed_images
        # data['rh_vs_in_cam'], data['rh_keys_in_cam'] = self.mano.layer(gt_pose, gt_betas)
        # data['rh_vs_in_cam'] = data['rh_vs_in_cam'].squeeze(0)
        # data['rh_keys_in_cam'] = data['rh_keys_in_cam'].squeeze(0)
        # data['rh_keys_proj_single'] = annotations['joints_2d'][:,:-1]

        # data["rh_vs_proj_single"] = torch.zeros(778, 2)
        # data["head_keys_proj_single"] = torch.zeros(68, 2)
        # data["head_vs_proj_single"] = torch.zeros(5023, 2)
        # data["head_vs_in_cam_deformed"] = torch.zeros(5023, 3)
        # data["deformation_cam"] = torch.zeros(5023, 3)
        # data["head_vs_in_cam"] = torch.zeros(5023, 3)
        # data["head_keys_in_cam"] = torch.zeros(68, 3)
        # data["rh_con_labels"] = torch.zeros(778)
        # data["head_con_labels"] = torch.zeros(5023)
        
        data["rh_betas"] = gt_betas.squeeze()
        data["rh_pose"] = gt_pose.squeeze()[3:]
        # data["rh_rot"] = torch.zeros(3)
        # data["rh_transl"] = torch.zeros(3)

        # data["face_shape"] = torch.zeros(100)
        # data["face_exp"] = torch.zeros(50)
        # data["face_rot"] = torch.zeros(3)
        # data["face_transl"] = torch.zeros(3)
        # data["face_pose"] = torch.zeros(6)

        # print(data["rh_vs_in_cam"][0].shape)
        # print(data["rh_vs_in_cam"][1].shape)
        # input()

        # for key, value in data.items():
        #     if hasattr(value, 'shape'):
        #         print(key, value.shape)
        #     else:
        #         print(key, value)

        # input("freihand shape")
        

        return data

