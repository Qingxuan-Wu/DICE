import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from src.decaf.FLAME_util import FLAME, flame_forwarding
from src.decaf.tracking_util import mano_forwarding
from src.modeling._mano import MANO
import src.decaf.system_util as su
import mano
import torch
import cv2
import random

def generate_random_choices(N, k=100, seed=42):
    random.seed(seed)
    return random.sample(range(1, N + 1), k)


class InTheWildDataset(Dataset):
    def __init__(self, root_dir, transform=None, itw_image_num=None):
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.image_folders = [os.path.join(root_dir, 'images', folder_name) for folder_name in os.listdir(os.path.join(root_dir, 'images'))]
        self.depth_map_folders = [os.path.join(root_dir, 'depth_maps', folder_name) for folder_name in os.listdir(os.path.join(root_dir, 'depth_maps'))]
        self.face_keypoints_folders = [os.path.join(root_dir, 'face_keypoints', folder_name) for folder_name in os.listdir(os.path.join(root_dir, 'face_keypoints'))]
        self.hand_keypoints_folders = [os.path.join(root_dir, 'hand_keypoints', folder_name) for folder_name in os.listdir(os.path.join(root_dir, 'hand_keypoints'))]

        self.image_folders = sorted(self.image_folders)
        self.depth_map_folders = sorted(self.depth_map_folders)
        self.face_keypoints_folders = sorted(self.face_keypoints_folders)
        self.hand_keypoints_folders = sorted(self.hand_keypoints_folders)

        self.image_list = []
        self.img_dict = {}
        self.depth_map_dict = {}
        self.face_keypoints_dict = {}
        self.hand_keypoints_dict = {}

        for image_folder in self.image_folders:
            image_files = sorted(os.listdir(image_folder))

            for image_file in image_files:
                img_name = os.path.join(image_folder, image_file)
                self.image_list.append(img_name)
                image = Image.open(img_name).convert('RGB')
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.img_dict[img_name] = image
                depth_name = img_name.replace("images", "depth_maps").replace(".jpg", "_pred.npy")
                face_keypoints_name = img_name.replace("images", "face_keypoints").replace(".jpg", ".npy")
                hand_keypoints_name = img_name.replace("images", "hand_keypoints").replace(".jpg", ".npy")
                self.depth_map_dict[img_name] = np.load(depth_name)
                self.face_keypoints_dict[img_name] = np.load(face_keypoints_name)
                self.hand_keypoints_dict[img_name] = np.load(hand_keypoints_name)
                # self.depth_map_list.append(os.path.join(depth_map_folder, depth_map_file))
                # self.face_keypoints_list.append(os.path.join(face_keypoints_folder, face_keypoints_file))
                # self.hand_keypoints_list.append(os.path.join(hand_keypoints_folder, hand_keypoints_file))

        data_path = "/code/datasets/DecafDataset"

        asset_path = data_path+"/assets/"
        mano_model_path = asset_path+'/mano_v1_2/models/MANO_RIGHT.pkl'
        flame_model_path = asset_path+"/generic_model.pkl"
        flame_landmark_path = asset_path+"/landmark_embedding.npy"

        self.mano_model = MANO()
        self.flame_model = FLAME(flame_model_path, flame_landmark_path)
        self.rh_model = mano.model.load(
              model_path=mano_model_path,
              is_right= True,
              num_pca_comps=45,
              flat_hand_mean=True)
        self.rh_model.eval()

        print(len(self.image_list), "length")

        self.length = len(self.image_list) if itw_image_num is None else itw_image_num
        self.itw_image_num = itw_image_num
        self.image_subset = None
        if itw_image_num is not None:
            print("Randomly selecting", itw_image_num, "images")
            self.image_subset = generate_random_choices(len(self.image_list), k=itw_image_num)

    def __len__(self):
        return self.length

    def get_head_ref_vs(self):
        [head_vs_model_rest, _] = \
            flame_forwarding(
            flame_model=self.flame_model,
            head_shape_params=torch.zeros(1, 100),
            head_expression_params=torch.zeros(1, 50),
            head_pose_params=torch.zeros(1, 6),
            head_rotation=torch.zeros(1, 3),
            head_transl=torch.zeros(1, 3),
            head_scale_params=su.np2tor(np.array([1.0])),
        )
        head_vs_model_rest = head_vs_model_rest[0].numpy()
        return head_vs_model_rest
    
    def get_rh_ref_vs(self):
        [rh_vs_model_rest, _] = \
            mano_forwarding(
            h_model=self.rh_model,
            betas=torch.zeros(1, 10),
            transl=torch.zeros(1, 3),
            rot=torch.zeros(1, 3),
            pose=torch.zeros(1, 45),
            return_2d=False,
        )
        rh_vs_model_rest = rh_vs_model_rest[0].numpy()
        return rh_vs_model_rest

    def __getitem__(self, idx):
        # img_name = os.path.join(self.image_folder, self.image_list[idx])
        # depth_name = os.path.join(self.depth_folder, self.depth_list[idx])
        # face_keypoints_name = os.path.join(self.face_keypoints_folder, self.face_keypoints_list[idx])
        # hand_keypoints_name = os.path.join(self.hand_keypoints_folder, self.hand_keypoints_list[idx])
        if self.itw_image_num is not None:
            idx = self.image_subset[idx]
        img_name = self.image_list[idx]
        # depth_name = img_name.replace("images", "depth_maps").replace(".jpg", "_pred.npy")
        # face_keypoints_name = img_name.replace("images", "face_keypoints").replace(".jpg", ".npy")
        # hand_keypoints_name = img_name.replace("images", "hand_keypoints").replace(".jpg", ".npy")

        ori_img = self.img_dict[img_name]
        depth_map = self.depth_map_dict[img_name]
        face_keypoints = self.face_keypoints_dict[img_name]
        hand_keypoints = self.hand_keypoints_dict[img_name]
        # depth_map = np.load(depth_name)
        # face_keypoints = np.load(face_keypoints_name)
        # hand_keypoints = np.load(hand_keypoints_name)

        # Resize depth map
        depth_map = Image.fromarray(depth_map)
        depth_map = depth_map.resize((224, 224), Image.BILINEAR)
        depth_map = np.array(depth_map)

        if self.transform:
            image = self.transform(ori_img).squeeze()
            depth_map = transforms.ToTensor()(depth_map).squeeze()
            face_keypoints = transforms.ToTensor()(face_keypoints).squeeze()
            hand_keypoints = transforms.ToTensor()(hand_keypoints).squeeze()

        rh_ref_vs = self.get_rh_ref_vs()
        head_ref_vs = self.get_head_ref_vs()

        return {
            'ori_img': ori_img, # 'ori_img' is the original image, 'image' is the transformed image
            'image': image, 
            'depth_map': depth_map, 
            'face_keypoints': face_keypoints, 
            'hand_keypoints': hand_keypoints,
            'rh_ref_vs': rh_ref_vs,
            'head_ref_vs': head_ref_vs,
            'img_name': img_name
            }


# Create dataset
