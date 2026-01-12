import copy
import torch
import numpy as np
import torchvision.transforms as transforms
from src.decaf.util import denormalize_keys, keypoint_overlay

def visualize_keypoints_single(img, hand_kps, head_kps, color1=(255,0,0), color2=(255,0,0), w=224, h=224):
    hand_kps_vis = copy.deepcopy(hand_kps)
    head_kps_vis = copy.deepcopy(head_kps)
    hand_kps_vis = denormalize_keys(keys=hand_kps_vis,w=w,h=h)
    hand_kps_vis = hand_kps_vis.squeeze().cpu().detach().numpy()
    head_kps_vis = denormalize_keys(keys=head_kps_vis,w=w,h=h)
    head_kps_vis = head_kps_vis.squeeze().cpu().detach().numpy()
    img = img.squeeze().cpu().detach()
    denormalizer = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                            std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                            std=[1., 1., 1.])
    ])
    img = denormalizer(img).permute(1,2,0).numpy()
    img = 255 * img
    img = np.ascontiguousarray(img, dtype=np.uint8)
    img = keypoint_overlay(
        head_kps_vis.astype(int),
        c=color1,
        img=img)
    img = keypoint_overlay(
        hand_kps_vis.astype(int),
        c=color2,
        img=img)
    return img