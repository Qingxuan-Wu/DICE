
import copy
import numpy as np
import functools
import random
from PIL import Image 
import torch
import torchvision.transforms as T 

def get_valid_label(data: list, n_frames: int) -> np.ndarray:
    """
    a function that returns binary valid labels of a given sequence
    args:
      data : a list of tuples that contains start and end frame id of invalid seqs
      n_frames: number of frames of the target sequence
    returns:
      valid_ids: a binary label vector that indicates the validity of the frames
    """
    invalid_ids = [list(np.arange(seq[0], seq[1])) for seq in data]
    invalid_ids = np.array(functools.reduce(lambda x, y: x+y, invalid_ids))
    invalid_ids = np.unique(invalid_ids)
    valid_ids = np.ones(n_frames)
    valid_ids[invalid_ids] = 0
    return valid_ids

def get_valid_label_act_rem(data: list, n_frames: int, remacts: list) -> np.ndarray:
    """
    a function that returns binary valid labels of a given sequence
    args:
      data : a list of tuples that contains start and end frame id of invalid seqs
      n_frames: number of frames of the target sequence
    returns:
      valid_ids: a binary label vector that indicates the validity of the frames
    """
    invalid_ids = [list(np.arange(seq[0], seq[1])) for seq in data]
    #print(invalid_ids)
    
    invalid_ids_act_all=[]
    for act in remacts:
      invalid_ids_act = [list(np.arange(seq[0], seq[1])) for seq in act]
      #invalid_ids = np.array(functools.reduce(lambda x, y: x+y, invalid_ids))
      invalid_ids_act= [j for sub in invalid_ids_act for j in sub]
      invalid_ids_act = np.unique(invalid_ids_act)
      #print(invalid_ids_act)
      invalid_ids_act_all+=list(invalid_ids_act)
      #print(invalid_ids_act_all)
    invalid_ids_act_all = np.unique(invalid_ids_act_all)
    invalid_ids = np.array(functools.reduce(lambda x, y: x+y, invalid_ids))
    
    invalid_ids=np.concatenate((invalid_ids,invalid_ids_act_all))
    invalid_ids = np.unique(invalid_ids) 
    valid_ids = np.ones(n_frames)
    valid_ids[invalid_ids] = 0
     
    return valid_ids

class Augmenter(torch.nn.Module):
    def __init__(self, aug, agu_prob=0.8):
        super().__init__()
        blurrer = T.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5))
        jitter = T.ColorJitter(brightness=.5, hue=.3)
        posterizer = T.RandomPosterize(bits=2)
        sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=2)
        autocontraster = T.RandomAutocontrast()
        equalizer = T.RandomEqualize()
        self.augs = [blurrer, jitter, posterizer,
                     sharpness_adjuster, autocontraster, equalizer]
        self.aug = aug
        self.agu_prob = agu_prob
        self.toten = T.ToTensor()
        self.normalization = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def catter(self, imgs):
        for i in range(len(imgs)):
            if i == 0:
                catimg = imgs[i]
            else:
                catimg = np.concatenate((catimg, imgs[i]), axis=1)

        return catimg

    def forward(self, imgs):
        h, w, c = imgs[0].shape
        n_imgs = len(imgs)
        imgs_cat = self.catter(imgs)

        if self.aug and random.uniform(0, 1) <= self.agu_prob:
            augid = random.randint(0, len(self.augs)-1)
            imgs_cat = np.array(self.augs[augid](Image.fromarray(imgs_cat)))
        else:
            imgs_cat = np.array(Image.fromarray(imgs_cat))
        imgs_cat = self.normalization(imgs_cat)
        imgs = torch.stack([imgs_cat[:, :, i*w:(i+1)*w]
                           for i in range(n_imgs)])
        return imgs
    
    #@classmethod
    #def _tensorimg2numpyimg(self,tensorimg):
    #    #tensorimg:Bxwin_sizexCxHxW 
    #    return system_util.tor2np(tensorimg.permute(0,1,3,4,2))
     
    def testing_norm(self, imgs):
        h, w, c = imgs[0].shape
        n_imgs = len(imgs)
        imgs_cat = self.catter(imgs) 
        imgs_cat = np.array(Image.fromarray(imgs_cat))
        imgs_cat = self.normalization(imgs_cat)
        imgs = torch.stack([imgs_cat[:, :, i*w:(i+1)*w]
                           for i in range(n_imgs)])
        return imgs
    

def ten2opencv(tensor_img):
    return (tensor_img.permute(1, 2, 0).numpy()*255).astype(np.uint8)


def prepro_backimage(back_img):
    """ 
    Args:
        back_img (H,W,C):opencv image

    Returns:
         (H,W,C):opencv image after cropping and random flipping
    """
    toten = T.Compose([
        T.ToTensor(),
        T.RandomCrop(224),
        # T.RandomHorizontalFlip(p=0.5),
    ])
    # .permute(1,2,0).numpy()*255).astype(np.uint8)
    return ten2opencv(toten(back_img))


def get_inverse_mask(mask):
    inv_mask = copy.copy(mask)
    inv_mask[inv_mask > 0] = -1
    inv_mask += 1
    return inv_mask


def get_image_with_bg(fg_img, bg_img, mask, is_cropped=False):
    fg_img = copy.copy(fg_img)
    bg_img = copy.copy(bg_img)
    mask = copy.copy(mask)
    
    if is_cropped:
        cropped_bg_img = bg_img
    else:
        cropped_bg_img = prepro_backimage(back_img=bg_img)
        
    inv_mask = get_inverse_mask(mask)
    cropped_bg_img *= inv_mask
    #import matplotlib.pyplot as plt 
    #print("==============")
    #final = fg_img+cropped_bg_img
    #plt.imshow(np.concatenate((fg_img, cropped_bg_img,final), axis=1)) 
    #plt.axis('off')  # Turn off axis numbers
    #plt.show() 
     
    fg_img += cropped_bg_img
    return fg_img
 

def normalize_bb(bb, img_w, img_h):
    bb = copy.copy(bb)
    bb[0] /= img_w
    bb[1] /= img_h
    bb[2] /= img_w
    bb[3] /= img_h
    return bb


def normalize_keys_single(keys, w, h):
    """
    keys:BxNx2
    """
    keys[:, 0] /= w
    keys[:, 1] /= h
    return keys


def denormalize_keys_single(keys, w, h):
    """
    keys:BxNx2
    """
    keys[:, 0] *= w
    keys[:, 1] *= h
    return keys


def normalize_keys(keys, w, h):
    """
    keys:BxNx2
    """
    keys[:, :, 0] /= w
    keys[:, :, 1] /= h
    return keys


def normalize_keys_batch(keys, w, h):
    """
    keys:Bxn_viewsxNx2
    """
    keys[:, :, :, 0] /= w
    keys[:, :, :, 1] /= h
    return keys


def denormalize_keys(keys, w, h):
    """
    keys:BxNx2
    """
    keys[:, :, 0] *= w
    keys[:, :, 1] *= h
    return keys


def key_flipper_batch(w, h, keys, flip_h, flip_v):
    """
    args:
        w (int) : image width
        h (int) : image height
        keys (*,N,2): batch of 2D keypoints
        flip_h (bool): horizontal flipping
        flip_v (bool): vertical flipping
    return

    """
    if flip_h:
        keys[:, :, 1] = h-keys[:, :, 1]-1
    if flip_v:
        keys[:, :, 0] = w-keys[:, :, 0]-1
    return keys


def key_flipper(w, h, keys, flip_h, flip_v):
    """
    args:
        w (int) : image width
        h (int) : image height
        keys (N,2): batch of 2D keypoints
        flip_h (bool): horizontal flipping
        flip_v (bool): vertical flipping
    return

    """
    if flip_h:
        keys[:, 1] = h-keys[:, 1]-1
    if flip_v:
        keys[:, 0] = w-keys[:, 0]-1
    return keys


def face_key_flipper_wo_inversion(w, h, keys, face_lr_corresp_idx):
    keys_f = key_flipper(w=w,
                         h=h,
                         keys=copy.copy(keys),
                         flip_h=False,
                         flip_v=True)
    return keys_f[face_lr_corresp_idx]


####### right2left_inversion_pairs for 68 points in FLAME model ######
right2left_inversion_pairs = \
    [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (17, 26), (18, 25),
     (19, 24), (20, 23), (21, 22), (36, 45), (37,44), (38, 43), (39, 42), (40, 47), (41, 46),
     (32, 34), (31, 35), (50, 52), (49, 53), (61, 63), (48, 64), (67, 65), (59, 55), (58, 56), (60, 54)]


def get_id_array_from_pairs(pairs: list) -> np.array:
    """
    a function that returns the id array for the right2left inversion
    """
    id_array = np.arange(0, 68)
    for pair in pairs:
        id_array[pair[0]] = pair[1]
        id_array[pair[1]] = pair[0]
    return id_array


face_lr_corresp_idx = get_id_array_from_pairs(right2left_inversion_pairs)


def rotate_2d_data(data:np.ndarray, angle_degrees:float):
    """
    Rotates 2D data clockwise by a given angle in degrees with respect to the origin.

    Args:
        data (numpy array): 2D data points to be rotated. Should be in the shape (N, 2)
        angle_degrees (float): Angle in degrees to rotate the data

    Returns:
        numpy array: Rotated 2D data points (N,2)
    """
    # Convert angle from degrees to radians
    angle_radians = np.deg2rad(angle_degrees)

    # Extract x and y coordinates of data
    x = data[:, 0]
    y = data[:, 1]

    # Compute rotated x and y coordinates
    x_rotated = x * np.cos(angle_radians) + y * np.sin(angle_radians)
    y_rotated = -x * np.sin(angle_radians) + y * np.cos(angle_radians)

    # Combine rotated x and y coordinates into a single array
    rotated_data = np.column_stack((x_rotated, y_rotated))

    return rotated_data 