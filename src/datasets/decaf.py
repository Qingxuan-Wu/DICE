
import torch.utils.data
from random import shuffle
import copy,re
import os, sys
from PIL import Image
from torchvision import transforms
import cv2
from random import randint
import numpy as np
import random
import pytorch3d.transforms
import src.decaf.transformation_util as tu
import src.decaf.system_util as su
import src.decaf.FLAME_util as FLAME_util
import src.decaf.data_process_util as dp
from scipy import ndimage
import src.decaf.tracking_util as tracking_util
import mano
import scipy
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage



def atoi( text):
    """
    Code adapted from a link below:
    https://gist.github.com/hhanh/1947923
    received Mar 2019
    """
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    Code adapted from a link below:
    https://gist.github.com/hhanh/1947923
    received Mar 2019
    """
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [ atoi(c) for c in re.split('(\d+)', text)]

def params_parser(params):
        #print(params.keys())
        fids = list(params.keys())
        fids.sort()

        parsed_dict = {"head":{},"right_hand":{}}
        for i,fid in enumerate(fids):

          if i==0:
            parsed_dict['head']['transl'] = torch.FloatTensor(params[fid]['head']['transl']).unsqueeze(0)
            parsed_dict['head']['global_orient'] = torch.FloatTensor(params[fid]['head']['global_orient']).unsqueeze(0)
            parsed_dict['head']['shape_params'] = torch.FloatTensor(params[fid]['head']['shape_params']).unsqueeze(0)
            parsed_dict['head']['pose_params'] = torch.FloatTensor(params[fid]['head']['pose_params']).unsqueeze(0)
            parsed_dict['head']['expression_params'] = torch.FloatTensor(params[fid]['head']['expression_params']).unsqueeze(0)
            parsed_dict['right_hand']['betas'] = torch.FloatTensor(params[fid]['right_hand']['betas']).unsqueeze(0)
            parsed_dict['right_hand']['global_orient'] = torch.FloatTensor(params[fid]['right_hand']['global_orient']).unsqueeze(0)
            parsed_dict['right_hand']['hand_pose'] = torch.FloatTensor(params[fid]['right_hand']['hand_pose']).unsqueeze(0)
            parsed_dict['right_hand']['transl'] = torch.FloatTensor(params[fid]['right_hand']['transl']).unsqueeze(0)
          else:
            parsed_dict['head']['transl'] = torch.cat((parsed_dict['head']['transl'],torch.FloatTensor(params[fid]['head']['transl']).unsqueeze(0)),dim=0)
            parsed_dict['head']['global_orient'] = torch.cat((parsed_dict['head']['global_orient'],torch.FloatTensor(params[fid]['head']['global_orient']).unsqueeze(0)),dim=0)
            parsed_dict['head']['shape_params'] = torch.cat((parsed_dict['head']['shape_params'],torch.FloatTensor(params[fid]['head']['shape_params']).unsqueeze(0)),dim=0)
            parsed_dict['head']['pose_params'] = torch.cat((parsed_dict['head']['pose_params'],torch.FloatTensor(params[fid]['head']['pose_params']).unsqueeze(0)),dim=0)
            parsed_dict['head']['expression_params'] = torch.cat((parsed_dict['head']['expression_params'],torch.FloatTensor(params[fid]['head']['expression_params']).unsqueeze(0)),dim=0)
            parsed_dict['right_hand']['betas'] = torch.cat((parsed_dict['right_hand']['betas'],torch.FloatTensor(params[fid]['right_hand']['betas']).unsqueeze(0)),dim=0)
            parsed_dict['right_hand']['global_orient'] = torch.cat((parsed_dict['right_hand']['global_orient'],torch.FloatTensor(params[fid]['right_hand']['global_orient']).unsqueeze(0)),dim=0)
            parsed_dict['right_hand']['hand_pose'] = torch.cat((parsed_dict['right_hand']['hand_pose'],torch.FloatTensor(params[fid]['right_hand']['hand_pose']).unsqueeze(0)),dim=0)
            parsed_dict['right_hand']['transl'] = torch.cat((parsed_dict['right_hand']['transl'],torch.FloatTensor(params[fid]['right_hand']['transl']).unsqueeze(0)),dim=0)

        return parsed_dict

def make_decaf_dataset(args, is_train=True, cams=None):
    load_dict={  'single_img': 1, 'deforms': 1, 'head_cons': 1, 'head_cons_f': 1, 'rh_cons': 1,
                    'lh_cons': 1, 'head_dists': 0, 'head_dists_f': 0, 'rh_dists': 0,
                    'lh_dists': 0, 'head_poses': 1,   'head_img': 0,
                    'head_img_f': 0, 'rh_img': 0, 'lh_img': 0, 'rh_pose_cano': 1,
                    'head_keys2d': 1,'rh_keys_cano':1, 'rh_keys2d': 1, 'mediapipe': 0, "contact_matrix": 1, 'ref_vs': 1}
    
    print('Using: ', args.device)
    if is_train:
        valid_vid_ids = ['084', '100', '102', '108', '110', '111', '121', '122']
    else:
        valid_vid_ids = ['108']
    
    if cams:
        valid_vid_ids = cams

    train_subs = ['S2',"S4", "S5", "S7", "S8"]
    val_subs = ["S1", "S3", "S6"]
    dataset_path = args.data_path
    dataset_image_path =args.image_data_path
    single_image_path = args.single_image_path

    path_dict={"dataset":dataset_path, 
               "dataset_image":dataset_image_path,
               "single_image":single_image_path,
               "aug_path":dataset_path+"/assets/aug_back_ground/", }
    
    #MP = MeshProcessing(dataset_path+"/assets/default_mesh.ply")
    mode = "train" if is_train else "test"
    subs = train_subs if is_train else val_subs

    if is_train:
        args.back_aug = True
        aug_prob=0.8
    else:
        args.back_aug = False
        aug_prob=0.0

    min_scale, max_scale = args.scale_range.split(",")
    min_scale = float(min_scale)
    max_scale = float(max_scale)

    print("scale_range", args.scale_range)

    dataset = DecafDataset(base_path=dataset_path,
                                mode=mode,
                                win_size=args.win_size,
                                load_dict=load_dict,
                                subs=subs, 
                                n_pca=args.n_pca, 
                                img_wh=args.img_wh,
                                dyn_iter=args.dyn_iter,
                                deform_thresh=args.deform_thresh,
                                cam_space_deform=args.cam_space_deform,
                                back_aug=args.back_aug,
                                aug_prob=aug_prob,
                                valid_vid_ids=valid_vid_ids,
                                train_img_rotate_aug=args.train_imgrot_aug,
                                path_dict=path_dict,
                                multiscale=args.decaf_multiscale,
                                min_scale=min_scale,
                                max_scale=max_scale)
    return dataset

class DecafDataset(torch.utils.data.Dataset):
    def __init__(self,
                 base_path,
                 cam_space_deform,
                 load_dict,
                 dyn_iter,
                 deform_thresh,
                 img_wh,
                 path_dict,
                 mode,
                 key_flipping=True,
                 win_size=2,
                 min_deform_thresh=0.005,
                 subs=[],
                 normalize_image=True,
                 n_pca=45,
                 hand_dim=778,
                 head_dim=5023,
                 batch_size=1,
                 back_aug=1,
                 back_aug_prob=0.8,
                 rot_prob=0.3,
                 aug_prob=0.8,
                 n_batch_per_epoch=50,
                 valid_vid_ids=['84', '100', '102',
                                '108', '110', '111', '121', '122'],
                 train_img_rotate_aug=0,
                 multiscale=False,
                 min_scale = 0.75,
                 max_scale = 1.25):
        self.img_wh = img_wh
        self.key_flipping = key_flipping
        self.rot_prob = rot_prob
        self.dyn_iter = dyn_iter
        self.deform_thresh = deform_thresh
        self.hand_dim = hand_dim
        self.head_dim = head_dim
        self.normalize_image = normalize_image
        self.cam_space_deform = cam_space_deform
        self.n_batch_per_epoch = n_batch_per_epoch
        self.batch_size = batch_size
        self.min_deform_thresh = min_deform_thresh
        self.win_size = win_size
        self.mode = mode
        self.n_data = 0
        self.load_dict = load_dict
        self.base_path = base_path
        self.subs = subs
        self.back_aug = back_aug
        self.back_aug_prob = back_aug_prob
        self.train_img_rotate_aug = train_img_rotate_aug
        self.valid_vid_ids = valid_vid_ids
        self.n_pca = n_pca
        self.aug_path = path_dict['aug_path']
        self.path_dict = path_dict
        self.data_all_dic = {}
        self.ignore_idx = np.load(path_dict['dataset']+"/assets/FLAME_neck_idx.npy" )
        self.neck_idx = np.load(path_dict['dataset']+"/assets/neck_idx.npy")
        self.multiscale = multiscale

        self.lr_head_corresp_idx = np.load(path_dict['dataset']+"/assets/left_right_face_corresps.npy"  )
        self.aug_back_imgs = self.aug_back_img_loader(
            self.aug_path, w=1920, h=1080)
        self.flame_faces =FLAME_util.get_FLAME_faces(path_dict['dataset']+"/assets/generic_model.pkl")
        self.flame_model = FLAME_util.FLAME(
           path_dict['dataset']+"/assets/generic_model.pkl",
           path_dict['dataset']+"/assets/landmark_embedding.npy"
        )
        self.flame_model.eval()
        mano_model_path = path_dict['dataset']+"/assets/mano_v1_2/models/MANO_RIGHT.pkl"
        self.rh_model = mano.model.load(
              model_path=mano_model_path,
              is_right= True,
              num_pca_comps=self.n_pca,
              flat_hand_mean=True)
        self.rh_model.eval()
        self.cam_pos_dict = {}
        self.sub_start_index = []
        for sub in self.subs:
            self.sub_start_index.append(self.n_data)
            self.get_sub_data_dict(mode=mode, sub=sub)

        # print(self.n_data, "n_data")

        self.auglayer = dp.Augmenter(aug=self.back_aug, agu_prob=aug_prob)
        self.augfile_names = os.listdir(self.aug_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.transform_no_normalization = transforms.Compose([
            transforms.ToTensor()])

        self.scale_min = min_scale
        self.scale_max = max_scale

    def key_loader_for_all_cams(self,keys_path):
            key_dit = {}
            for cam in os.listdir(keys_path):
                if cam[:-4] in self.valid_vid_ids:
                    key_dit[cam[:-4]]=np.load(keys_path+cam)
            return key_dit

    def get_sub_data_dict(self, mode, sub):
        ###### obtain GT ######
        def_path =  self.base_path+"/"+mode+"/deformations/"+sub+"/deforms.npy"
        con_path =  self.base_path+"/"+mode+"/contacts/"+sub+"/"
        rh_bb_path  = self.base_path+"/"+mode+"/right_hand_bbs/"+sub+"/"
        head_bb_path = self.base_path+"/"+mode+"/head_bbs/"+sub+"/"
        shape_path = self.base_path+"/assets/accurate_shapes/" +  sub+".pkl"
        head_global_pos_path = \
            self.base_path+"/assets/transformation/"+mode +\
            "/"+sub + "_RT_FromCanToRef_wo_poseparams.npy"
        rh_canonical_pose_path =  \
            self.base_path+"/assets/transformation/"+mode +\
            "/"+sub + "_rh_pose_in_canonical.npy"
        rh_canonical_keys_path =  \
             self.base_path+"/assets/transformation/"+mode +\
            "/"+sub + "_rh_keys_tipShortened_in_canonical.npy"
        processed_data_base_path = \
            self.base_path+"processed/"+mode+"/"+sub+"_01/"
        params_path = self.base_path + '/'+mode+'/params/'+sub+'/params_smoothed.pkl'


        head_and_face_params = su.pickle_loader(params_path)
        parsed_params = params_parser(head_and_face_params)
        data_type = "_"+str(self.dyn_iter)+"_th_"+str(self.deform_thresh)

        shape_data = su.pickle_loader(shape_path)
        head_shape = shape_data['head']['shape_params']
        rh_shape = shape_data['right_hand']['betas']
        head_vs_rest = self.get_head_vs_rest(head_shape=head_shape)
        n_frames = len(np.load(def_path))

        cam_params= su.pickle_loader(self.base_path + '/'+mode+'/cameras/'+sub+'/cameras.pkl')

        if self.load_dict['deforms']:
            all_deformations = np.load(def_path)
            all_deformations = self.clean_deform_batch(
                deforms=all_deformations)
        else:
            all_deformations = np.zeros(n_frames)

        if self.load_dict['head_cons']:
            all_head_cons = np.load(con_path + "/contacts_head.npy")
            all_head_cons = self.clean_head_con_batch(all_head_cons)
        else:
            all_head_cons = np.zeros(n_frames)

        if self.load_dict['head_cons_f']:
            all_head_cons_f = np.load(con_path + "/contacts_head_flipped.npy")
            all_head_cons_f = self.clean_head_con_batch(all_head_cons_f)
        else:
            all_head_cons_f = np.zeros(n_frames)

        if self.load_dict['rh_cons']:
            all_rh_cons =  np.load(con_path + "/contacts_rh.npy")
        else:
            all_rh_cons = np.zeros(n_frames)

        if self.load_dict['lh_cons']:
            all_lh_cons =  np.load(con_path + "/contacts_lh.npy")
        else:
            all_lh_cons = np.zeros(n_frames)

        if self.load_dict['head_dists']:
            all_head_dists = \
                np.load(def_con_base_path+"/packed/distances"+data_type +
                        "/distances_head.npy")
            all_head_dists = np.sqrt(all_head_dists)
        else:
            all_head_dists = np.zeros(n_frames)

        if self.load_dict['head_dists_f']:
            ######## NOTE need to make sure below is correct #######
            all_head_dists_f = copy.copy(all_head_dists)[
                :, self.lr_head_corresp_idx]
        else:
            all_head_dists_f = np.zeros(n_frames)

        if self.load_dict['rh_dists']:
            all_rh_dists = \
                np.load(def_con_base_path+"/packed/distances"+data_type +
                        "/distances_rh.npy")
            all_rh_dists = np.sqrt(all_rh_dists)
        else:
            all_rh_dists = np.zeros(n_frames)

        if self.load_dict['lh_dists']:
            all_lh_dists = \
                np.load(def_con_base_path+"/packed/distances"+data_type +
                        "/distances_lh.npy")
            all_lh_dists = np.sqrt(all_lh_dists)
        else:
            all_lh_dists = np.zeros(n_frames)

        if self.load_dict['head_poses'] or self.load_dict['deforms']:
            all_head_poses = np.load(head_global_pos_path)
        else:
            all_head_poses = np.zeros(n_frames)

        # print(f"mode: {mode}, sub: {sub}: all_head_poses.shape: {all_head_poses.shape}, all_head_poses[0]: {all_head_poses[0]}")
        # input()

        if self.load_dict['rh_pose_cano']:
            all_rh_poses_cano = np.load(rh_canonical_pose_path)
        else:
            all_rh_poses_cano = np.zeros(n_frames)
        if self.load_dict['rh_keys_cano']:
            all_rh_keys_cano = np.load(rh_canonical_keys_path)
        else:
            all_rh_keys_cano = np.zeros(n_frames)

        rh_keys2d_path =  self.base_path+"/"+mode+"/right_hand_keys/"+sub+"/"
        head_keys2d_path =  self.base_path+"/"+mode+"/head_keys/"+sub+"/"
        rh_keys2d_dict = self.key_loader_for_all_cams(keys_path=rh_keys2d_path)
        head_keys2d_dict = self.key_loader_for_all_cams(keys_path=head_keys2d_path)


        #file_ids = [x[:-4] for x in os.listdir(
        #    def_con_base_path+"/deforms"+data_type+"/")]

        file_ids =[x.zfill(5) for x in np.arange(n_frames).astype(str)]
        file_ids.sort(key=natural_keys)

        print(all_deformations.shape, all_head_cons.shape,
              all_rh_cons.shape,   all_head_cons_f.shape, )
        n_frames = len(all_deformations)
        #valid_labels = dp.get_valid_label(
        #    data=invalid_label_dict[mode][sub],
        #    n_frames=n_frames)
        cut_ids = su.pickle_loader(self.base_path+"/"+mode+"/cutIDs/"+sub+".pkl")
       # print(cut_ids)

        ###### obtain input paths  ######
        #processed_data_base_path = \
        #    self.base_path+"processed/"+mode+"/"+sub+"_01/"
        vid_names = os.listdir(self.base_path+mode+"/videos/"+sub+"/")
        vid_names.sort(key=natural_keys)
        vid_names = [x for i, x in enumerate(
            vid_names) if x[:-4] in self.valid_vid_ids]

        print(sub, vid_names)

        #bb_hand_base_path = self.base_path+"/"+mode+"/right_hand_bbs/"
        #bb_head_base_path = self.base_path+"/"+mode+"/head_bbs/"
        # \cropped_head_images\S1\084
        #path_dict['dataset_image']+"/"+mode+"/"
        img_base_path = self.path_dict['dataset_image']+"/"+mode+"/"
        head_img_base_path = img_base_path+"/cropped_head_images/" +sub+"/"
        rh_img_base_path = img_base_path+"/cropped_rh_images/" +sub+"/"
        head_mask_base_path = img_base_path+"/cropped_head_masks/" +sub+"/"
        rh_mask_base_path = img_base_path+"/cropped_rh_masks/" +sub+"/"

        single_base_path = self.path_dict['single_image']+"/"+mode+"/"
        single_bb_base_path = single_base_path + "/cropped_bbs/" + sub + "/"
        single_img_base_path = single_base_path + "/cropped_images/" + sub + "/"
        single_img_with_bg_base_path = single_base_path + "/cropped_images_with_bg/" + sub + "/"
        single_mask_base_path = single_base_path + "/cropped_masks/" + sub + "/"

        #mask_path = processed_data_base_path+"/masks/"
        self.n_data += len(file_ids)*len(vid_names)

        rh_bbs = {}
        head_bbs = {}
        for camid in self.valid_vid_ids:
            if sub == "S5" and camid == "111":
                continue
            head_bbs[camid] = np.load(head_bb_path + camid + ".npy").astype(int)
            rh_bbs[camid] = np.load(rh_bb_path + camid + ".npy").astype(int)

        single_bbs = {}
        for camid in self.valid_vid_ids:
            if sub == "S5" and camid == "111":
                continue
            single_bbs[camid] = np.load(single_bb_base_path + camid + ".npy").astype(int)

        #path_dict['dataset']+"/assets/left_right_face_corresps.npy"
        sub_dic = {"deforms": all_deformations, "head_cons": all_head_cons,
                   "head_cons_f": all_head_cons_f, "lh_cons": all_lh_cons,
                   "rh_cons": all_rh_cons,
                   "head_dists": all_head_dists,
                   "head_dists_f": all_head_dists_f,
                    "rh_dists": all_rh_dists,
                    "lh_dists": all_lh_dists,
                   "head_vs_rest": head_vs_rest, "cut_ids": cut_ids,
                   "vid_names": vid_names, "rh_keys2d": rh_keys2d_dict,
                   "head_keys2d": head_keys2d_dict,
#                   "bb_hand_base_path": bb_hand_base_path,
                   "head_img_base_path": head_img_base_path,#  "mask_path": mask_path,
                    "rh_img_base_path": rh_img_base_path,
                    "single_img_base_path": single_img_base_path,
                    "single_img_with_bg_base_path": single_img_with_bg_base_path,
                    "head_mask_base_path": head_mask_base_path,
                    "rh_mask_base_path": rh_mask_base_path,
                    "single_mask_base_path": single_mask_base_path,

            #       "bb_head_base_path": bb_head_base_path,
                   "head_poses": all_head_poses,
                   "rh_poses_cano": all_rh_poses_cano,
                   "rh_keys_cano": all_rh_keys_cano,  "file_ids": file_ids,
                   "head_shape": head_shape, "rh_shape": rh_shape,
                   "head_and_face_params": parsed_params,
                   "cam_params": cam_params,
                   "rh_bbs": rh_bbs, "head_bbs": head_bbs, "single_bbs": single_bbs,}



        # self.intrinsic_dict = {}
        self.data_all_dic[sub] = sub_dic
        if self.cam_space_deform:

            calibrations = su.pickle_loader(self.base_path+"/"+mode+"/cameras/"+sub+"/cameras.pkl")
            self.cam_pos_dict[sub] = {}
            #MCP = MetaShapeCameraPreProcessing(
            #    xml_path=self.base_path+"/processed/calibrations/"+mode+"/"+sub
            #    + "/cameras.xml", image_size=(1920, 1080))
            for camid in self.valid_vid_ids:
                if sub == "S5" and camid == "111":
                    continue
                self.cam_pos_dict[sub][camid.zfill(
                    3)] =np.array(calibrations[camid.zfill(3)]["extrinsic"])

            # for camid in self.valid_vid_ids:
            #     self.cam_pos_dict[sub][camid.zfill(
            #         3)] =np.array(calibrations[camid.zfill(3)]["extrinsic"])

                #MCP.cam_inv_dict[camid.zfill(3)]
        return

    def get_bb_head_hand_dict(self, processed_data_base_path, vid_names):
        all_head_bb_dict = {}
        all_rh_bb_dict = {}
        for vid_name in vid_names:
            head_folder_name = \
                processed_data_base_path+"/bb_head_packed/"+vid_name
            hand_folder_name = \
                processed_data_base_path+"/bb_hand_packed/"+vid_name

            all_head_bb_dict[vid_name] = \
                np.load(head_folder_name+"/head_bb.npy")
            all_rh_bb_dict[vid_name] = \
                np.load(hand_folder_name+"/rh_bb.npy")
        return all_head_bb_dict, all_rh_bb_dict

    def get_head_vs_rest(self, head_shape):
        [head_vs_model_rest, _] = \
            FLAME_util.flame_forwarding(
            flame_model=self.flame_model,
            head_shape_params=su.np2tor(head_shape).view(1, -1),
            head_expression_params=torch.zeros(1, 50),
            head_pose_params=torch.zeros(1, 6),
            head_rotation=torch.zeros(1, 3),
            head_transl=torch.zeros(1, 3),
            head_scale_params=su.np2tor(np.array([1.0])),
        )
        head_vs_model_rest = head_vs_model_rest[0].numpy()
        return head_vs_model_rest

    def get_head_vs(self, params, sub_id, cam_id, frame_id):
        # for use in getitem
        cam_params = self.get_cam_params(sub_id, cam_id)
        # input(f"cam_params: {cam_params}")
        Ps= np.matmul(cam_params['intrinsic'],
                  cam_params['extrinsic'][:-1])
        device = next(self.flame_model.parameters()).device
        Ps= torch.FloatTensor(Ps).to(device)
        Ps_batch = Ps.clone().view(1,  1, 3, 4).expand(1, -1, -1, -1)
        [head_vs, landmarks3d, head_keys_proj, head_vs_proj] = \
            FLAME_util.flame_forwarding(
            flame_model=self.flame_model,
            head_shape_params=params['head']['shape_params'][frame_id].unsqueeze(0),
            head_expression_params=params['head']['expression_params'][frame_id].unsqueeze(0),
            head_pose_params=params['head']['pose_params'][frame_id].unsqueeze(0),
            head_rotation= params['head']['global_orient'][frame_id].unsqueeze(0),
            head_transl= params['head']['transl'][frame_id].unsqueeze(0),
            head_scale_params=  torch.ones((1,1)),
            Ps=Ps_batch,
            return2d=True,
            img_size=(1, 1),
            device=device,
            return_2d_verts=True
        )

        return head_vs.squeeze(), landmarks3d.squeeze(), head_keys_proj.squeeze(), head_vs_proj.squeeze()

    def get_rh_vs(self, params, sub_id, cam_id, frame_id):
        # for use in getitem
        cam_params = self.get_cam_params(sub_id, cam_id)
        # input(f"cam_params: {cam_params}")
        Ps= np.matmul(cam_params['intrinsic'],
                  cam_params['extrinsic'][:-1])
        device = next(self.rh_model.parameters()).device
        Ps= torch.FloatTensor(Ps).to(device)
        Ps_batch = Ps.clone().view(1,  1, 3, 4).expand(1, -1, -1, -1)

        [rh_vs, rh_keys_3ds, rh_keys_proj, rh_vs_proj] = tracking_util.mano_forwarding(
              h_model=self.rh_model,
              betas=params['right_hand']['betas'][frame_id].unsqueeze(0),
              transl= params['right_hand']['transl'][frame_id].unsqueeze(0),
              rot= params['right_hand']['global_orient'][frame_id].unsqueeze(0),
              pose= params['right_hand']['hand_pose'][frame_id].unsqueeze(0),
              Ps=Ps_batch,
              return_2d=True,
              img_size=(1, 1),
              return_2d_verts=True
          )
        return rh_vs.squeeze(), rh_keys_3ds.squeeze(), rh_keys_proj.squeeze(), rh_vs_proj.squeeze()

    def get_head_ref_vs(self):
        [head_vs_model_rest, _] = \
            FLAME_util.flame_forwarding(
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
            tracking_util.mano_forwarding(
            h_model=self.rh_model,
            betas=torch.zeros(1, 10),
            transl=torch.zeros(1, 3),
            rot=torch.zeros(1, 3),
            pose=torch.zeros(1, 45),
            return_2d=False,
        )
        rh_vs_model_rest = rh_vs_model_rest[0].numpy()
        return rh_vs_model_rest


    def aug_back_img_loader(self, path, w, h):
        return [cv2.resize(cv2.imread(path+x), (w, h)) for x in os.listdir(path)]

    def get_img_ids_dict(self, path):
        all_files_dict = {}
        for folder in self.folder_names:
            img_names = os.listdir(path+'/'+folder)
            img_names.sort()
            img_names = img_names[:len(self.deform_names)]
            tmp_files = []
            [tmp_files.append(path+'/'+folder+'/'+img_name)
             for img_name in img_names]
            all_files_dict[folder] = tmp_files
        return all_files_dict

    def get_valid_id_dict(self, path):
        all_files_dict = {}
        ori_labels = np.load(path)
        for folder in self.folder_names:
            bbs = self.bbs_dict[folder]
            bb_success_labels = np.sum(bbs, axis=1)
            bb_success_labels[bb_success_labels != -4] = 1
            bb_success_labels[bb_success_labels == -4] = 0
            tmp_labels = bb_success_labels*copy.copy(ori_labels)
            self.n_data += int(np.sum(bb_success_labels))
            all_files_dict[folder] = tmp_labels
        return all_files_dict

    def get_image_name_dict(self):
        self.img_folder_names
        return 0

    def get_bbs_dict(self, path):
        all_files_dict = {}
        for folder in self.folder_names:
            tmp_files = np.load(path+'/'+folder+'.npy')
            tmp_files = tmp_files[:len(self.deform_names)]
            all_files_dict[folder] = tmp_files
        return all_files_dict

    def __len__(self):
        # 100*self.batch_size #10#self.n_data#500# self.n_data
        return self.n_data
        # return self.n_data
        # return self.n_batch_per_epoch*self.batch_size

    def process_imgs(self, image_paths):
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            image = image.resize((224, 224), Image.ANTIALIAS)
            tensor_image = self.transform(image)

            c, w, h = tensor_image.shape
            if i == 0:
                img_seqs = tensor_image.clone().view(1, c, w, h)
            else:
                img_seqs = torch.cat(
                    (img_seqs, tensor_image.view(1, c, w, h)), 0)

        return img_seqs

    def clean_deform(self, deforms):
        """
        Input: deforms (5023x3)
        output: valid deforms (5023x3)
        """
        deforms[self.ignore_idx] = 0
        return deforms

    def clean_deform_batch(self, deforms):
        """
        Input: deforms (*,5023,3)
        output: valid deforms (*,5023,3)
        """
        deforms[:, self.ignore_idx] = 0
        return deforms

    def clean_head_con_batch(self, cons):
        """
        Input: deforms (*,5023,3)
        output: valid deforms (*,5023,3)
        """
        cons[:, self.neck_idx] = 0
        return cons

    def con_idx_to_label(self, con_idx, dim):
        labels = np.zeros(dim)
        labels[con_idx] = 1
        return labels

    def get_head_rot6d_in_camframe(self, target_sub, target_id, target_cam_id,
                                   return_flipped=False):
        head_pose = copy.copy(
            self.data_all_dic[target_sub]['head_poses'][target_id])
        head_rotmat = head_pose[:3, :3]  # .reshape(1,3,3)

        cam_pose = copy.copy(self.cam_pos_dict[target_sub][target_cam_id])
        cam_rotmat = cam_pose[:3, :3]  # .reshape(1,3,3)

        rotmat = np.dot(cam_rotmat, head_rotmat).reshape(1, 3, 3)

        rot6ds = pytorch3d.transforms.matrix_to_rotation_6d(
            torch.FloatTensor(rotmat)).numpy()[0]

        if return_flipped:
            head_vs_model_rest = copy.copy(
                self.data_all_dic[target_sub]['head_vs_rest'])
            head_rest_vs_in_cam = tu.apply_transform_np(
                head_vs_model_rest,
                RT=np.dot(cam_pose, head_pose))
            head_rest_vs_f_in_cam = copy.copy(head_rest_vs_in_cam)
            head_rest_vs_f_in_cam[:, 0] *= -1
            head_rest_vs_f_in_cam = head_rest_vs_f_in_cam[self.lr_head_corresp_idx]
            head_rotmat_f, head_transl_f = tu.procrastes(
                head_vs_model_rest, head_rest_vs_f_in_cam)

            rot6ds_f = pytorch3d.transforms.matrix_to_rotation_6d(
                torch.FloatTensor(head_rotmat_f).view(1, 3, 3)).numpy()[0]
        else:
            rot6ds_f = None

        return rot6ds, rot6ds_f,head_rest_vs_in_cam,head_rest_vs_f_in_cam

    def get_transform_from_canonical2camera(self,
                                            target_sub,
                                            target_id,
                                            target_cam_id,
                                            return_flipped=False):
        head_RT = copy.copy(
            self.data_all_dic[target_sub]['head_poses'][target_id])

        cam_RT = copy.copy(self.cam_pos_dict[target_sub][target_cam_id])

        can2cam_RT = np.dot(cam_RT, head_RT).reshape(4, 4)

        head_rest_vs_in_canonical = copy.copy(
            self.data_all_dic[target_sub]['head_vs_rest'])
        head_rest_vs_in_camera = tu.apply_transform_np(
            head_rest_vs_in_canonical,
            RT=can2cam_RT)
        head_rest_vs_f_in_camera = copy.copy(head_rest_vs_in_camera)
        head_rest_vs_f_in_camera[:, 0] *= -1
        head_rest_vs_f_in_camera = head_rest_vs_f_in_camera[self.lr_head_corresp_idx]
        head_rotmat_f, head_transl_f = tu.procrastes(
            head_rest_vs_in_camera, head_rest_vs_f_in_camera)
        flipping_RT = tu.convert_R_T_to_RT4x4_np(head_rotmat_f, head_transl_f)
        can2cam_RT_f = np.dot(flipping_RT, can2cam_RT)
        return can2cam_RT, can2cam_RT_f

    def convert_canonical_deforms_into_cam_space(self,
                                                 target_frame,
                                                 target_sub,
                                                 target_cam_id,
                                                 ):
        """
        cannonical -> 3D space with the given poses -> camera frame
        """

        can_deforms = self.data_all_dic[target_sub]["deforms"][target_frame]

        ##### cannonical-> posed #####
        head_pose = copy.copy(
            self.data_all_dic[target_sub]['head_poses'][target_frame])
        head_pose[:3, -1] = 0
        posed_deforms = tu.apply_transform_np(data=can_deforms,
                                              RT=head_pose)
        ##### posed -> cam space ###

        cam_transform = copy.copy(self.cam_pos_dict[target_sub][target_cam_id])
        cam_transform[:3, -1] = 0
        cam_space_deforms = tu.apply_transform_np(data=posed_deforms,
                                                  RT=cam_transform)

        return cam_space_deforms

    def image_pre_process_cv2(self, image, size=(224, 224)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        #image = image.resize(size, Image.ANTIALIAS)
        if self.normalize_image:
            tensor_image = self.transform(image)
        else:
            tensor_image = self.transform_no_normalization(image)
        #tensor_image = self.transform(image)
        return tensor_image

    def get_lh_keys_cano(self, rh_keys_cano, target_sub, target_id, target_cam_id):
        can2cam_RT, can2cam_RT_f = self.get_transform_from_canonical2camera(
            target_sub=target_sub,
            target_id=target_id,  # int(seq_file_ids[self.win_size]),
            target_cam_id=target_cam_id)  # target_video_name[14:17] )

        rh_keys_camera = tu.apply_transform_np(data=rh_keys_cano,
                                               RT=can2cam_RT)
        lh_keys_camera = copy.copy(rh_keys_camera)
        lh_keys_camera[:, 0] *= -1
        lh_keys_cano = tu.apply_transform_np(data=lh_keys_camera,
                                             RT=np.linalg.inv(can2cam_RT_f))
        return lh_keys_cano

    def check_validty(self,seq_ids, cut_ids):
        # print("seq_ids",seq_ids)
        # print("cut_ids",cut_ids)
        seq_ids = np.array(seq_ids).astype(int)
        for id_pair in cut_ids:

            if id_pair[0] in seq_ids and id_pair[1] in seq_ids:
                return False

        return True

    def get_cam_params(self, sub_id, cam_id):
        cam_params = self.data_all_dic[sub_id]["cam_params"][cam_id]
        return cam_params

    def crop_rh_verts(self, sub_id, cam_id, target_frame, rh_vs):
        vs = copy.deepcopy(rh_vs)
        bbox = self.data_all_dic[sub_id]["rh_bbs"][cam_id][target_frame]
        if bbox.min() < 0 or bbox[2] > 1920 or bbox[3] > 1080:
            bbox = [0, 0, 1920, 1080]
        vs[:, 0] =  (vs[:, 0] - bbox[0]) / (bbox[2] - bbox[0])
        vs[:, 1] =  (vs[:, 1] - bbox[1]) / (bbox[3] - bbox[1])
        return vs

    def crop_head_verts(self, sub_id, cam_id, target_frame, head_vs):
        vs = copy.deepcopy(head_vs)
        bbox = self.data_all_dic[sub_id]["head_bbs"][cam_id][target_frame]
        if bbox.min() < 0 or bbox[2] > 1920 or bbox[3] > 1080:
            bbox = [0, 0, 1920, 1080]
        vs[:, 0] =  (vs[:, 0] - bbox[0]) / (bbox[2] - bbox[0])
        vs[:, 1] =  (vs[:, 1] - bbox[1]) / (bbox[3] - bbox[1])
        return vs

    def crop_single_verts(self, sub_id, cam_id, target_frame, single_vs):
        vs = copy.deepcopy(single_vs)
        bbox = self.data_all_dic[sub_id]["single_bbs"][cam_id][target_frame]
        vs[:, 0] =  (single_vs[:, 0] - bbox[0]) / (bbox[2] - bbox[0])
        vs[:, 1] =  (single_vs[:, 1] - bbox[1]) / (bbox[3] - bbox[1])
        return vs

    def __getitem__(self, idx):

        count = 0
        if self.back_aug:
            aug_file_id = random.randint(0, len(self.augfile_names)-1)
            back = cv2.imread(self.aug_path+self.augfile_names[aug_file_id])
        # start=time.time()

        sub_index = -1
        for i in range(len(self.subs)):
            if idx < self.sub_start_index[i]:
                break
            sub_index += 1

        target_sub = self.subs[sub_index]
        sub_start = self.sub_start_index[sub_index]

        n_frames = len(self.data_all_dic[target_sub]["file_ids"])
        video_select_id = (idx - sub_start) // n_frames
        random_frame_id = (idx - sub_start) % n_frames
        # print("idx", idx)
        # print("self.sub_start_index", self.sub_start_index)
        # pfrrint(n_frames, "n_frames")
        # print(video_select_id, "video_select_id")
        # print(random_frame_id, "random_frame_id")
        # print(f"idx: {idx}, sub_index: {sub_index}, sub_start: {sub_start}, video_select_id: {video_select_id}, random_frame_id: {random_frame_id}")

        # while (1):
        # target_sub = self.subs[randint(0, len(self.subs)-1)]

        # video_select_id = randint(
        #     0, len(self.data_all_dic[target_sub]["vid_names"]) - 1)
        target_video_name = \
            self.data_all_dic[target_sub]["vid_names"][video_select_id]

        # n_frames = len(self.data_all_dic[target_sub]["file_ids"])
        # random_frame_id = randint(self.win_size, n_frames-self.win_size-1)

        startid = random_frame_id - self.win_size
        endid = random_frame_id+self.win_size+1
        seq_file_ids = self.data_all_dic[target_sub]["file_ids"][startid:endid]

        target_frame = int(seq_file_ids[self.win_size])

        cut_ids =  self.data_all_dic[target_sub]["cut_ids"]

        # Get hand vertices

        cam_params = self.get_cam_params(target_sub, target_video_name[:-4])

        assert self.win_size == 0
        hand_and_face_params = self.data_all_dic[target_sub]["head_and_face_params"]
        rh_vs, rh_keys_3ds, rh_keys_proj, rh_vs_proj = self.get_rh_vs(hand_and_face_params, target_sub, target_video_name[:-4], target_frame)
        head_vs, head_keys_3ds, head_keys_proj, head_vs_proj = self.get_head_vs(hand_and_face_params, target_sub, target_video_name[:-4], target_frame)

        rh_ref_vs = self.get_rh_ref_vs()
        head_ref_vs = self.get_head_ref_vs()

        rh_keys_2d = self.data_all_dic[target_sub]["rh_keys2d"][target_video_name[:-4]][target_frame][:, :2]
        head_keys_2d = self.data_all_dic[target_sub]["head_keys2d"][target_video_name[:-4]][target_frame][:, :2]

        rh_keys_proj_single = self.crop_single_verts(target_sub, target_video_name[:-4], target_frame, rh_keys_2d) # using gt 2d instead of reprojected
        head_keys_proj_single = self.crop_single_verts(target_sub, target_video_name[:-4], target_frame, head_keys_2d) # using gt 2d instead of reprojected
        rh_keys_proj_single = torch.from_numpy(rh_keys_proj_single)
        head_keys_proj_single = torch.from_numpy(head_keys_proj_single)
        # rh_keys_proj_single = self.crop_single_verts(target_sub, target_video_name[:-4], target_frame, rh_keys_proj)
        # head_keys_proj_single = self.crop_single_verts(target_sub, target_video_name[:-4], target_frame, head_keys_proj)
        rh_vs_proj_single = self.crop_single_verts(target_sub, target_video_name[:-4], target_frame, rh_vs_proj)
        head_vs_proj_single = self.crop_single_verts(target_sub, target_video_name[:-4], target_frame, head_vs_proj)


        rh_keys_proj = self.crop_rh_verts(target_sub, target_video_name[:-4], target_frame, rh_keys_proj)
        head_keys_proj = self.crop_head_verts(target_sub, target_video_name[:-4], target_frame, head_keys_proj)
        rh_vs_proj = self.crop_rh_verts(target_sub, target_video_name[:-4], target_frame, rh_vs_proj)
        head_vs_proj = self.crop_head_verts(target_sub, target_video_name[:-4], target_frame, head_vs_proj)


        rh_betas = hand_and_face_params['right_hand']['betas'][target_frame]
        rh_transl = hand_and_face_params['right_hand']['transl'][target_frame]
        rh_rot = hand_and_face_params['right_hand']['global_orient'][target_frame]
        rh_pose = hand_and_face_params['right_hand']['hand_pose'][target_frame]

        head_shape_params = hand_and_face_params['head']['shape_params'][target_frame]
        head_expression_params = hand_and_face_params['head']['expression_params'][target_frame]
        head_pose_params = hand_and_face_params['head']['pose_params'][target_frame]
        head_rotation = hand_and_face_params['head']['global_orient'][target_frame]
        head_transl = hand_and_face_params['head']['transl'][target_frame]


        RT_np = np.array(self.cam_pos_dict[target_sub][target_video_name[:-4]])

        RT_np[:, -1] = np.array([0,0,0,1]) # set T=0 to normalize

        head_vs_in_cam = tu.apply_transform_np(head_vs,RT=RT_np)
        rh_vs_in_cam = tu.apply_transform_np(rh_vs,RT=RT_np)
        head_keys_in_cam = tu.apply_transform_np(head_keys_3ds,RT=RT_np)
        rh_keys_in_cam = tu.apply_transform_np(rh_keys_3ds,RT=RT_np)

        head_vs_in_cam_f = copy.copy(head_vs_in_cam)
        head_vs_in_cam_f[:, 0] *= -1
        head_vs_in_cam_f = head_vs_in_cam_f[self.lr_head_corresp_idx]

        rh_vs_in_cam_f = copy.copy(rh_vs_in_cam)
        rh_vs_in_cam_f[:, 0] *= -1



        # if self.check_validty(seq_file_ids,cut_ids):
        #     print("valid")
        #     break
        # print("invalid")
        # return
        # if count >= 1000:
        #     break
        # count += 1

        if self.mode == "train" and self.train_img_rotate_aug and random.uniform(0, 1) <= self.rot_prob:
            # angle = np.random.randint(40)-20
            angle = 0
        else:
            angle = 0

        if angle != 0:
            raise ValueError("non-zero z-rotation angle. there is bugs with z-rotation, don't use it.")

        tmp_all_imgs = []
        for i in range(len(seq_file_ids)):
            assert len(seq_file_ids) == 1
            if self.load_dict['mediapipe']:
                mediapipe_hand_kp = np.load(self.base_path + "/" + self.mode + "/hand_kps/" + target_sub + "/" + target_video_name[:-4] + "/" + seq_file_ids[i] + ".npy")
                mediapipe_face_kp = np.load(self.base_path + "/" + self.mode + "/face_kps/" + target_sub + "/" + target_video_name[:-4] + "/" + seq_file_ids[i] + ".npy")

                # print(f"sub: {target_sub}, cam: {target_video_name[:-4]}", "mediapipe_face_kp.shape", mediapipe_face_kp.shape)
                # print("mediapipe_hand_kp.shape", mediapipe_hand_kp.shape)

                if mediapipe_hand_kp.shape[0] == 0:
                    mediapipe_hand_kp = np.zeros((21, 3))
                if mediapipe_face_kp.shape[0] == 0:
                    mediapipe_face_kp = np.zeros((478, 3))
            else:
                mediapipe_hand_kp = np.zeros((21, 3))
                mediapipe_face_kp = np.zeros((478, 3))

            #### load cropped RGB images, maks, bbs ########
            if self.load_dict['contact_matrix']:
                contact_matrix = scipy.sparse.load_npz(self.base_path + "/" + self.mode + "/contact_matrices/" + target_sub + "/" + seq_file_ids[i] + ".npz").toarray()

            if self.load_dict['single_img']:
                single_image = cv2.imread(
                    self.data_all_dic[target_sub]["single_img_base_path"] +
                    target_video_name[:-4]+"/"+seq_file_ids[i]+".jpg")
                single_image = ndimage.rotate(single_image, -angle)
                single_image_with_bg = cv2.imread(
                    self.data_all_dic[target_sub]["single_img_with_bg_base_path"] +
                    target_video_name[:-4]+"/"+seq_file_ids[i]+".jpg")
                single_image_with_bg = ndimage.rotate(single_image_with_bg, -angle)

            if self.load_dict['head_img']:
                head_image = cv2.imread(
                    self.data_all_dic[target_sub]["head_img_base_path"] +
                    target_video_name[:-4]+"/"+seq_file_ids[i]+".jpg")

                head_image = ndimage.rotate(head_image, -angle)
            else:
                head_image = np.zeros([225, 225, 3]).astype(np.uint8)

            if self.load_dict['rh_img']:
                rh_image = cv2.imread(
                    self.data_all_dic[target_sub]["rh_img_base_path"] +
                    target_video_name[:-4]+"/"+seq_file_ids[i]+".jpg")
                rh_image = ndimage.rotate(rh_image, -angle)
            else:
                rh_image = np.zeros([225, 225, 3]).astype(np.uint8)

            if self.load_dict['head_img']:
                #head_image_f = cv2.imread(
                #    self.data_all_dic[target_sub]["img_path"] +
                #    target_video_name[:-4]+"/head_flipped/"+seq_file_ids[i]+".jpg")
                head_image_f = cv2.flip(head_image, 1)
                head_image_f = ndimage.rotate(head_image_f, -angle)
            else:
                # np.zeros([225, 225, 3]).astype(np.uint8)
                head_image_f = cv2.flip(head_image, 1)

            if self.load_dict['lh_img']:
                #lh_image = cv2.imread(
                #    self.data_all_dic[target_sub]["img_path"] +
                #    target_video_name[:-4]+"/left_hand/"+seq_file_ids[i]+".jpg")
                lh_image = cv2.flip(rh_image, 1)
                lh_image = ndimage.rotate(lh_image, -angle)
            else:
                #lh_image = np.zeros([225, 225, 3]).astype(np.uint8)
                lh_image = cv2.flip(rh_image, 1)


            single_image = cv2.resize(
                single_image, (224, 224), interpolation=cv2.INTER_AREA)
            single_image_with_bg = cv2.resize(
                single_image_with_bg, (224, 224), interpolation=cv2.INTER_AREA)
            head_image = cv2.resize(
                head_image, (224, 224), interpolation=cv2.INTER_AREA)
            rh_image = cv2.resize(rh_image, (224, 224),
                                  interpolation=cv2.INTER_AREA)
            head_image_f = cv2.resize(
                head_image_f, (224, 224), interpolation=cv2.INTER_AREA)
            lh_image = cv2.resize(lh_image, (224, 224),
                                  interpolation=cv2.INTER_AREA)

            # print(single_image.max(), single_image.min(), single_image.mean(), single_image.std(), "single_image")
            # print(single_image_with_bg.max(), single_image_with_bg.min(), single_image_with_bg.mean(), single_image_with_bg.std(), "single_image_with_bg")


            if self.back_aug and random.uniform(0, 1) <= self.back_aug_prob:

                head_mask = cv2.imread(
                    self.data_all_dic[target_sub]["head_mask_base_path"] +
                    target_video_name[:-4]+"/"+seq_file_ids[i]+".png")
                rh_mask = cv2.imread(
                    self.data_all_dic[target_sub]["rh_mask_base_path"]+
                    target_video_name[:-4]+"/"+seq_file_ids[i]+".png")
                single_mask = cv2.imread(
                    self.data_all_dic[target_sub]["single_mask_base_path"]+
                    target_video_name[:-4]+"/"+seq_file_ids[i]+".png")


                lh_mask = cv2.flip(rh_mask, 1)
                head_mask_f = cv2.flip(head_mask, 1)

                head_mask = ndimage.rotate(head_mask, -angle)
                rh_mask = ndimage.rotate(rh_mask, -angle)
                head_mask_f = ndimage.rotate(head_mask_f, -angle)
                lh_mask = ndimage.rotate(lh_mask, -angle)
                single_mask = ndimage.rotate(single_mask, -angle)

                head_mask = cv2.resize(
                    head_mask, (224, 224), interpolation=cv2.INTER_AREA)
                rh_mask = cv2.resize(rh_mask, (224, 224),
                                     interpolation=cv2.INTER_AREA)
                lh_mask = cv2.resize(lh_mask, (224, 224),
                                     interpolation=cv2.INTER_AREA)
                head_mask_f = cv2.resize(
                    head_mask_f, (224, 224), interpolation=cv2.INTER_AREA)
                single_mask = cv2.resize(single_mask, (224, 224),
                                        interpolation=cv2.INTER_AREA)

                back = dp.prepro_backimage(back_img=back)
                single_image = dp.get_image_with_bg(
                    fg_img=single_image, bg_img=back, mask=single_mask,is_cropped=True)
                head_image = dp.get_image_with_bg(
                    fg_img=head_image, bg_img=back, mask=head_mask,is_cropped=True)
                head_image_f = dp.get_image_with_bg(
                    fg_img=head_image_f, bg_img=back, mask=head_mask_f,is_cropped=True)
                rh_image = dp.get_image_with_bg(
                    fg_img=rh_image, bg_img=back, mask=rh_mask,is_cropped=True)
                lh_image = dp.get_image_with_bg(
                    fg_img=lh_image, bg_img=back, mask=lh_mask,is_cropped=True)

                #cv2.imshow('img', head_mask)
                #cv2.imshow('ba', head_image)
                #cv2.imshow('ba2', rh_image)
                #cv2.imshow('ba3', lh_image)
                #cv2.imshow('ba4', head_image_f)
                # time.sleep(10)
                # cv2.waitKey(0)
                # sys.exit()
                # cv2.waitKey(0)
            #w,h,c = head_image

            tmp_all_imgs.append(head_image)
            tmp_all_imgs.append(head_image_f)
            tmp_all_imgs.append(rh_image)
            tmp_all_imgs.append(lh_image)
            tmp_all_imgs.append(single_image)
            tmp_all_imgs.append(single_image_with_bg)


        tmp_all_imgs = self.auglayer(tmp_all_imgs)
        all_n = len(tmp_all_imgs)
        n_frames = self.win_size*2+1

        head_img_idx = np.arange(0, all_n, all_n/n_frames).astype(int)
        head_image_f_idx = np.arange(1, all_n, all_n/n_frames).astype(int)
        rh_image_idx = np.arange(2, all_n, all_n/n_frames).astype(int)
        lh_image_idx = np.arange(3, all_n, all_n/n_frames).astype(int)
        single_image_idx = np.arange(4, all_n, all_n/n_frames).astype(int)
        single_image_with_bg_idx = np.arange(5, all_n, all_n/n_frames).astype(int)


        head_img_seqs = tmp_all_imgs[head_img_idx]
        head_img_f_seqs = tmp_all_imgs[head_image_f_idx]
        rh_img_seqs = tmp_all_imgs[rh_image_idx]
        lh_img_seqs = tmp_all_imgs[lh_image_idx]
        single_img_seqs = tmp_all_imgs[single_image_idx].squeeze()
        single_image_with_bg = tmp_all_imgs[single_image_with_bg_idx].squeeze()

        # print(single_img_seqs.shape, 'single_img_seqs')
        # print(single_image_with_bg.shape, 'single_image_with_bg')
        if self.mode == "test" or random.uniform(0, 1) <= 0.5:
            single_img_seqs = single_image_with_bg

        ##### obtain flipped and original head pose in cam frame #####
        if self.load_dict['head_poses']:
            head_rot6d, head_rot6d_f,head_rest_vs_in_cam,head_rest_vs_f_in_cam =\
                self.get_head_rot6d_in_camframe(
                target_sub=target_sub,
                target_id=int(seq_file_ids[self.win_size]),
                target_cam_id=target_video_name[:-4],
                return_flipped=True)
        else:
            head_rot6d, head_rot6d_f = np.array([]), np.array([])

        if self.load_dict['deforms']:
            deformation_cam = self.convert_canonical_deforms_into_cam_space(
                target_frame=int(seq_file_ids[self.win_size]),
                target_sub=target_sub,
                target_cam_id=target_video_name[:-4])
            deformation_cam_f = copy.copy(deformation_cam)
            deformation_cam_f[:, 0] *= -1
            deformation_cam_f = deformation_cam_f[self.lr_head_corresp_idx]
            deformation_canonical = self.data_all_dic[target_sub]["deforms"][int(
                seq_file_ids[self.win_size])]

        else:
            deformation_cam, deformation_cam_f = np.array([]), np.array([])
            deformation_canonical = np.array([]),

        if angle != 0:
            ######## apply rotation along z axis on the deformation ########
            z_rot_transform = tu.get_RT_from_z_angle(angle=angle)

            deformation_cam = tu.apply_transform_np(
                data=deformation_cam,
                RT=z_rot_transform)
            deformation_cam_f = tu.apply_transform_np(
                data=deformation_cam_f,
                RT=z_rot_transform)
            head_rest_vs_in_cam = tu.apply_transform_np(
                data=head_rest_vs_in_cam,
                RT=z_rot_transform)
            head_rest_vs_f_in_cam = tu.apply_transform_np(
                data=head_rest_vs_f_in_cam,
                RT=z_rot_transform)
            rh_vs_in_cam = tu.apply_transform_np(
                data=rh_vs_in_cam,
                RT=z_rot_transform)
            rh_vs_in_cam_f = tu.apply_transform_np(
                data=rh_vs_in_cam_f,
                RT=z_rot_transform)
            head_vs_in_cam = tu.apply_transform_np(
                data=head_vs_in_cam,
                RT=z_rot_transform)
            head_vs_in_cam_f = tu.apply_transform_np(
                data=head_vs_in_cam_f,
                RT=z_rot_transform)
            rh_keys_in_cam = tu.apply_transform_np(
                data=rh_keys_in_cam,
                RT=z_rot_transform)
            head_keys_in_cam = tu.apply_transform_np(
                data=head_keys_in_cam,
                RT=z_rot_transform)

            head_rotmat = tu.rot6d2rotmat_np(head_rot6d)
            head_rotmat_f = tu.rot6d2rotmat_np(head_rot6d_f)
            z_rot_R = copy.copy(z_rot_transform)[:3, :3]
            head_rotmat = np.dot(z_rot_R, head_rotmat)
            head_rotmat_f = np.dot(z_rot_R, head_rotmat_f)

            head_rot6d = tu.rotmat2rot6d_np(head_rotmat)
            head_rot6d_f = tu.rotmat2rot6d_np(head_rotmat_f)


        head_con_labels = self.data_all_dic[target_sub]["head_cons"][target_frame]
        rh_con_labels = self.data_all_dic[target_sub]["rh_cons"][target_frame]
        head_con_labels_f = self.data_all_dic[target_sub]["head_cons_f"][target_frame]
        lh_con_labels = self.data_all_dic[target_sub]["lh_cons"][target_frame]
        head_dists = self.data_all_dic[target_sub]["head_dists"][target_frame]
        head_dists_f = self.data_all_dic[target_sub]["head_dists_f"][target_frame]
        rh_dists = self.data_all_dic[target_sub]["rh_dists"][target_frame]
        lh_dists = self.data_all_dic[target_sub]["lh_dists"][target_frame]
        rh_poses_cano = self.data_all_dic[target_sub]["rh_poses_cano"][target_frame]
        rh_keys_cano = self.data_all_dic[target_sub]["rh_keys_cano"][target_frame]
        lh_keys_cano = self.get_lh_keys_cano(
            rh_keys_cano=rh_keys_cano,
            target_sub=target_sub,
            target_id=int(seq_file_ids[self.win_size]),
            target_cam_id=target_video_name[:-4]).astype(np.float32)
        """
        rh_bb = np.load(self.data_all_dic[target_sub]["bb_hand_base_path"] +
                        target_video_name[:-4]+"/"+str(target_frame).zfill(5) +
                        ".npy").astype(np.float32)
        head_bb = np.load(self.data_all_dic[target_sub]["bb_head_base_path"] +
                          target_video_name[:-4]+"/"+str(target_frame).zfill(5) +
                          ".npy").astype(np.float32)

        rh2head_bb = head_bb-rh_bb
        rh_bb = dp.normalize_bb(
            bb=rh_bb, img_w=self.img_wh[0], img_h=self.img_wh[1]).astype(np.float32)
        head_bb = dp.normalize_bb(
            bb=head_bb, img_w=self.img_wh[0], img_h=self.img_wh[1]).astype(np.float32)
        rh2head_bb = dp.normalize_bb(
            bb=rh2head_bb, img_w=self.img_wh[0], img_h=self.img_wh[1]).astype(np.float32)
        if rh_bb[0] < 0 or head_bb[0] < 0:
            valid_flag = 0
        """
        if self.load_dict['rh_keys2d']:
            rh_keys_data = self.data_all_dic[target_sub]["rh_keys2d"]\
                [target_video_name[:-4]][target_frame].astype(np.float32)

            rh_keys_confs = rh_keys_data[:, -1]
            rh_keys = rh_keys_data[:, :-1]
            lh_keys_confs = copy.copy(rh_keys_confs)
            if self.key_flipping:
                rh_keys = dp.key_flipper(w=self.img_wh[0],
                                         h=self.img_wh[1],
                                         keys=rh_keys,
                                         flip_h=False,
                                         flip_v=True)

            lh_keys = dp.key_flipper(w=self.img_wh[0],
                                     h=self.img_wh[1],
                                     keys=copy.copy(rh_keys),
                                     flip_h=False,
                                     flip_v=True)

        else:
            rh_keys, rh_keys_confs = np.zeros((21, 2)), np.zeros(21)
            lh_keys, lh_keys_confs = np.zeros((21, 2)), np.zeros(21)

        if self.load_dict['head_keys2d']:
            head_keys_data =self.data_all_dic[target_sub]["head_keys2d"]\
                [target_video_name[:-4]][target_frame].astype(np.float32)
            head_keys_confs = head_keys_data[:, -1]
            head_keys = head_keys_data[:, :-1]

            head_keys_f = dp.face_key_flipper_wo_inversion(
                w=self.img_wh[0], h=self.img_wh[1], keys=head_keys,
                face_lr_corresp_idx=dp.face_lr_corresp_idx)
            head_keys_confs_f = copy.copy(
                head_keys_confs)  # [dp.face_lr_corresp_idx]
            nose_ids = [27, 28, 29, 30]
            face_center_key = np.average(head_keys[nose_ids],
                                         axis=0).reshape(1, 2)  # .astype(int)
            face_center_key_f = np.average(head_keys_f[nose_ids],
                                           axis=0).reshape(1, 2)  # .astype(int)

        else:
            head_keys, head_keys_confs = np.zeros((68, 2)), np.zeros(68)

        head_shape = self.data_all_dic[target_sub]["head_shape"].reshape(-1)
        rh_shape = self.data_all_dic[target_sub]["rh_shape"].reshape(-1)

        rr_head_keys = head_keys-face_center_key
        rr_head_keys_f = head_keys_f-face_center_key_f
        rr_rh_keys = rh_keys-face_center_key
        rr_lh_keys = lh_keys-face_center_key_f

        if angle != 0:

            rr_head_keys = dp.rotate_2d_data(
                rr_head_keys, angle_degrees=-angle)
            rr_head_keys_f = dp.rotate_2d_data(
                rr_head_keys_f, angle_degrees=-angle)
            rr_rh_keys = dp.rotate_2d_data(
                rr_rh_keys, angle_degrees=-angle)
            rr_lh_keys = dp.rotate_2d_data(
                rr_lh_keys, angle_degrees=-angle)

        rh_keys = dp.normalize_keys_single(rh_keys,
                                           w=self.img_wh[0],
                                           h=self.img_wh[1])
        lh_keys = dp.normalize_keys_single(lh_keys,
                                           w=self.img_wh[0],
                                           h=self.img_wh[1])
        head_keys = dp.normalize_keys_single(head_keys,
                                             w=self.img_wh[0],
                                             h=self.img_wh[1])
        head_keys_f = dp.normalize_keys_single(head_keys_f,
                                               w=self.img_wh[0],
                                               h=self.img_wh[1])
        face_center_key = dp.normalize_keys_single(face_center_key,
                                                   w=self.img_wh[0],
                                                   h=self.img_wh[1])
        face_center_key_f = dp.normalize_keys_single(face_center_key_f,
                                                     w=self.img_wh[0],
                                                     h=self.img_wh[1])

        rr_head_keys = dp.normalize_keys_single(rr_head_keys,
                                                w=self.img_wh[0],
                                                h=self.img_wh[1])
        rr_head_keys_f = dp.normalize_keys_single(rr_head_keys_f,
                                                  w=self.img_wh[0],
                                                  h=self.img_wh[1])
        rr_rh_keys = dp.normalize_keys_single(rr_rh_keys,
                                              w=self.img_wh[0],
                                              h=self.img_wh[1])
        rr_lh_keys = dp.normalize_keys_single(rr_lh_keys,
                                              w=self.img_wh[0],
                                              h=self.img_wh[1])

        head_vs_in_cam_deformed = head_vs_in_cam - deformation_cam
        head_vs_in_cam_deformed_f = head_vs_in_cam_f - deformation_cam_f

        if self.mode == "train" and self.multiscale:
            scale = random.uniform(self.scale_min, self.scale_max)
            aug = iaa.Affine(scale=scale)
            img_np = single_img_seqs.numpy()
            img_np = img_np.transpose(1, 2, 0)

            iaa_rh_kps = [Keypoint(x, y) for x, y in rh_keys_proj_single * img_np.shape[0]]
            iaa_head_kps = [Keypoint(x, y) for x, y in head_keys_proj_single * img_np.shape[0]]
            iaa_rh_kps_oi = KeypointsOnImage(iaa_rh_kps, shape=img_np.shape)
            iaa_head_kps_oi = KeypointsOnImage(iaa_head_kps, shape=img_np.shape)
            iaa_rh_vs = [Keypoint(x, y) for x, y in rh_vs_proj_single * img_np.shape[0]]
            iaa_head_vs = [Keypoint(x, y) for x, y in head_vs_proj_single * img_np.shape[0]]
            iaa_rh_vs_oi = KeypointsOnImage(iaa_rh_vs, shape=img_np.shape)
            iaa_head_vs_oi = KeypointsOnImage(iaa_head_vs, shape=img_np.shape)

            img_np = aug.augment_image(img_np)

            iaa_rh_kps_oi = aug.augment_keypoints(iaa_rh_kps_oi)
            iaa_head_kps_oi = aug.augment_keypoints(iaa_head_kps_oi)
            iaa_rh_vs_oi = aug.augment_keypoints(iaa_rh_vs_oi)
            iaa_head_vs_oi = aug.augment_keypoints(iaa_head_vs_oi)

            single_img_seqs = torch.from_numpy(img_np.transpose(2, 0, 1))

            rh_keys_proj_single = iaa_rh_kps_oi.to_xy_array() / img_np.shape[0]
            head_keys_proj_single = iaa_head_kps_oi.to_xy_array() / img_np.shape[0]
            rh_vs_proj_single = iaa_rh_vs_oi.to_xy_array() / img_np.shape[0]
            head_vs_proj_single = iaa_head_vs_oi.to_xy_array() / img_np.shape[0]

            rh_keys_proj_single = torch.from_numpy(rh_keys_proj_single)
            head_keys_proj_single = torch.from_numpy(head_keys_proj_single)
            rh_vs_proj_single = torch.from_numpy(rh_vs_proj_single)
            head_vs_proj_single = torch.from_numpy(head_vs_proj_single)



        


        return_dict = {}

        return_dict["dataset_name"] = "decaf"
        return_dict["data_index"] = idx

        # return_dict["rh_vs"] = rh_vs
        # return_dict["head_vs"] = head_vs
        return_dict["single_img_seqs"] = single_img_seqs
        return_dict["rh_keys_proj_single"] = rh_keys_proj_single
        return_dict["head_keys_proj_single"] = head_keys_proj_single
        return_dict["rh_vs_proj_single"] = rh_vs_proj_single
        return_dict["head_vs_proj_single"] = head_vs_proj_single

        # return_dict["cam_params"] = cam_params
        # return_dict["rh_vs_proj"] = rh_vs_proj
        # return_dict["head_vs_proj"] = head_vs_proj
        # return_dict["rh_keys_proj"] = rh_keys_proj
        # return_dict["head_keys_proj"] = head_keys_proj
        return_dict["head_vs_in_cam_deformed"] = torch.from_numpy(head_vs_in_cam_deformed)
        # return_dict["head_vs_in_cam_deformed_f"] = head_vs_in_cam_deformed_f
        return_dict["deformation_cam"] = torch.from_numpy(deformation_cam)
        # return_dict["deformation_cam_f"] = deformation_cam_f
        return_dict["head_vs_in_cam"] = torch.from_numpy(head_vs_in_cam)
        # return_dict["head_vs_in_cam_f"] = head_vs_in_cam_f
        return_dict["rh_vs_in_cam"] = torch.from_numpy(rh_vs_in_cam)
        # return_dict["rh_vs_in_cam_f"] = rh_vs_in_cam_f
        return_dict["rh_keys_in_cam"] = torch.from_numpy(rh_keys_in_cam)
        return_dict["head_keys_in_cam"] = torch.from_numpy(head_keys_in_cam)

        # return_dict["target_frame"] = target_frame
        # return_dict["rh_img_seqs"] = rh_img_seqs.squeeze()
        # return_dict["head_img_seqs"] = head_img_seqs.squeeze()
        # return_dict["lh_img_seqs"] = lh_img_seqs.squeeze()
        # return_dict["head_img_f_seqs"] = head_img_f_seqs.squeeze()
        return_dict["rh_con_labels"] = torch.from_numpy(rh_con_labels)
        # return_dict["lh_con_labels"] = lh_con_labels
        return_dict["head_con_labels"] = torch.from_numpy(head_con_labels)
        # return_dict["head_con_labels_f"] = head_con_labels_f
        # return_dict["deformation_cam"] = deformation_cam
        # return_dict["deformation_cam_f"] = deformation_cam_f
        # return_dict["deformation_canonical"] = deformation_canonical
        # return_dict["head_rot6d"] = head_rot6d
        # return_dict["head_rot6d_f"] = head_rot6d_f
        # return_dict["head_dists"] = head_dists
        # return_dict["head_dists_f"] = head_dists_f
        # return_dict["rh_dists"] = rh_dists
        # return_dict["lh_dists"] = lh_dists
        # return_dict["rh_poses_cano"] = rh_poses_cano
        # return_dict["rh_keys_cano"] = rh_keys_cano
        # return_dict["lh_keys_cano"] = lh_keys_cano
        # return_dict["head_keys"] = head_keys
        # return_dict["head_keys_confs"] = head_keys_confs
        # return_dict["head_keys_f"] = head_keys_f
        # return_dict["head_keys_confs_f"] = head_keys_confs_f
        # return_dict["rh_keys"] = rh_keys
        # return_dict["rh_keys_confs"] = rh_keys_confs
        # return_dict["lh_keys"] = lh_keys
        # return_dict["lh_keys_confs"] = lh_keys_confs
        # return_dict["rr_head_keys"] = rr_head_keys
        # return_dict["rr_head_keys_f"] = rr_head_keys_f
        # return_dict["rr_rh_keys"] = rr_rh_keys
        # return_dict["rr_lh_keys"] = rr_lh_keys
        # return_dict["head_shape"] = head_shape
        # return_dict["rh_shape"] = rh_shape
        # return_dict["head_rest_vs_in_cam"] = head_rest_vs_in_cam
        # return_dict["head_rest_vs_f_in_cam"] = head_rest_vs_f_in_cam

        return_dict["mode"] = self.mode
        return_dict["sub_id"] = target_sub
        return_dict["cam_id"] = target_video_name[:-4]
        return_dict["frame_id"] = seq_file_ids[0]

        # return_dict["mediapipe_face_kp"] = mediapipe_face_kp
        # return_dict["mediapipe_hand_kp"] = mediapipe_hand_kp
        
        if self.load_dict["ref_vs"]:
            return_dict["rh_ref_vs"] = rh_ref_vs
            return_dict["head_ref_vs"] = head_ref_vs

        return_dict["rh_betas"] = rh_betas
        return_dict["rh_transl"] = rh_transl
        return_dict["rh_rot"] = rh_rot
        return_dict["rh_pose"] = rh_pose
        return_dict["face_shape"] = head_shape_params
        return_dict["face_exp"] = head_expression_params
        return_dict["face_pose"] = head_pose_params
        return_dict["face_rot"] = head_rotation
        return_dict["face_transl"] = head_transl

        return_dict["has_2d_kp"] = 1
        return_dict["has_3d_kp"] = 1
        return_dict["has_3d_mesh"] = 1
        return_dict["has_depth"] = 0
        return_dict["has_contact"] = 1
        return_dict["has_params"] = 1
        return_dict["has_deform"] = 1
        return_dict["depth_map"] = torch.zeros(224,224)

        # return_dict["contact_matrix"] = contact_matrix

        # for key, value in return_dict.items():
        #     if hasattr(value, 'shape'):
        #         print(key, value.shape)
        #     else:
        #         print(key, value)

        # input("decaf shape")

        return return_dict
