"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""


import os.path as op
import torch
import logging
import code
from src.utils.comm import get_world_size
from src.datasets.human_mesh_tsv import (MeshTSVDataset, MeshTSVYamlDataset)
from src.datasets.hand_mesh_tsv import (HandMeshTSVDataset, HandMeshTSVYamlDataset)

#Decaf
from src.datasets.decaf import make_decaf_dataset, DecafDataset
from torch.utils.data import Dataset
from src.datasets.decaf_and_inthewild import DeCafAndInTheWildDataset
from src.datasets.folder_dataset import FolderDataset



def build_dataset(yaml_file, args, is_train=True, scale_factor=1):
    print(yaml_file)
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        # code.interact(local=locals())
        assert op.isfile(yaml_file)
    return MeshTSVYamlDataset(yaml_file, is_train, False, scale_factor)


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_batch_data_sampler(sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, yaml_file, is_distributed=True, 
        is_train=True, start_iter=0, scale_factor=1):

    dataset = build_dataset(yaml_file, args, is_train=is_train, scale_factor=scale_factor)
    logger = logging.getLogger(__name__)
    if is_train==True:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader


#==============================================================================================

def build_hand_dataset(yaml_file, args, is_train=True, scale_factor=1):
    print(yaml_file)
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        # code.interact(local=locals())
        assert op.isfile(yaml_file)
    return HandMeshTSVYamlDataset(args, yaml_file, is_train, False, scale_factor)


def make_hand_data_loader(args, yaml_file, is_distributed=True, 
        is_train=True, start_iter=0, scale_factor=1):

    dataset = build_hand_dataset(yaml_file, args, is_train=is_train, scale_factor=scale_factor)
    logger = logging.getLogger(__name__)
    if is_train==True:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader


def make_decaf_data_loader(args, is_distributed=True, is_train=True, start_iter=0, shuffle=None):
    dataset = make_decaf_dataset(args, is_train)

    logger = logging.getLogger(__name__)
    if is_train==True:
        if shuffle is None:
            shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        if shuffle is None: 
            shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader

def make_combined_data_loader(args, is_distributed=True, is_train=True, start_iter=0, shuffle=None):
    print("using combined dataset")
    dataset = CombinedDataset(args)

    logger = logging.getLogger(__name__)
    if is_train==True:
        if shuffle is None:
            shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        if shuffle is None: 
            shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader



def make_decaf_and_inthewild_data_loader(args, is_distributed=True, is_train=True, start_iter=0, shuffle=None):
    print("using decaf and inthewild dataset (8 samples)")
    dataset = DeCafAndInTheWildDataset(args)

    logger = logging.getLogger(__name__)
    if is_train==True:
        if shuffle is None:
            shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        if shuffle is None: 
            shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader


def make_naive_decaf_data_loader(args, is_distributed=True, is_train=True, start_iter=0, shuffle=None):
    load_dict={  'single_img': 1, 'deforms': 1, 'head_cons': 1, 'head_cons_f': 1, 'rh_cons': 1,
                    'lh_cons': 1, 'head_dists': 0, 'head_dists_f': 0, 'rh_dists': 0,
                    'lh_dists': 0, 'head_poses': 1,   'head_img': 0,
                    'head_img_f': 0, 'rh_img': 0, 'lh_img': 0, 'rh_pose_cano': 1,
                    'head_keys2d': 1,'rh_keys_cano':1, 'rh_keys2d': 1, 'mediapipe': 1, "contact_matrix": 1, 'ref_vs': 1}

    # if is_train:
    #     load_dict={  'single_img': 1, 'deforms': 1, 'head_cons': 1, 'head_cons_f': 1, 'rh_cons': 1,
    #                 'lh_cons': 1, 'head_dists': 0, 'head_dists_f': 0, 'rh_dists': 0,
    #                 'lh_dists': 0, 'head_poses': 1,   'head_img': 0,
    #                 'head_img_f': 0, 'rh_img': 0, 'lh_img': 0, 'rh_pose_cano': 1,
    #                 'head_keys2d': 1,'rh_keys_cano':1, 'rh_keys2d': 1,}
    # else:
    #     load_dict={  'single_img': 1, 'deforms': 1, 'head_cons': 0, 'head_cons_f': 0, 'rh_cons': 0,
    #              'lh_cons': 0, 'head_dists': 0, 'head_dists_f': 0, 'rh_dists': 0,
    #              'lh_dists': 0, 'head_poses': 1,   'head_img': 1,
    #              'head_img_f': 1, 'rh_img': 1, 'lh_img': 1, 'rh_pose_cano': 1,
    #              'head_keys2d': 1,'rh_keys_cano':1, 'rh_keys2d': 0,}
    print('Using: ', args.device)
    if is_train:
        valid_vid_ids = ['084', '100', '102', '108', '110', '111', '121', '122']
    else:
        valid_vid_ids = ['108']
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
                                path_dict=path_dict)

    dataset = FirstElementDataset(dataset)

    logger = logging.getLogger(__name__)
    if is_train==True:
        if shuffle is None:
            shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        if shuffle is None: 
            shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader


class FirstElementDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[42]

    def __len__(self):
        return 1

def make_folder_data_loader(args, is_distributed=True, is_train=True, start_iter=0, shuffle=None, drop_last=True):
    print(f"using dataset from folder {args.input_folder}")
    dataset = FolderDataset(args.input_folder)

    logger = logging.getLogger(__name__)
    # if is_train==True:
    #     if shuffle is None:
    #         shuffle = True
    #     images_per_gpu = args.per_gpu_train_batch_size
    #     images_per_batch = images_per_gpu * get_world_size()
    #     iters_per_batch = len(dataset) // images_per_batch
    #     num_iters = iters_per_batch * args.num_train_epochs
    #     logger.info("Train with {} images per GPU.".format(images_per_gpu))
    #     logger.info("Total batch size {}".format(images_per_batch))
    #     logger.info("Total training steps {}".format(num_iters))
    # else:
    #     if shuffle is None: 
    #         shuffle = False
    #     images_per_gpu = args.per_gpu_eval_batch_size
    #     num_iters = None
    #     start_iter = 0
    if shuffle is None: 
        shuffle = False
    images_per_gpu = args.per_gpu_eval_batch_size
    num_iters = None
    start_iter = 0

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True, 
        drop_last=drop_last,
    )
    return data_loader
