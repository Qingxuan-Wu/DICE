import argparse
import torch.utils.data  
from  simple_data_loader import DeformLoaderOffLine 
import sys 
import numpy as np 
import matplotlib.pyplot as plt 
def cat_images(img_seq ):
    cat_img=None
    for i in range(len(img_seq)):
        if i==0:
            cat_img=img_seq[i]
        else:
            cat_img = np.concatenate((cat_img,img_seq[i]),axis=1)
    
    return cat_img 

def training( ): 
    
    path_dict={"dataset":dataset_path, 
               "dataset_image":dataset_image_path,
               "aug_path":dataset_path+"/assets/aug_back_ground/", }
    
    #MP = MeshProcessing(dataset_path+"/assets/default_mesh.ply")

    train_dataset = DeformLoaderOffLine(base_path=dataset_path,
                                mode="train",
                                win_size=args.win_size,
                                load_dict=load_dict,
                                subs=train_subs, 
                                n_pca=args.n_pca, 
                                img_wh=args.img_wh,
                                dyn_iter=args.dyn_iter,
                                deform_thresh=args.deform_thresh,
                                cam_space_deform=args.cam_space_deform,
                                batch_size=args.batch_size,
                                back_aug=args.back_aug,
                                valid_vid_ids=valid_vid_ids,
                                n_batch_per_epoch=args.train_n_batch_per_epoch,
                                img_rotate_aug=args.imgrot_aug,
                                path_dict=path_dict)

    
    data_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True,
                                               drop_last=True)
    
    for i, data in enumerate(data_loader):
        rh_img_seqs = data['rh_img_seqs']
        _head_img_seqs = data['head_img_seqs']
        lh_img_seqs = data['lh_img_seqs']


        print(rh_img_seqs.shape)
        rh_img_seqs=rh_img_seqs[0].permute(0,2,3,1).cpu().numpy() 
        rh_img_seqs=cat_images(rh_img_seqs)
        _head_img_seqs=_head_img_seqs[0].permute(0,2,3,1).cpu().numpy() 
        _head_img_seqs=cat_images(_head_img_seqs) 
        rh_img_seqs = (rh_img_seqs - rh_img_seqs.min()) / (rh_img_seqs.max() - rh_img_seqs.min())
        lh_img_seqs=lh_img_seqs[0].permute(0,2,3,1).cpu().numpy()
        lh_img_seqs=cat_images(lh_img_seqs)
        lh_img_seqs = (lh_img_seqs - lh_img_seqs.min()) / (lh_img_seqs.max() - lh_img_seqs.min())
        _head_img_seqs = (_head_img_seqs - _head_img_seqs.min()) / (_head_img_seqs.max() - _head_img_seqs.min())

        plt.imsave(f"test_images/vis_{i}.png", np.concatenate((rh_img_seqs,_head_img_seqs, lh_img_seqs), axis=0)[:, :, ::-1])
        print("index:", i)
        plt.axis('off')  # Turn off axis numbers
        plt.show() 
        # sys.exit()
 
                 
 
 
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configuratoins') 
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--win_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cam_space_deform', type=int, default=1) 
    parser.add_argument('--back_aug', type=int, default=1)
    parser.add_argument('--imgrot_aug', type=int, default=1)
    parser.add_argument('--img_wh', type=tuple, default=(1920,1080))
    parser.add_argument('--max_epoch', type=int, default=1500)
    parser.add_argument('--n_pca', type=int, default=45)
    parser.add_argument('--pre_train', type=int, default=199)
    parser.add_argument('--dist_thresh', type=float, default=0.1)
    parser.add_argument('--hidden', type=int, default=5023*1)
    parser.add_argument('--dyn_iter', type=int, default=200)#args.num_workers
    parser.add_argument('--deform_thresh', type=int, default=0)
    parser.add_argument('--flipping', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=12) 
    parser.add_argument('--train_n_batch_per_epoch', type=int, default=1000)
    parser.add_argument('--valid_n_batch_per_epoch', type=int, default=1)
    parser.add_argument('--data_path', type=str, default="/mnt/d/DecafDataset/") 
    parser.add_argument('--image_data_path', type=str, default="/home/sshimada/remote/PhysicsHuman4/static00/DecafDataset_images/")   
    args = parser.parse_args()
    
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    load_dict={  'deforms': 1, 'head_cons': 1, 'head_cons_f': 1, 'rh_cons': 1,
                 'lh_cons': 1, 'head_dists': 0, 'head_dists_f': 0, 'rh_dists': 0,
                 'lh_dists': 0, 'head_poses': 1,   'head_img': 1,
                 'head_img_f': 1, 'rh_img': 1, 'lh_img': 1, 'rh_pose_cano': 1,
                 'head_keys2d': 1,'rh_keys_cano':1, 'rh_keys2d': 1,}
    print('Using: ', device)
    valid_vid_ids = ['084', '100', '102', '108', '110', '111', '121', '122']
    train_subs = ['S2',"S4", "S5", "S7", "S8"]  
    dataset_path = args.data_path
    dataset_image_path =args.image_data_path
    training()


    """
    target_frame torch.Size([2])
    rh_img_seqs torch.Size([2, 5, 3, 224, 224])
    head_img_seqs torch.Size([2, 5, 3, 224, 224])
    lh_img_seqs torch.Size([2, 5, 3, 224, 224])
    head_img_f_seqs torch.Size([2, 5, 3, 224, 224])
    rh_con_labels torch.Size([2, 778])
    lh_con_labels torch.Size([2, 778])
    head_con_labels torch.Size([2, 5023])
    head_con_labels_f torch.Size([2, 5023])
    deformation_cam torch.Size([2, 5023, 3])
    deformation_cam_f torch.Size([2, 5023, 3])
    deformation_canonical torch.Size([2, 5023, 3])
    head_rot6d torch.Size([2, 6])
    head_rot6d_f torch.Size([2, 6])
    head_dists torch.Size([2])
    head_dists_f torch.Size([2])
    rh_dists torch.Size([2])
    lh_dists torch.Size([2])
    rh_poses_cano torch.Size([2, 51])
    rh_keys_cano torch.Size([2, 21, 3])
    lh_keys_cano torch.Size([2, 21, 3])
    head_keys torch.Size([2, 68, 2])
    head_keys_confs torch.Size([2, 68])
    head_keys_f torch.Size([2, 68, 2])
    head_keys_confs_f torch.Size([2, 68])
    rh_keys torch.Size([2, 21, 2])
    rh_keys_confs torch.Size([2, 21])
    lh_keys torch.Size([2, 21, 2])
    lh_keys_confs torch.Size([2, 21])
    rr_head_keys torch.Size([2, 68, 2])
    rr_head_keys_f torch.Size([2, 68, 2])
    rr_rh_keys torch.Size([2, 21, 2])
    rr_lh_keys torch.Size([2, 21, 2])
    head_shape torch.Size([2, 100])
    rh_shape torch.Size([2, 10])
    head_rest_vs_in_cam torch.Size([2, 5023, 3])
    head_rest_vs_f_in_cam torch.Size([2, 5023, 3])
    """