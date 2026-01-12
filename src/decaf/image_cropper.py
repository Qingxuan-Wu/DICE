import sys
import os 
import argparse
from tqdm import tqdm
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
def make_dirs(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("new dir created :", save_path)
    return
  
def process( )->None:    

    for mode in modes:
      for sub in subs:
        cams = [x[:-4] for x in os.listdir(args.data_path+"/"+mode+"/videos/"+sub)]
        cams.sort()
        for cam in cams:
          print("===========",mode, sub,cam,"===========")
          vid_path = args.data_path+"/"+mode+"/videos/"+sub+"/"+cam+".mp4"
          seg_path =args.data_path+"/"+mode+"/segmentations/"+sub+"/"+cam+".mp4"
          head_bb_path = args.data_path+"/"+mode+"/head_bbs/"+sub+"/"+cam+".npy"
          rh_bb_path = args.data_path+"/"+mode+"/right_hand_bbs/"+sub+"/"+cam+".npy"
          rh_img_save_path =args.save_path+"/"+mode+"/cropped_rh_images/"+sub+"/"+cam+"/"
          head_img_save_path =args.save_path+"/"+mode+"/cropped_head_images/"+sub+"/"+cam+"/"
          rh_mask_save_path =args.save_path+"/"+mode+"/cropped_rh_masks/"+sub+"/"+cam+"/"
          head_mask_save_path =args.save_path+"/"+mode+"/cropped_head_masks/"+sub+"/"+cam+"/"

          make_dirs(rh_img_save_path)
          make_dirs(head_img_save_path)
          make_dirs(rh_mask_save_path)
          make_dirs(head_mask_save_path)
          
          ############### initializations ################
          rh_bbs = np.load(rh_bb_path).astype(int)
          head_bbs = np.load(head_bb_path).astype(int)
          cap = cv2.VideoCapture(vid_path)  
          seg_cap = cv2.VideoCapture(seg_path)  
          # imageWidth = int(cap.get(3))
          # imageHeight = int(cap.get(4))  
          length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          
          for i in tqdm(range(length)): 
              rh_bb=rh_bbs[i]
              head_bb=head_bbs[i]
              res, image = cap.read() #target_seg_cap
              res, mask = seg_cap.read()
              ret, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
              mask[mask!=0]=1  
              image*=mask
              
              ## handle invalid bbs ##
              if rh_bb[0]<0 or rh_bb[1]<0 or rh_bb[2]<0 or rh_bb[3]<0 or rh_bb[0]>1920 or rh_bb[1]>1080 or rh_bb[2]>1920 or rh_bb[3]>1080:
                  
                  rh_bb=[0,0,1920,1080] 
              if head_bb[0]<0 or head_bb[1]<0 or head_bb[2]<0 or head_bb[3]<0 or head_bb[0]>1920 or head_bb[1]>1080 or head_bb[2]>1920 or head_bb[3]>1080:
                  
                  head_bb=[0,0,1920,1080] 
              ## crop and save images ##
              rh_image = image[rh_bb[1]:rh_bb[3],rh_bb[0]:rh_bb[2]]
              head_image = image[head_bb[1]:head_bb[3],head_bb[0]:head_bb[2]]
              rh_mask = mask[rh_bb[1]:rh_bb[3],rh_bb[0]:rh_bb[2]]
              head_mask = mask[head_bb[1]:head_bb[3],head_bb[0]:head_bb[2]]
              
               
              cv2.imwrite(rh_img_save_path+str(i).zfill(5)+".jpg",rh_image)
              cv2.imwrite(head_img_save_path+str(i).zfill(5)+".jpg",head_image)
              cv2.imwrite(rh_mask_save_path+str(i).zfill(5)+".png",rh_mask)
              cv2.imwrite(head_mask_save_path+str(i).zfill(5)+".png",head_mask)
          cap.release() 
          seg_cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configuratoins') 
    parser.add_argument('--data_path', type=str, default="/mnt/d/DecafDataset/") 
    parser.add_argument('--save_path', type=str, default="/mnt/d/DecafDataset_images/") 
    args = parser.parse_args()
 
    modes =  ["test","train" ]#, 
    subs =  ["S1" ,"S2","S3", "S4", "S5", "S6","S7", "S8"]
  
    process() 