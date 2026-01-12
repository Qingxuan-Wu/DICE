import cv2,torch
import numpy as np

def get_image(cap,cap_seg=None,frame_id=None): 
    #ret, img = cap.read() 
    # get image specified by frame_id
    
    if frame_id is not None:
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, img = cap.read()
    
    
    
    img = cv2.putText( img,#cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                          str(frame_id),(100,100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          4, 
                          (0, 255, 0), 
                          2, 
                          cv2.LINE_AA)
    #img = np.concatenate((img,seg_img),axis=1)
    if cap_seg is not None:
      if frame_id is not None:
        cap_seg.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
      ret, seg_img = cap_seg.read()
      seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
      seg_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
      return np.array(img),np.array(seg_img)
    else:
      return np.array(img),None
  
def keypoint_overlay(kyes_2d,img,c=(0,255,0), radius=1): 
  
  [cv2.circle(img,(joint[0],joint[1]), radius, c, -1) for joint in kyes_2d]
  return img

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
      
      
def denormalize_keys_batch(keys,w,h):
    """
     keys:Bxn_viewsxNx2
    """
    keys[:, :,0]*=w
    keys[:, :,1]*=h
    return keys

def denormalize_keys(keys,w,h):
  """
    keys:Bxn_viewsxNx2
  """
  keys[:,0]*=w
  keys[:,1]*=h
  return keys