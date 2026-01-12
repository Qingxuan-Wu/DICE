import numpy as np
import torch 
from typing import Tuple 
TT =torch.Tensor# TypeVar('TT', torch.Tensor )

def normalize_keys_batch(keys,w,h):
    """
    keys:Bxn_viewsxNx2
    """
    keys[:,:,:,0]/=w
    keys[:,:,:,1]/=h
    return keys

def rigid_transform_with_scaling(R,T,s,data):
    """
    R:Bx3x3; T:Bx3, data:BxNx3, s:Bx1
    """ 
    B,N,_=data.shape
    centroid = data.clone().detach().mean(1) 
    center_data = (s.view(B,1,1))*(data-centroid.view(B,1,3)) 
    data_transformed_T = torch.bmm(R, torch.transpose(center_data,2,1))+centroid.detach().view(B,3,1)  + T.view(B,3,1) 
    return data_transformed_T
  
def multiview_projection_batch(Ps,points,device=None):
    """
    Ps:B x n_view x 3 x 4
    points: B x n_points x 3
    returns: B x n_view x n_points x 2
    """
    #print(Ps.shape,points.shape)
    B,n_view,dim1,dim2=Ps.shape
    _,n_p,d = points.shape
    points=points.view(B,1,n_p,d).expand(-1,n_view,-1,-1)# B x n_views x n_points x 3
    device = Ps.device
    points_homo=torch.cat((points,torch.ones(B,n_view,n_p,1).to(device)),3)  
    proj_homo = torch.transpose(torch.bmm(Ps.reshape(-1,dim1,dim2),torch.transpose(points_homo.reshape(-1,n_p,d+1),2,1)),2,1)
    
    proj_homo = proj_homo/(proj_homo[:,:,-1].view(B*n_view,n_p,1)) 
    return proj_homo[:,:,:-1].reshape(B,n_view,n_p,2)
  
  

def mano_tip_shortner(j_3ds,
                      extending_idx=[(15, 16), (3, 17),
                                     (6, 18), (12, 19), (9, 20)],
                      multiplier=0.8):  # 1.1
    """
    data: B x n_j x 3
    """
    for idx in extending_idx:
        vec = multiplier*(j_3ds[:, idx[1]]-j_3ds[:, idx[0]])
        j_3ds[:, idx[1]] = j_3ds[:, idx[0]]+vec
    return j_3ds
#  extending_mano_idx = [(15,16),(3,17),(6,18),(12,19),(9,20)]

def projection_np(K,data, ): 
    """
    K:3x3; data:Nx3
    """ 
    keys_proj = np.dot(K,data.T).T
    keys_proj = (keys_proj/(keys_proj[:,-1].reshape(-1,1)))[:,:-1]  
    return keys_proj

def mano_forwarding(h_model,
                    betas,
                    transl,
                    rot,
                    pose,
                    device=None, 
                    Ps=None,
                    img_size=None ,
                    tip_shortner=True,
                    return_2d=False,
                    return_2d_verts=False):
  """_summary_

  Args:
      h_model : mano hand model 
      betas (*,10): model parameter
      transl (*,3): model parameter
      rot (*,3): model parameter
      pose (*,n_pca): model parameter
      device  : device type gpu/cpu in Pytorch format
      Ps (*,n_view,3,4): projection matrices for each view. Defaults to None.
      img_size (2): image size (w,h). Defaults to None 
      return_2d=True.

  Returns:
      list:[3D hand keypoints (*,21,3), optionally 2D hand keypoints (*,21,2)]
  """ 
#   print(f"rot.shape: {rot.shape}")
#   print(f"pose.shape: {pose.shape}")
  h_output = h_model(
      betas=betas,
      global_orient=rot,
      hand_pose=pose,
      transl=transl,
      return_verts=True,
      return_tips=True,)
  j_3ds = h_output.joints
  if tip_shortner:
      j_3ds = mano_tip_shortner(j_3ds, multiplier=0.7)#0.7
  h_vs = h_output.vertices
  if return_2d:
      h_keys_proj = multiview_projection_batch(Ps.detach(), j_3ds, device=device)
      h_keys_proj = normalize_keys_batch(h_keys_proj, img_size[0], img_size[1])

  if return_2d:
    if return_2d_verts:
        h_vs_proj = multiview_projection_batch(Ps.detach(), h_vs, device=device)
        h_vs_proj = normalize_keys_batch(h_vs_proj, img_size[0], img_size[1])
        return [h_vs,j_3ds, h_keys_proj, h_vs_proj]
    else:
      return [h_vs,j_3ds, h_keys_proj]
  else:
      return [h_vs,j_3ds]